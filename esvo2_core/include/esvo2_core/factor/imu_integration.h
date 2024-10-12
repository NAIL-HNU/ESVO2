#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <esvo2_core/factor/utility.h>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

namespace esvo2_core
{
namespace factor
{
enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

class IntegrationBase
{
public:
    IntegrationBase() = delete;
    IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg, const Eigen::Vector3d &optimal_g)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg}, G{optimal_g},
          jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance{Eigen::Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, begin_time{0.0}, last_time{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}

    {
        ACC_N = 4.4011810950619089e-02;
        ACC_W = 7.6577511511273218e-04;
        GYR_N = 2.9197115508062086e-03;
        GYR_W = 3.2163500109799395e-05;
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }

    void initialization(Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
    }

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    void pop()
    {
        if (dt_buf.size() <= 0)
            return;
        acc_0 = acc_buf[0];
        gyr_0 = gyr_buf[0];
        dt_buf.erase(dt_buf.begin());
        acc_buf.erase(acc_buf.begin());
        gyr_buf.erase(gyr_buf.begin());
        if (dt_buf.size() != 0)
            begin_time = begin_time + dt_buf[0];
    }

    void getPose(double last_TS_time, double cur_TS_time, bool compute_t, double point_cloud_time)
    {
        if (acc_buf.size() <= 0)
            return;
        if (!compute_t)
        {
            while (begin_time < last_TS_time)
            {
                pop();
            }

            dt_buf[0] = begin_time - last_TS_time;
            double all_time = 0;
            int i = 0;
            for (i = 0; i < dt_buf.size(); i++)
            {
                all_time += dt_buf[i];
                if (all_time > (cur_TS_time - last_TS_time))
                    break;
            }

            if (i == dt_buf.size())
            {
                repropagate(linearized_ba, linearized_bg);
                i--;
            }
            else
            {
                dt_buf[i] = dt_buf[i] - (all_time - (cur_TS_time - last_TS_time));
                begin_time += (all_time - (cur_TS_time - last_TS_time));
                repropagate(linearized_ba, linearized_bg, i + 1);
            }
        }
        else
        {
            int begin_i = 0;
            int end_i = 0;
            while (begin_time < point_cloud_time)
            {
                pop();
            }
            double curr_time = begin_time;
            while (curr_time < last_TS_time)
            {
                curr_time = curr_time + dt_buf[begin_i];
                begin_i++;
            }

            double all_time = 0;
            int i = begin_i;
            for (i = begin_i; i < dt_buf.size(); i++)
            {
                all_time += dt_buf[i];
                if (all_time > (cur_TS_time - last_TS_time))
                    break;
            }
            end_i = i;
            if (i == dt_buf.size())
            {
                repropagate(linearized_ba, linearized_bg, begin_i, end_i);
                i--;
            }
            else
            {
                dt_buf[i] = dt_buf[i] - (all_time - (cur_TS_time - last_TS_time));
                begin_time += (all_time - (cur_TS_time - last_TS_time));
                repropagate(linearized_ba, linearized_bg, begin_i, end_i);
            }
        }
    }

    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg, int size)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        if (size > static_cast<int>(dt_buf.size()))
            size = static_cast<int>(dt_buf.size());
        for (int i = 0; i < size; i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg, const int begin_i, const double end_i)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = begin_i; i < end_i; i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    void update_v(double point_cloud_time, double last_TS_time)
    {
        if (acc_buf.size() <= 0)
            return;
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        while (begin_time < t_v_last_mapping.first)
        {
            pop();
        }
        int i;
        for (i = 0; i < dt_buf.size(); i++)
        {
            sum_dt += dt_buf[i];
            if (sum_dt > (last_TS_time - point_cloud_time))
                break;
        }
        Eigen::Vector3d temp_delta_p, temp_delta_v;
        Eigen::Quaterniond temp_delta_q;
        temp_delta_q.setIdentity();
        temp_delta_p.setZero();
        temp_delta_v.setZero();
        for (; i > 0; i--)
        {
            Eigen::Vector3d un_acc_0 = delta_q * (acc_0 - linearized_ba);
            Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr_buf[i]) - linearized_bg;
            temp_delta_q = delta_q * Eigen::Quaterniond(1, un_gyr(0) * dt_buf[i] / 2, un_gyr(1) * dt_buf[i] / 2, un_gyr(2) * dt_buf[i] / 2);
            Eigen::Vector3d un_acc_1 = temp_delta_q * (acc_buf[i] - linearized_ba);
            Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            temp_delta_v = delta_v + un_acc * dt_buf[i];
            i--;
        }
        t_v_last_mapping.second = t_v_last_mapping.second + temp_delta_v;
        t_v_last_mapping.first = point_cloud_time;
    }

    void midPointIntegration(double _dt,
                             const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                             const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                             const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                             const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                             Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                             Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        Eigen::Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Eigen::Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        result_delta_q = delta_q * Eigen::Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Eigen::Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;

        if (update_jacobian)
        {
            Eigen::Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Eigen::Vector3d a_0_x = _acc_0 - linearized_ba;
            Eigen::Vector3d a_1_x = _acc_1 - linearized_ba;
            Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x << 0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x << 0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x << 0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3, 3) * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * Eigen::MatrixXd::Identity(3, 3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

            Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15, 18);
            V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) = 0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * _dt;
            V.block<3, 3>(3, 9) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * _dt;
            V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) = 0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3, 3) * _dt;
            V.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3, 3) * _dt;

            // step_jacobian = F;
            // step_V = V;
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }
    }

    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Eigen::Vector3d result_delta_p;
        Eigen::Quaterniond result_delta_q;
        Eigen::Vector3d result_delta_v;
        Eigen::Vector3d result_linearized_ba;
        Eigen::Vector3d result_linearized_bg;

        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        // checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                     linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;
    }

    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg;

    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 15, 15> step_jacobian;
    Eigen::Matrix<double, 15, 18> step_V;
    Eigen::Matrix<double, 18, 18> noise;

    double sum_dt;
    double begin_time, last_time;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;
    std::pair<double, Eigen::Vector3d> t_v_last_mapping;

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;
    Eigen::Vector3d G{0, 9.8, 0};

    double ACC_N = 4.4011810950619089e-02;
    double ACC_W = 7.6577511511273218e-04;
    double GYR_N = 2.9197115508062086e-03;
    double GYR_W = 3.2163500109799395e-05;
};
} // namespace container
} // namespace esvo2_core
