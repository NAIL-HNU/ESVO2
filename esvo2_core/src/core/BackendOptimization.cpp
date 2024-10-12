#include <esvo2_core/core/BackendOptimization.h>

#define ESVO2_CORE_BACKEND_DEBUG

namespace esvo2_core
{
  namespace core
  {
    BackendOptimization::BackendOptimization(
        const CameraSystem::Ptr &camSysPtr) : camSysPtr_(camSysPtr)
    {
      // Init camera system
      RIC_ = camSysPtr_->cam_left_ptr_->T_b_c_.block<3, 3>(0, 0);
      TIC_ = camSysPtr_->cam_left_ptr_->T_b_c_.block<3, 1>(0, 3);
      T_i_c_.setIdentity();
      T_i_c_.block<3, 4>(0, 0) = camSysPtr_->cam_left_ptr_->T_b_c_;
      frame_count = 0;

      // Init Sliding Window imu
      for (int i = 0; i < (WINDOW_SIZE + 1); i++)
        pre_integrations[i] = NULL;
      
      // Init other variables
      initVsFlag = false;
      g_optimal << 0, 9.8, 0;
    }

    void BackendOptimization::setProblem(
        std::deque<DepthPointFrame> *pDepthPoints,
        TimeSurfaceHistory *pTS_history,
        ros::Publisher *pV_ba_bg_pub,
        bool bUSE_IMU)
    {
      pDepthPoints_ = pDepthPoints;
      pTS_history_ = pTS_history;
      pV_ba_bg_pub_ = pV_ba_bg_pub;
      bUSE_IMU_ = bUSE_IMU;
    }

    void BackendOptimization::sloveProblem()
    {
      TicToc t_optimization;
      // get parameters
      double para_Pose[WINDOW_SIZE + 1][7];
      double para_SpeedBias[WINDOW_SIZE + 1][9];
      double fx_ = camSysPtr_->cam_left_ptr_->P_(0, 0);
      double fy_ = camSysPtr_->cam_left_ptr_->P_(1, 1);
      double cx_ = camSysPtr_->cam_left_ptr_->P_(0, 2);
      double cy_ = camSysPtr_->cam_left_ptr_->P_(1, 2);

      ceres::Problem problem;
      ceres::LossFunction *loss_function;
      loss_function = new ceres::CauchyLoss(1.0);
      int dqvDepthPoints_size = (*pDepthPoints_).size();
      esvo2_core::container::TimeSurfaceHistory::iterator last_obs;

      // add parameter block
      for (int i = 0; i < WINDOW_SIZE + 1; i++)
      {
        ros::Time tt = (*pDepthPoints_)[dqvDepthPoints_size - WINDOW_SIZE - 1 + i].timestamp_;
        auto ts_obs = (*pTS_history_).find(tt);
        // convert vector to double for ceres
        vector2Double(ts_obs, last_obs, para_Pose, para_SpeedBias, i);
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], 7, local_parameterization);
        if (initVsFlag)
          problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
        last_obs = ts_obs;
      }

      // add residual block
      if (initVsFlag)
      {
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
          int j = i + 1;
          if (pre_integrations[j]->sum_dt > 10.0)
            continue;
          IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
          problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
      }
      problem.SetParameterBlockConstant(para_Pose[0]);
      for (int i = 1; i <= WINDOW_SIZE; i++)
        problem.SetParameterBlockConstant(para_Pose[i]);
      
      // options of ceres solver
      ceres::Solver::Options options;
      options.num_threads = 4;
      options.linear_solver_type = ceres::DENSE_QR;
      options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
      options.initial_trust_region_radius = 1e6;
      options.max_trust_region_radius = 1e6;
      options.min_trust_region_radius = 1e-5;
      options.function_tolerance = 1e-6;
      options.gradient_tolerance = 1e-6;
      options.max_num_iterations = 100;

      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
#ifdef ESVO2_CORE_BACKEND_DEBUG
if (!(Bgs[WINDOW_SIZE].norm() > 1 || Bas[WINDOW_SIZE].norm() > 1))
{
  LOG(INFO) << summary.BriefReport();
      LOG(INFO) << "Ba = " << Bas[WINDOW_SIZE].transpose();
      LOG(INFO) << "Bg = " << Bgs[WINDOW_SIZE].transpose();
}
      
#endif

      // get results from ceres
      double2Vector(para_Pose, para_SpeedBias);

      // if the solver not converged, update the Vs, Bas, Bgs from the last iteration
      if(summary.termination_type != ceres::CONVERGENCE)
      {
        for(int i = 1; i < WINDOW_SIZE + 1; i++)
        {
          Vs[i] = (Ps[i] - Ps[i - 1])/pre_integrations[i]->sum_dt;
          Bas[i] = Bas[i - 1];
          Bgs[i] = Bgs[i - 1];
        }
      }

      // Check if the solver converged
      if (initVsFlag && summary.termination_type == ceres::CONVERGENCE)
        publishVBaBg((*pDepthPoints_)[(*pDepthPoints_).size() - 1].timestamp_.toSec());

      // stereo + IMU initilization
      if (bUSE_IMU_)
      {
        bool result = false;
        Eigen::Vector3d Bg;
        Eigen::VectorXd x;
        if (!initVsFlag)
        {
          Eigen::Matrix3d RIC_temp = Eigen::Matrix3d::Zero();
          CalibrationExRotation(RIC_temp);
          solveGyroscopeBias(Bg);
          for (int i = 0; i <= WINDOW_SIZE; i++)
          {
            Bgs[i] = Bg;
            pre_integrations[i]->repropagate(Eigen::Vector3d::Zero(), Bgs[i]);
          }
          if (Bg.norm() < 2)
          {
            initVsFlag = LinearAlignment(g_optimal, x);
          }
        }
      }
    }

    void BackendOptimization::vector2Double(esvo2_core::container::TimeSurfaceHistory::iterator ts_obs,
                                            esvo2_core::container::TimeSurfaceHistory::iterator last_obs,
                                            double para_Pose[][7],
                                            double para_SpeedBias[][9],
                                            int i)
    {
      // convert the pose from Tcam to Timu
      Tcam2Timu(ts_obs, last_obs, para_Pose, i);
      if (initVsFlag)
      {
        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
      }
    }

    void BackendOptimization::Tcam2Timu(esvo2_core::container::TimeSurfaceHistory::iterator ts_obs,
                                        esvo2_core::container::TimeSurfaceHistory::iterator last_obs,
                                        double para_Pose[][7],
                                        int i)
    {
      if (i == 0)
      {
        T_wopt_window0_ = ts_obs->second.tr_.getTransformationMatrix();
        para_Pose[i][0] = 0;
        para_Pose[i][1] = 0;
        para_Pose[i][2] = 0;
        para_Pose[i][3] = 0;
        para_Pose[i][4] = 0;
        para_Pose[i][5] = 0;
        para_Pose[i][6] = 1;
      }
      else if (i != WINDOW_SIZE)
      {
        Eigen::Matrix4d T_wopt_cam = ts_obs->second.tr_.getTransformationMatrix();
        Eigen::Matrix4d T_imu = T_i_c_ * T_wopt_window0_.inverse() * T_wopt_cam * T_i_c_.inverse();
        para_Pose[i][0] = T_imu(0, 3);
        para_Pose[i][1] = T_imu(1, 3);
        para_Pose[i][2] = T_imu(2, 3);
        Eigen::Quaterniond q_imu(T_imu.block<3, 3>(0, 0));
        para_Pose[i][3] = q_imu.x();
        para_Pose[i][4] = q_imu.y();
        para_Pose[i][5] = q_imu.z();
        para_Pose[i][6] = q_imu.w();
      }
      else
      {
        Eigen::Matrix4d T_wopt_last, T_wori_last, T_wori_c, T_window0_i;
        T_wori_c = ts_obs->second.tr_ori_.getTransformationMatrix();
        T_wori_last = last_obs->second.tr_ori_.getTransformationMatrix();
        T_wopt_last = last_obs->second.tr_.getTransformationMatrix();
        T_window0_i = T_i_c_ * T_wopt_window0_.inverse() * T_wopt_last * T_wori_last.inverse() * T_wori_c * T_i_c_.inverse();

        para_Pose[i][0] = T_window0_i(0, 3);
        para_Pose[i][1] = T_window0_i(1, 3);
        para_Pose[i][2] = T_window0_i(2, 3);
        Eigen::Quaterniond q(T_window0_i.block<3, 3>(0, 0));
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();
      }
    }

    void BackendOptimization::double2Vector(double para_Pose[][7], double para_SpeedBias[][9])
    {
      Eigen::Vector3d origin_R0 = Utility::R2ypr(T_wopt_window0_.block<3, 3>(0, 0));
      Eigen::Matrix3d R_wopt_window0 = T_wopt_window0_.block<3, 3>(0, 0);
      Eigen::Vector3d origin_P0 = T_wopt_window0_.block<3, 1>(0, 3);
      for (int i = 0; i < WINDOW_SIZE + 1; i++)
      {
        Rs[i] = R_wopt_window0 * RIC_.inverse() * Eigen::Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).toRotationMatrix() * RIC_;
        Ps[i] = R_wopt_window0 * RIC_.inverse() * Eigen::Vector3d(para_Pose[i][0] - para_Pose[0][0], para_Pose[i][1] - para_Pose[0][1], para_Pose[i][2] - para_Pose[0][2]) + origin_P0;
        // if initialized Vs, Bas, Bgs, update them
        if (initVsFlag)
        {
          Vs[i] = Eigen::Vector3d(para_SpeedBias[i][0],
                                  para_SpeedBias[i][1],
                                  para_SpeedBias[i][2]);
          Bas[i] = Eigen::Vector3d(para_SpeedBias[i][3],
                                   para_SpeedBias[i][4],
                                   para_SpeedBias[i][5]);

          Bgs[i] = Eigen::Vector3d(para_SpeedBias[i][6],
                                   para_SpeedBias[i][7],
                                   para_SpeedBias[i][8]);
        }

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        if (!isOrthogonal(Rs[i]) || std::abs(Rs[i].determinant() - 1) > 1e-8)
        {
          Rs[i] = Utility::ypr2R(Utility::R2ypr(Rs[i]));
        }

        Rs[i] = fixRotationMatrix(Rs[i]);

        T.block<3, 3>(0, 0) = Rs[i];
        T.block<3, 1>(0, 3) = Ps[i];
        Transformation Tr(T);
        ros::Time tt = (*pDepthPoints_)[(*pDepthPoints_).size() - WINDOW_SIZE - 1 + i].timestamp_;
        auto ts_obs = (*pTS_history_).find(tt);
        ts_obs->second.setTransformation(Tr);
      }
    }

    void BackendOptimization::publishVBaBg(double time_v)
    {
      events_repacking_tool::V_ba_bg msg;
      Eigen::Vector3d V_temp = Rs[WINDOW_SIZE] * RIC_.transpose() * Vs[WINDOW_SIZE];
      msg.head.push_back(time_v);
      if (Bgs[WINDOW_SIZE].norm() > 1 || Bas[WINDOW_SIZE].norm() > 1)
        return;
      for (int i = 0; i < 3; i++)
      {
        msg.Vs.push_back(V_temp(i));
        msg.ba.push_back(Bas[WINDOW_SIZE](i));
        msg.bg.push_back(Bgs[WINDOW_SIZE](i));
        msg.g.push_back(g_optimal(i));
      }
      (*pV_ba_bg_pub_).publish(msg);
    }

    bool BackendOptimization::CalibrationExRotation(Eigen::Matrix3d &calib_ric_result)
    {
      vector<Eigen::Matrix3d> Rc, Rc_g, Rimu;
      for (int i = 0; i < WINDOW_SIZE; i++)
      {
        Rc.push_back(Rs[i].transpose() * Rs[i + 1]);
        Rc_g.push_back(pre_integrations[i + 1]->delta_q.toRotationMatrix());
        Rimu.push_back(pre_integrations[i + 1]->delta_q.toRotationMatrix());
      }

      Eigen::MatrixXd A((WINDOW_SIZE) * 4, 4);
      A.setZero();
      int sum_ok = 0;
      for (int i = 1; i <= WINDOW_SIZE; i++)
      {
        Eigen::Quaterniond r1(Rc[i]);
        Eigen::Quaterniond r2(Rc_g[i]);

        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG(
            "%d %f", i, angular_distance);

        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Eigen::Matrix4d L, R;

        double w = Eigen::Quaterniond(Rc[i]).w();
        Eigen::Vector3d q = Eigen::Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Eigen::Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Eigen::Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Eigen::Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3);
        Eigen::Quaterniond estimated_R(x);
        Eigen::Matrix3d ric = estimated_R.toRotationMatrix().inverse();
      }
      return false;
    }

    void BackendOptimization::solveGyroscopeBias(Eigen::Vector3d &Bgs)
    {
      Eigen::Matrix3d A;
      Eigen::Vector3d b;
      A.setZero();
      b.setZero();
      for (int i = 0; i < WINDOW_SIZE; i++)
      {
        Eigen::MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(Rs[i].transpose() * Rs[i + 1]);
        tmp_A = pre_integrations[i]->jacobian.template block<3, 3>(3, 12);
        tmp_b = 2 * (pre_integrations[i]->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
      }
      Bgs = A.ldlt().solve(b);
      ROS_WARN_STREAM("gyroscope bias initial calibration " << Bgs.transpose());
    }

    bool BackendOptimization::LinearAlignment(Eigen::Vector3d &g, Eigen::VectorXd &x)
    {
      int n_state = (WINDOW_SIZE + 1) * 3 + 3 + 1;

      Eigen::MatrixXd A{n_state, n_state};
      A.setZero();
      Eigen::VectorXd b{n_state};
      b.setZero();

      for (int i = 0; i < WINDOW_SIZE; i++)
      {
        int j = i + 1;
        Eigen::MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = pre_integrations[j]->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = Rs[i].transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = Rs[i].transpose() * (Ps[j] - Ps[i]) / 100.0;
        tmp_b.block<3, 1>(0, 0) = pre_integrations[j]->delta_p + Rs[i].transpose() * Rs[j] * TIC_ - TIC_;
        tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = Rs[i].transpose() * Rs[j];
        tmp_A.block<3, 3>(3, 6) = Rs[i].transpose() * dt * Eigen::Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = pre_integrations[j]->delta_v;

        Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();
        cov_inv.setIdentity();

        Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();
        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();
        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
      }
      A = A * 1000.0;
      b = b * 1000.0;

      x = A.ldlt().solve(b);
      double s = x(n_state - 1) / 100;
      g = x.segment<3>(n_state - 4);
      Eigen::Vector3d G{0, -9.8, 0};
      if (fabs(g.norm() - G.norm()) > 1)
        return false;

      RefineGravity(g, x);

      // save V as member Vs
      for (int i = 0; i < WINDOW_SIZE + 1; i++)
        Vs[i] = x.segment<3>(i * 3);

      // for evaluate
      Eigen::Vector3d V_cam[4], V_alianment[4];
      for (int i = 0; i < WINDOW_SIZE; i++)
      {
        V_cam[i] = (Ps[i + 1] - Ps[i]) / pre_integrations[i + 1]->sum_dt;
        Eigen::Vector3d v1 = x.segment<3>(i * 3);
        Eigen::Vector3d v2 = x.segment<3>(i * 3 + 3);
        V_alianment[i] = Rs[i] * RIC_ * (v1 + v2) / 2 / s;
      }
      return true;
    }

    Eigen::MatrixXd TangentBasis(Eigen::Vector3d &g0)
    {
      Eigen::Vector3d b, c;
      Eigen::Vector3d a = g0.normalized();
      Eigen::Vector3d tmp(0, 1, 0);
      if (a == tmp)
        tmp << 1, 0, 0;
      double d = a.transpose() * tmp;
      b = (tmp - a * d).normalized();
      c = a.cross(b);
      Eigen::MatrixXd bc(3, 2);
      bc.block<3, 1>(0, 0) = b;
      bc.block<3, 1>(0, 1) = c;
      return bc;
    }

    void BackendOptimization::RefineGravity(Eigen::Vector3d &g, Eigen::VectorXd &x)
    {
      Eigen::Vector3d g0 = g.normalized() * 9.81;
      Eigen::Vector3d lx, ly;
      int x_size = x.size();
      Eigen::VectorXd x_temp;
      int n_state = WINDOW_SIZE * 3 + 3;

      Eigen::MatrixXd A{n_state, n_state};
      A.setZero();
      Eigen::VectorXd b{n_state};
      b.setZero();

      for (int k = 0; k < 4; k++)
      {
        Eigen::MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (; i < WINDOW_SIZE; i++)
        {
          Eigen::MatrixXd tmp_A(6, 9);
          tmp_A.setZero();
          Eigen::VectorXd tmp_b(6);
          tmp_b.setZero();

          double dt = pre_integrations[i + 1]->sum_dt;

          tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
          tmp_A.block<3, 2>(0, 6) = Rs[i].transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity() * lxly;
          tmp_A.block<3, 1>(0, 8) = Rs[i].transpose() * (Ps[i + 1] - Ps[i]) / 100.0;
          tmp_b.block<3, 1>(0, 0) = pre_integrations[i]->delta_p + Rs[i].transpose() * Rs[i + 1] * TIC_ - TIC_ - Rs[i].transpose() * dt * dt / 2 * g0;

          tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
          tmp_A.block<3, 3>(3, 3) = Rs[i].transpose() * Rs[i + 1];
          tmp_A.block<3, 2>(3, 6) = Rs[i].transpose() * dt * Eigen::Matrix3d::Identity() * lxly;
          tmp_b.block<3, 1>(3, 0) = pre_integrations[i + 1]->delta_v - Rs[i].transpose() * dt * Eigen::Matrix3d::Identity() * g0;

          Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();
          cov_inv.setIdentity();

          Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
          Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

          A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
          b.segment<6>(i * 3) += r_b.head<6>();

          A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
          b.tail<3>() += r_b.tail<3>();

          A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
          A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }

        A = A * 1000;
        b = b * 1000;
        x_temp = A.ldlt().solve(b);
        Eigen::VectorXd dg = x.segment<2>(n_state - 3);
        g0 = (g0 + lxly * dg).normalized() * 9.81;
      }
      g = g0;
    }

    void BackendOptimization::slideWindow()
    {

      for (int i = 0; i < WINDOW_SIZE; i++)
      {
        Rs[i].swap(Rs[i + 1]);
        Ps[i].swap(Ps[i + 1]);
        if (bUSE_IMU_)
        {
          std::swap(pre_integrations[i], pre_integrations[i + 1]);
          Vs[i].swap(Vs[i + 1]);
          Bas[i].swap(Bas[i + 1]);
          Bgs[i].swap(Bgs[i + 1]);
        }
      }
      if (frame_count >= WINDOW_SIZE)
      {
        Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
        Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
        if (bUSE_IMU_)
        {
          Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
          Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
          Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];
          delete pre_integrations[WINDOW_SIZE];
          pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE], g_optimal};
        }
      }
      if (frame_count < WINDOW_SIZE)
        frame_count++;
    }

    bool BackendOptimization::isOrthogonal(const Eigen::Matrix3d &matrix)
    {
      Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
      Eigen::Matrix3d prod = matrix.transpose() * matrix;
      return prod.isApprox(identity, 1e-8);
    }

    Eigen::Matrix3d BackendOptimization::fixRotationMatrix(const Eigen::Matrix3d &R)
    {
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Matrix3d U = svd.matrixU();
      Eigen::Matrix3d V = svd.matrixV();
      return U * V.transpose();
    }
  } // core
} // esvo2_core
