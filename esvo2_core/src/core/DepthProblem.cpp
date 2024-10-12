#include <esvo2_core/core/DepthProblem.h>

namespace esvo2_core
{
namespace core
{
DepthProblem::DepthProblem(
  const DepthProblemConfig::Ptr &dpConfig_ptr,
  const CameraSystem::Ptr &camSysPtr) :
  factor::OptimizationFunctor<double>(1, 0),
  dpConfigPtr_(dpConfig_ptr),
  camSysPtr_(camSysPtr)
{

}

void DepthProblem::setProblem(
  Eigen::Vector2d & coor,
  Eigen::Matrix<double, 4, 4> & T_world_virtual,
  constStampedTimeSurfaceObs* pStampedTsObs,
  bool problem_lr)
{
  coordinate_      = coor;
  T_world_virtual_ = T_world_virtual;
  pStampedTsObs_ = pStampedTsObs;
  problem_lr_ = problem_lr;
  if(problem_lr)
  {
    T_last_now_ = pStampedTsObs_->second.tr_last_.getTransformationMatrix().inverse() * pStampedTsObs_->second.tr_.getTransformationMatrix();
  }

  vT_left_virtual_.clear();
  vT_left_virtual_.reserve(1);
  Eigen::Matrix<double,4,4> T_left_world = pStampedTsObs_->second.tr_.inverse().getTransformationMatrix();
  Eigen::Matrix<double,4,4> T_left_virtual = T_left_world * T_world_virtual_;
  vT_left_virtual_.push_back(T_left_virtual.block<3,4>(0,0));
  resetNumberValues(dpConfigPtr_->patchSize_X_ * dpConfigPtr_->patchSize_Y_);
}


void DepthProblem::setProblem(
  Eigen::Vector2d & coor)
{
  coordinate_  = coor;
}


int DepthProblem::operator()( const Eigen::VectorXd &x, Eigen::VectorXd & fvec ) const
{
  // TicToc t, t_in;
  // double t_all = 0, t_in_all = 0;
  size_t wx = dpConfigPtr_->patchSize_X_;
  size_t wy = dpConfigPtr_->patchSize_Y_;
  size_t patchSize = wx * wy;
  int numValid  = 0;
  Eigen::Vector2d x1_s, x2_s;
  if(!warping(coordinate_, x(0), vT_left_virtual_[0], x1_s, x2_s))
  {
    if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "l2") == 0)
      for(size_t i = 0; i < patchSize; i++)
        fvec[i] = 255;
    else if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "zncc") == 0)
      for(size_t i = 0; i < patchSize; i++)
        fvec[i] = 2 / sqrt(patchSize);
    else if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "Tdist") == 0)
      for(size_t i = 0; i < patchSize; i++)
      {
        double residual = 255;
        double weight = (dpConfigPtr_->td_nu_ + 1) / (dpConfigPtr_->td_nu_ + std::pow(residual / dpConfigPtr_->td_scale_, 2));
        fvec[i] = sqrt(weight) * residual;
      }
    else
      exit(-1);
    return numValid;
  }
  Eigen::MatrixXd tau1, tau2;
  bool pass = false;
  // t_in.tic();
  if (problem_lr_)
  {
    pass = patchInterpolation(pStampedTsObs_->second.TS_left_, x1_s, tau1, false) && patchInterpolation(pStampedTsObs_->second.TS_right_, x2_s, tau2, false);
  }
  else
  {
    pass = patchInterpolation(pStampedTsObs_->second.AA_map_, x1_s, tau1, false) && patchInterpolation(pStampedTsObs_->second.TS_last_, x2_s, tau2, false);
    // cv::Mat forshow, forshow2;
    // cv::eigen2cv(pStampedTsObs_->second.AA_map_, forshow);
    // cv::imshow("forshow", forshow);
    // cv::eigen2cv(pStampedTsObs_->second.TS_last_, forshow2);
    // cv::imshow("forshow2", forshow2);
    // cv::waitKey(1);
  }
  // t_in_all += t_in.toc(); 

  if (pass)
  {
    // compute temporal residual
    if (strcmp(dpConfigPtr_->LSnorm_.c_str(), "l2") == 0)
    {
      for(size_t y = 0; y < wy; y++)
        for(size_t x = 0; x < wx; x++)
        {
          size_t index = y * wx + x;
          fvec[index] = tau1(y,x) - tau2(y,x);
        }
    }
    else if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "zncc") == 0)
    {
      for(size_t y = 0; y < wy; y++)
        for(size_t x = 0; x < wx; x++)
        {
          size_t index = y * wx + x;
          double mu1, sigma1, mu2, sigma2;
          tools::meanStdDev(tau1, mu1, sigma1);
          tools::meanStdDev(tau2, mu2, sigma2);
          fvec[index] = ((tau1(y,x) - mu1) / sigma1 - (tau2(y,x) - mu2) / sigma2) / sqrt(patchSize);
        }
    }
    else if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "Tdist") == 0)
    {
      std::vector<double> vResidual(patchSize);
      std::vector<double> vResidualSquared(patchSize);
      double scaleSquaredTmp1 = dpConfigPtr_->td_scaleSquared_;
      double scaleSquaredTmp2 = -1.0;
      bool first_iteration = true;
      // loop for scale until it converges
      while(fabs(scaleSquaredTmp2 - scaleSquaredTmp1) / scaleSquaredTmp1 > 0.05 || first_iteration)
      {
        if(!first_iteration)
          scaleSquaredTmp1 = scaleSquaredTmp2;

        double sum_scaleSquared = 0;
        for(size_t y = 0; y < wy; y++)
        {
          for (size_t x = 0; x < wx; x++)
          {
            size_t index = y * wx + x;
            if (first_iteration)
            {
              vResidual[index] = tau1(y, x) - tau2(y, x);
              vResidualSquared[index] = std::pow(vResidual[index], 2);
            }
            if (vResidual[index] != 0)
              sum_scaleSquared += vResidualSquared[index] * (dpConfigPtr_->td_nu_ + 1) /
                              (dpConfigPtr_->td_nu_ + vResidualSquared[index] / scaleSquaredTmp1);
          }
        }
        if(sum_scaleSquared == 0)
        {
          scaleSquaredTmp2 = dpConfigPtr_->td_scaleSquared_;
          break;
        }
        scaleSquaredTmp2 = sum_scaleSquared / patchSize;
        first_iteration = false;
      }

      // assign reweighted residual
      for(size_t y = 0; y < wy; y++)
      {
        for (size_t x = 0; x < wx; x++)
        {
          size_t index = y * wx + x;
          double weight = (dpConfigPtr_->td_nu_ + 1) / (dpConfigPtr_->td_nu_ + vResidualSquared[index] / scaleSquaredTmp2);
          fvec[index] = sqrt(weight) * vResidual[index];
        }
      }
    }
    else
      exit(-1);
    numValid = 1;
  }
  else
  {
    if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "l2") == 0)
      for(size_t i = 0; i < patchSize; i++)
        fvec[i] = 255;
    else if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "zncc") == 0)
      for(size_t i = 0; i < wx * wy; i++)
        fvec[i] = 2 / sqrt(patchSize);
    else if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "Tdist") == 0)
      for(size_t i = 0; i < patchSize; i++)
      {
        double residual = 255;
        double weight = (dpConfigPtr_->td_nu_ + 1) / (dpConfigPtr_->td_nu_ + std::pow(residual / dpConfigPtr_->td_scale_, 2));
        fvec[i] = sqrt(weight) * residual;
      }
    else
      exit(-1);
  }
  return numValid;
}


int DepthProblem::df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
{
  Eigen::Matrix<double, 1, 2> jacobian_grad;
  Eigen::Vector2d x1_s, x2_s;
  Eigen::MatrixXd tau_u, tau_v;
  double fjac_temp = 0;
  int wx = dpConfigPtr_->patchSize_X_;
  int wy = dpConfigPtr_->patchSize_Y_;
  int width = camSysPtr_->cam_left_ptr_->width_;
  int height = camSysPtr_->cam_left_ptr_->height_;
  Eigen::Vector3d p_l, p_r, p_l_unit;
  Eigen::Matrix<double, 2, 3> jacobian_u_q;
  Eigen::Vector3d jacobian_q_lamda;

  int numValid = 0;
  bool warping_flag = warping(coordinate_, x(0), vT_left_virtual_[0], x1_s, x2_s);
  if (!warping_flag)
  {
    for(int i = 0; i < fjac.rows(); i++)
      fjac(i, 0) = 0;
    return numValid;
  }
  bool pass = false;
  if(problem_lr_)
    pass = patchInterpolation(pStampedTsObs_->second.dTS_du_right_, x2_s, tau_u, 3, 3) && patchInterpolation(pStampedTsObs_->second.dTS_dv_right_, x2_s, tau_v, 3, 3);
  else
    pass = patchInterpolation(pStampedTsObs_->second.TS_last_du, x2_s, tau_u, 3, 3) && patchInterpolation(pStampedTsObs_->second.TS_last_dv, x2_s, tau_v, 3, 3);
  jacobian_grad(0, 0) = tau_u.sum() / 9;
  jacobian_grad(0, 1) = tau_v.sum() / 9;

  camSysPtr_->cam_left_ptr_->cam2World(x1_s, x(0), p_l);
  camSysPtr_->cam_right_ptr_->cam2World(x2_s, x(0), p_r);
  double fx = camSysPtr_->cam_left_ptr_->P_(0, 0);
  double fy = camSysPtr_->cam_left_ptr_->P_(1, 1);
  double inv_r = 1 / p_r.z();

  jacobian_u_q << fx * inv_r, 0, -fx * p_r.x() * inv_r * inv_r,
  0, fy * inv_r, -fy * p_r.y() * inv_r * inv_r;

  p_l_unit = p_l / p_l.z();
  if(problem_lr_)
    jacobian_q_lamda = -1/x(0) * 1/x(0) * (camSysPtr_->T_right_left_.block<3, 3>(0, 0) * (p_l_unit) + camSysPtr_->T_right_left_.block<3, 1>(0, 3));
  else
    jacobian_q_lamda = -1/x(0) * 1/x(0) * (T_last_now_.block<3, 3>(0, 0) * (p_l_unit) + T_last_now_.block<3, 1>(0, 3));
  fjac_temp = -jacobian_grad * jacobian_u_q * jacobian_q_lamda;
  for(int i = 0; i < fjac.rows(); i++)
    fjac(i, 0) = fjac_temp;
  return 0;
}

bool DepthProblem::warping(
  const Eigen::Vector2d &x,
  double d,
  const Eigen::Matrix<double, 3, 4> &T_left_virtual,
  Eigen::Vector2d &x1_s,
  Eigen::Vector2d &x2_s) const
{
  // back-project to 3D
  Eigen::Vector3d p_rv, p_last;
  Eigen::Vector3d x1_hom, x2_hom;
  Eigen::Matrix3d R_last_now = T_last_now_.block(0, 0, 3, 3);
  Eigen::Vector3d t_last_now = T_last_now_.block(0, 3, 3, 1);
  camSysPtr_->cam_left_ptr_->cam2World(x, d, p_rv);
  // transfer to left DVS coordinate
  Eigen::Vector3d p_left = T_left_virtual.block<3, 3>(0, 0) * p_rv + T_left_virtual.block<3, 1>(0, 3);
  if(problem_lr_)
  {
    // project onto left and right DVS image plane
    x1_hom = camSysPtr_->cam_left_ptr_->P_.block<3, 3>(0, 0) * p_left +
      camSysPtr_->cam_left_ptr_->P_.block<3, 1>(0, 3);
    x2_hom = camSysPtr_->cam_right_ptr_->P_.block<3, 3>(0, 0) * p_left +
      camSysPtr_->cam_right_ptr_->P_.block<3, 1>(0, 3);
    x1_s = x1_hom.block<2, 1>(0, 0) / x1_hom(2);
    x2_s = x2_hom.block<2, 1>(0, 0) / x2_hom(2);
  }
  else
  {
    // project onto left and last DVS image plane
    x1_hom = camSysPtr_->cam_left_ptr_->P_.block<3, 3>(0, 0) * p_left +
      camSysPtr_->cam_left_ptr_->P_.block<3, 1>(0, 3);
    p_last =  (R_last_now * (p_left) + t_last_now);
    x2_hom = camSysPtr_->cam_left_ptr_->P_.block<3, 3>(0, 0) * p_last +
      camSysPtr_->cam_left_ptr_->P_.block<3, 1>(0, 3);
    x1_s = x1_hom.block<2, 1>(0, 0) / x1_hom(2);
    x2_s = x2_hom.block<2, 1>(0, 0) / x2_hom(2);
  }

  int wx = dpConfigPtr_->patchSize_X_;
  int wy = dpConfigPtr_->patchSize_Y_;
  int width  = camSysPtr_->cam_left_ptr_->width_;
  int height = camSysPtr_->cam_left_ptr_->height_;
      
  if (x1_s(0) < (wx - 1) / 2 || x1_s(0) > width - (wx - 1) / 2 || x1_s(1) < (wy - 1) / 2 || x1_s(1) > height - (wy - 1) / 2)
    return false;
  if (x2_s(0) < (wx - 1) / 2 || x2_s(0) > width - (wx - 1) / 2 || x2_s(1) < (wy - 1) / 2 || x2_s(1) > height - (wy - 1) / 2)
    return false;    
  return true;
}

bool DepthProblem::patchInterpolation(
  const Eigen::MatrixXd &img,
  const Eigen::Vector2d &location,
  Eigen::MatrixXd &patch,
  bool debug) const
{
  int wx = dpConfigPtr_->patchSize_X_;
  int wy = dpConfigPtr_->patchSize_Y_;
  // compute SrcPatch_UpLeft coordinate and SrcPatch_DownRight coordinate
  // check patch boundary is inside img boundary
  Eigen::Vector2i SrcPatch_UpLeft, SrcPatch_DownRight;
  SrcPatch_UpLeft << floor(location[0]) - (wx - 1) / 2, floor(location[1]) - (wy - 1) / 2;
  SrcPatch_DownRight << floor(location[0]) + (wx - 1) / 2, floor(location[1]) + (wy - 1) / 2;

  if (SrcPatch_UpLeft[0] < 0 || SrcPatch_UpLeft[1] < 0)
  {
    if(debug)
    {
      LOG(INFO) << "patchInterpolation 1: " << SrcPatch_UpLeft.transpose();
    }
    return false;
  }
  if (SrcPatch_DownRight[0] >= img.cols() || SrcPatch_DownRight[1] >= img.rows())
  {
    if(debug)
    {
      LOG(INFO) << "patchInterpolation 2: " << SrcPatch_DownRight.transpose();
    }
    return false;
  }

  // compute q1 q2 q3 q4
  Eigen::Vector2d double_indices;
  double_indices << location[1], location[0];

  std::pair<int, int> lower_indices(floor(double_indices[0]), floor(double_indices[1]));
  std::pair<int, int> upper_indices(lower_indices.first + 1, lower_indices.second + 1);

  double q1 = upper_indices.second - double_indices[1];// x
  double q2 = double_indices[1] - lower_indices.second;// x
  double q3 = upper_indices.first - double_indices[0];// y
  double q4 = double_indices[0] - lower_indices.first;// y

  // extract Src patch, size (wy+1) * (wx+1)
  int wx2 = wx + 1;
  int wy2 = wy + 1;
  if (SrcPatch_UpLeft[1] + wy >= img.rows() || SrcPatch_UpLeft[0] + wx >= img.cols())
  {
    if(debug)
    {
      LOG(INFO) << "patchInterpolation 3: " << SrcPatch_UpLeft.transpose()
                << ", location: " << location.transpose()
                << ", floor(location[0]): " << floor(location[0])
                << ", (wx - 1) / 2: " << (wx - 1) / 2
                << ", ans: " << floor(location[0]) - (wx - 1) / 2
                << ", wx: " << wx << " wy: " << wy
                << ", img.row: " << img.rows() << " img.col: " << img.cols();
    }
    return false;
  }
  Eigen::MatrixXd SrcPatch = img.block(SrcPatch_UpLeft[1], SrcPatch_UpLeft[0], wy2, wx2);

  // Compute R, size (wy+1) * wx.
  Eigen::MatrixXd R;
  R = q1 * SrcPatch.block(0, 0, wy2, wx) + q2 * SrcPatch.block(0, 1, wy2, wx);

  // Compute F, size wy * wx.
  patch = q3 * R.block(0, 0, wy, wx) + q4 * R.block(1, 0, wy, wx);
  return true;
}

bool DepthProblem::patchInterpolation(
  const Eigen::MatrixXd &img,
  const Eigen::Vector2d &location,
  Eigen::MatrixXd &patch,
  int wx,
  int wy,
  bool debug) const
{
  // int wx = dpConfigPtr_->patchSize_X_;
  // int wy = dpConfigPtr_->patchSize_Y_;
  // compute SrcPatch_UpLeft coordinate and SrcPatch_DownRight coordinate
  // check patch boundary is inside img boundary
  Eigen::Vector2i SrcPatch_UpLeft, SrcPatch_DownRight;
  SrcPatch_UpLeft << floor(location[0]) - (wx - 1) / 2, floor(location[1]) - (wy - 1) / 2;
  SrcPatch_DownRight << floor(location[0]) + (wx - 1) / 2, floor(location[1]) + (wy - 1) / 2;

  if (SrcPatch_UpLeft[0] < 0 || SrcPatch_UpLeft[1] < 0)
  {
    if(debug)
    {
      LOG(INFO) << "patchInterpolation 1: " << SrcPatch_UpLeft.transpose();
    }
    return false;
  }
  if (SrcPatch_DownRight[0] >= img.cols() || SrcPatch_DownRight[1] >= img.rows())
  {
    if(debug)
    {
      LOG(INFO) << "patchInterpolation 2: " << SrcPatch_DownRight.transpose();
    }
    return false;
  }

  // compute q1 q2 q3 q4
  Eigen::Vector2d double_indices;
  double_indices << location[1], location[0];

  std::pair<int, int> lower_indices(floor(double_indices[0]), floor(double_indices[1]));
  std::pair<int, int> upper_indices(lower_indices.first + 1, lower_indices.second + 1);

  double q1 = upper_indices.second - double_indices[1];// x
  double q2 = double_indices[1] - lower_indices.second;// x
  double q3 = upper_indices.first - double_indices[0];// y
  double q4 = double_indices[0] - lower_indices.first;// y

  // extract Src patch, size (wy+1) * (wx+1)
  int wx2 = wx + 1;
  int wy2 = wy + 1;
  if (SrcPatch_UpLeft[1] + wy >= img.rows() || SrcPatch_UpLeft[0] + wx >= img.cols())
  {
    if(debug)
    {
      LOG(INFO) << "patchInterpolation 3: " << SrcPatch_UpLeft.transpose()
                << ", location: " << location.transpose()
                << ", floor(location[0]): " << floor(location[0])
                << ", (wx - 1) / 2: " << (wx - 1) / 2
                << ", ans: " << floor(location[0]) - (wx - 1) / 2
                << ", wx: " << wx << " wy: " << wy
                << ", img.row: " << img.rows() << " img.col: " << img.cols();
    }
    return false;
  }
  Eigen::MatrixXd SrcPatch = img.block(SrcPatch_UpLeft[1], SrcPatch_UpLeft[0], wy2, wx2);

  // Compute R, size (wy+1) * wx.
  Eigen::MatrixXd R;
  R = q1 * SrcPatch.block(0, 0, wy2, wx) + q2 * SrcPatch.block(0, 1, wy2, wx);

  // Compute F, size wy * wx.
  patch = q3 * R.block(0, 0, wy, wx) + q4 * R.block(1, 0, wy, wx);
  return true;
}


}// core
}// esvo2_core