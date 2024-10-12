#include <esvo2_core/core/EventBM.h>
#include <thread>
#include <numeric>
#include <vector>
#include <Eigen/Eigenvalues>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>

esvo2_core::core::EventBM::EventBM(
  esvo2_core::CameraSystem::Ptr camSysPtr,
  size_t numThread,
  bool bSmoothTS,
  size_t patch_size_X,
  size_t patch_size_Y,
  size_t min_disparity,
  size_t max_disparity,
  size_t step,
  double ZNCC_Threshold,
  bool bUpDownConfiguration):
  camSysPtr_(camSysPtr), sb_(Sobel(3)), NUM_THREAD_(numThread),
  patch_size_X_(patch_size_X), patch_size_Y_(patch_size_Y),
  min_disparity_(min_disparity), max_disparity_(max_disparity), step_(step),
  ZNCC_Threshold_(ZNCC_Threshold),
  bUpDownConfiguration_(bUpDownConfiguration)
{
  ZNCC_MAX_ = 1.0;
  bSmoothTS_ = bSmoothTS;
  /*** For Test ***/
  coarseSearchingFailNum_ = 0;
  fineSearchingFailNum_ = 0;
  infoNoiseRatioLowNum_ = 0;
  outsideNum_ = 0;
}

esvo2_core::core::EventBM::~EventBM()
{}

void esvo2_core::core::EventBM::resetParameters(
  size_t patch_size_X,
  size_t patch_size_Y,
  size_t min_disparity,
  size_t max_disparity,
  size_t step,
  double ZNCC_Threshold,
  bool bUpDownConfiguration,
  size_t patch_size_X_2,
  size_t patch_size_Y_2)
{
  patch_size_X_  = patch_size_X;
  patch_size_Y_  = patch_size_Y;
  min_disparity_ = min_disparity;
  max_disparity_ = max_disparity;
  patch_size_X_2_ = patch_size_X_2;
  patch_size_Y_2_ = patch_size_Y_2;
  step_ = step;
  ZNCC_Threshold_ = ZNCC_Threshold;
  bUpDownConfiguration_ = bUpDownConfiguration;
  /*** For Test ***/
  coarseSearchingFailNum_ = 0;
  fineSearchingFailNum_ = 0;
  infoNoiseRatioLowNum_ = 0;
  outsideNum_ = 0;
}

void esvo2_core::core::EventBM::createMatchProblem(
  constStampedTimeSurfaceObs * pStampedTsObs,
  StampTransformationMap * pSt_map,
  std::vector<dvs_msgs::Event *>* pvEventsPtr)
{
  pStampedTsObs_ = pStampedTsObs;
  pSt_map_ = pSt_map;
  size_t numEvents = pvEventsPtr->size();
  vEventsPtr_.clear();
  vEventsPtr_.reserve(numEvents);
  vEventsPtr_.insert(vEventsPtr_.end(), pvEventsPtr->begin(), pvEventsPtr->end());

  if(bSmoothTS_)
  {
    if(pStampedTsObs_)
      pStampedTsObs_->second.GaussianBlurTS(5);
  }

  vpDisparitySearchBound_.clear();
  vpDisparitySearchBound_.reserve(numEvents);
  for(size_t i = 0; i < vEventsPtr_.size(); i++)
    vpDisparitySearchBound_.push_back(std::make_pair(min_disparity_, max_disparity_));

  K_rtf_inv_ = camSysPtr_->cam_left_ptr_->P_.block(0,0,3,3).inverse();
}

void esvo2_core::core::EventBM::createMatchProblemTwoFrames(
  constStampedTimeSurfaceObs * pStampedTsObs,
  StampTransformationMap * pSt_map,
  std::vector<dvs_msgs::Event *>* pvEventsPtr,
  std::vector<EventMatchPair> * vEMP_lr)
{
  pStampedTsObs_ = pStampedTsObs;
  pSt_map_ = pSt_map;
  size_t numEvents = vEMP_lr->size();
  size_t numEvents2 = pvEventsPtr->size();
  vEventsPtr_.clear();
  vEventsPtr_.reserve(numEvents2);
  vEventsPtr_.insert(vEventsPtr_.end(), pvEventsPtr->begin(), pvEventsPtr->end());
  vEMP_lr_.clear();
  vEMP_lr_.reserve(numEvents);
  vEMP_lr_.insert(vEMP_lr_.end(), vEMP_lr->begin(), vEMP_lr->end());

  vpDisparitySearchBound_.clear();
  vpDisparitySearchBound_.reserve(numEvents);
  for(size_t i = 0; i < vEventsPtr_.size(); i++)
    vpDisparitySearchBound_.push_back(std::make_pair(min_disparity_, max_disparity_));
  T_last_now_ = pStampedTsObs_->second.tr_last_.getTransformationMatrix().inverse() * pStampedTsObs_->second.tr_.getTransformationMatrix();
  num_of_need_ssim_ = 0;
  num_of_success_ssim_ = 0;
}



bool esvo2_core::core::EventBM::match_an_event2(
  const dvs_msgs::Event* pEvent,
  std::pair<size_t, size_t>& pDisparityBound,
  esvo2_core::core::EventMatchPair& emPair)
{
  TicToc t_cost;
  double time_temp;
  size_t lowDisparity = 1;
  size_t upDisparity  = pDisparityBound.second;

  // Prevent zncc calculation from going out of bounds
  int updisp = pEvent->x - (patch_size_X_ + 1)/2;
  if(updisp < 1)
    return false;
  if(updisp < (int)upDisparity)
    upDisparity = (size_t)updisp;
  if(upDisparity < lowDisparity)
    return false;

  // rectify and floor the coordinate
  int redundant = 4;
  Eigen::Vector2d x_rect = camSysPtr_->cam_left_ptr_->getRectifiedUndistortedCoordinate(pEvent->x, pEvent->y);
  
  // check if the rectified and undistorted coordinates are outside the image plane. (Added by Yi Zhou on 12 Jan 2021)
  if(x_rect(0) < 0 || x_rect(0) > camSysPtr_->cam_left_ptr_->width_ - 1 ||
     x_rect(1) < 0 || x_rect(1) > camSysPtr_->cam_left_ptr_->height_ - 1)
  {
    infoNoiseRatioLowNum_++;
    return false;
  }
  
  // This is to avoid depth estimation happening in the mask area.
  if(camSysPtr_->cam_left_ptr_->UndistortRectify_mask_((int)x_rect(1), (int)x_rect(0)) <= 125)
  {
    infoNoiseRatioLowNum_++;
    return false;
  }
  Eigen::Vector2i x1(std::floor(x_rect(0)), std::floor(x_rect(1)));
  Eigen::Vector2i x1_left_top;
  if(!isValidPatch(x1, x1_left_top, patch_size_Y_, patch_size_X_))
  {
    infoNoiseRatioLowNum_++;return false;
  }

  // extract the template patch in the left time_surface
  Eigen::MatrixXd patch_src = pStampedTsObs_->second.TS_left_.block(
    x1_left_top(1), x1_left_top(0), patch_size_Y_, patch_size_X_);
  Eigen::MatrixXd patch_src_normalized;
  if((patch_src.array() < 1).count() > 0.95 * patch_src.size())
  {
    infoNoiseRatioLowNum_++;
    return false;
  }

  // searching along the epipolar line (heading to the left direction)
  double min_cost = ZNCC_MAX_;
  Eigen::Vector2i bestMatch;
  size_t bestDisp;
  Eigen::MatrixXd patch_dst = Eigen::MatrixXd::Zero(patch_size_Y_, patch_size_X_);

  // coarse searching
  std::vector<bool> searching_or_not;
  searching_or_not.reserve(upDisparity - lowDisparity + 1);
  for(int i = 0; i < upDisparity - lowDisparity + 1; i++)
  {
    searching_or_not.push_back(false);
  }

  // for zncc speed up, from "Optimizing ZNCC calculation in binocular stereo matching "
  // mean_l is the mean sum of left patch, Tl_square is the sum of square of the right patch.
  double mean_l = 0, Tl_square = 0, Tr = 0, Tr_square = 0;
  Eigen::VectorXd colSum, colSquareSum;
  Eigen::MatrixXd patch_r;
  use_fast_zncc_ = true;
  
  if(use_fast_zncc_)
  {
    mean_l = patch_src.array().sum() / (patch_src.rows() * patch_src.cols());
    
    Tl_square = (patch_src.array().pow(2)).sum();
    if(x1_left_top(0) - (int)upDisparity < 0)
      return false;
    patch_r = pStampedTsObs_->second.TS_right_.block(
            x1_left_top(1), x1_left_top(0) - (int)upDisparity, patch_size_Y_, patch_size_X_ + (int)upDisparity);
    
    colSum = patch_r.colwise().sum();
    
    colSquareSum = (patch_r.array().pow(2)).colwise().sum();
    
    for(int m = lowDisparity; m < lowDisparity + patch_src.cols() && (colSum.size() - 1 - m >= 0); m++)
    {
      Tr += colSum(colSum.size() - 1 - m);
      Tr_square += colSquareSum(colSum.size() - 1 - m);
    }
  }

  // To improve search efficiency, static stereo block matching performs searches at a certain step size and conducts additional searches near points that preliminarily match.
  if(!epipolarSearchingCoarse(min_cost, bestMatch, bestDisp, patch_dst,
    lowDisparity, upDisparity, step_,
    x1, patch_src, bUpDownConfiguration_, searching_or_not,
    colSum, colSquareSum, mean_l, Tl_square, Tr, Tr_square))
  {
    coarseSearchingFailNum_++;
    return false;
  }

  std::vector<size_t> searching_radius;
  searching_radius.reserve(upDisparity - lowDisparity + 1);
  for(int i = 0; i < searching_or_not.size(); i++)
  {
    if(searching_or_not[i] == true)
    {
      searching_radius.push_back(lowDisparity + i);
    }
  }
  if(step_ > 1)
  {
    if(!epipolarSearchingFine(min_cost, bestMatch, bestDisp, patch_dst,
      searching_radius, 
      x1, patch_src, bUpDownConfiguration_))
    {
      fineSearchingFailNum_++;
      return false;
    }
  }

  // transfer best match to emPair
  if(min_cost <= ZNCC_Threshold_*1.02)
  {
    emPair.x_left_raw_ = Eigen::Vector2d((double)pEvent->x, (double)pEvent->y);
    emPair.x_left_ = x_rect;
    emPair.x_right_ = Eigen::Vector2d((double)bestMatch(0), (double)bestMatch(1)) ;
    emPair.t_ = pEvent->ts;
    double disparity;
    if(bUpDownConfiguration_)
      disparity = x1(1) - bestMatch(1);
    else
      disparity = x1(0) - bestMatch(0);
    double depth = camSysPtr_->baseline_ * camSysPtr_->cam_left_ptr_->P_(0,0) / disparity;
    emPair.trans_ = pStampedTsObs_->second.tr_;
    emPair.invDepth_ = 1.0 / depth;
    emPair.cost_ = min_cost;
    emPair.disp_ = disparity;
    return true;
  }
  else
  {
    return false;
  }
}

bool esvo2_core::core::EventBM::match_an_event(
  dvs_msgs::Event* pEvent,
  std::pair<size_t, size_t>& pDisparityBound,
  esvo2_core::core::EventMatchPair& emPair)
{
  size_t lowDisparity = pDisparityBound.first;
  size_t upDisparity  = pDisparityBound.second;
  // rectify and floor the coordinate
  Eigen::Vector2d x_rect = camSysPtr_->cam_left_ptr_->getRectifiedUndistortedCoordinate(pEvent->x, pEvent->y);
  // check if the rectified and undistorted coordinates are outside the image plane. (Added by Yi Zhou on 12 Jan 2021)
  if(x_rect(0) < 0 || x_rect(0) > camSysPtr_->cam_left_ptr_->width_ - 1 ||
     x_rect(1) < 0 || x_rect(1) > camSysPtr_->cam_left_ptr_->height_ - 1)
    return false;
  // This is to avoid depth estimation happening in the mask area.
  if(camSysPtr_->cam_left_ptr_->UndistortRectify_mask_((int)x_rect(1), (int)x_rect(0)) <= 125)
    return false;
  Eigen::Vector2i x1(std::floor(x_rect(0)), std::floor(x_rect(1)));
  Eigen::Vector2i x1_left_top;
  if(!isValidPatch(x1, x1_left_top, patch_size_X_2_, patch_size_Y_2_))
    return false;
  // extract the template patch in the left time_surface
  Eigen::MatrixXd patch_src = pStampedTsObs_->second.TS_left_.block(
    x1_left_top(1), x1_left_top(0), patch_size_Y_, patch_size_X_);

  if((patch_src.array() < 1).count() > 0.95 * patch_src.size())
  {
    infoNoiseRatioLowNum_++;
    return false;
  }

  // LOG(INFO) << "patch_src is extracted";

  // searching along the epipolar line (heading to the left direction)
  double min_cost = ZNCC_MAX_;
  Eigen::Vector2i bestMatch;
  size_t bestDisp;
  Eigen::MatrixXd patch_dst = Eigen::MatrixXd::Zero(patch_size_Y_, patch_size_X_);
  // coarse searching
  if(!epipolarSearching(min_cost, bestMatch, bestDisp, patch_dst,
    lowDisparity, upDisparity, step_,
    x1, patch_src, bUpDownConfiguration_))
  {
    coarseSearchingFailNum_++;
    return false;
  }
  // fine searching
  size_t fine_searching_start_pos = bestDisp-(step_-1) >= 0 ? bestDisp-(step_-1) : 0;
  if(!epipolarSearching(min_cost, bestMatch, bestDisp, patch_dst,
                    fine_searching_start_pos, bestDisp+(step_-1), 1,
                    x1, patch_src, bUpDownConfiguration_))
  {
    // This indicates the local minima is not surrounded by two neighbors with larger cost,
    // This case happens when the best match locates over/outside the boundary of the Time Surface.
    fineSearchingFailNum_++;
    return false;
  }

  // transfer best match to emPair
  if(min_cost <= ZNCC_Threshold_)
  {
    emPair.x_left_raw_ = Eigen::Vector2d((double)pEvent->x, (double)pEvent->y);
    emPair.x_left_ = x_rect;
    emPair.x_right_ = Eigen::Vector2d((double)bestMatch(0), (double)bestMatch(1)) ;
    emPair.t_ = pEvent->ts;
    double disparity;
    if(bUpDownConfiguration_)
      disparity = x1(1) - bestMatch(1);
    else
      disparity = x1(0) - bestMatch(0);
    double depth = camSysPtr_->baseline_ * camSysPtr_->cam_left_ptr_->P_(0,0) / disparity;

    // auto st_map_iter = tools::StampTransformationMap_lower_bound(*pSt_map_, emPair.t_);
    // if(st_map_iter == pSt_map_->end())
    //   return false;
    emPair.trans_ = emPair.trans_ = pStampedTsObs_->second.tr_;
    emPair.invDepth_ = 1.0 / depth;
    emPair.cost_ = min_cost;
    emPair.disp_ = disparity;
    return true;
  }
  else
  {
    // LOG(INFO) << "BM fails because: " << min_cost << " > " << ZNCC_Threshold_;
    return false;
  }
}

bool esvo2_core::core::EventBM::match_an_eventTwoFrames(
  EventMatchPair EMP_lr,
  std::pair<size_t, size_t>& pDisparityBound,
  esvo2_core::core::EventMatchPair& emPair)
{
  int lowDisparity = -10;
  int upDisparity  = 70;
  
  Eigen::Vector2d x_rect;
  x_rect = EMP_lr.x_left_;
  double dx = EMP_lr.x_last_(0) - EMP_lr.x_left_(0);
  double dy = EMP_lr.x_last_(1) - EMP_lr.x_left_(1);
  if(dx == 0 && dy == 0)
  {
    dx = 0;
    dy = 1;
  }
  if(abs(dx) >= abs(dy))
  {
    dy = dy / abs(dx);
    dx = dx / abs(dx);
  }
  else
  {
    dx = dx / abs(dy);
    dy = dy / abs(dy);
  }
  
  // check if the rectified and undistorted coordinates are outside the image plane. (Added by Yi Zhou on 12 Jan 2021)

  if(x_rect(0) < 0 || x_rect(0) > camSysPtr_->cam_left_ptr_->width_ - 1 ||
     x_rect(1) < 0 || x_rect(1) > camSysPtr_->cam_left_ptr_->height_ - 1)
  {
    outsideNum_ ++;
    return false;
  }
  
  // This is to avoid depth estimation happening in the mask area.
  if(camSysPtr_->cam_left_ptr_->UndistortRectify_mask_((int)x_rect(1), (int)x_rect(0)) <= 125)
  {
    infoNoiseRatioLowNum_++;
    return false;
  }
  // Eigen::Vector2i x1(std::floor(EMP_lr.x_last_(0)), std::floor(EMP_lr.x_last_(1)));
  Eigen::Vector2i x1(std::floor(x_rect(0)), std::floor(x_rect(1)));
  Eigen::Vector2i x1_left_top;
  if(!isValidPatch(x1, x1_left_top, patch_size_X_2_, patch_size_Y_2_))
  {
    infoNoiseRatioLowNum_++;
    return false;
  }
  // extract the template patch in the left time_surface
  Eigen::MatrixXd patch_src = pStampedTsObs_->second.TS_left_.block(
    x1_left_top(1), x1_left_top(0), patch_size_X_2_, patch_size_Y_2_);
  if((patch_src.array() < 1).count() > 0.95 * patch_src.size())
  {
    infoNoiseRatioLowNum_++;
    return false;
  }

  // searching along the epipolar line (heading to the left direction)
  double min_cost = ZNCC_MAX_;
  Eigen::Vector2i bestMatch;
  int bestDisp;
  Eigen::MatrixXd patch_dst = Eigen::MatrixXd::Zero(patch_size_X_2_, patch_size_Y_2_);

  // coarse searching
  if(sqrt(dx*dx+dy*dy)>1)
  {
    upDisparity = upDisparity/sqrt(dx*dx+dy*dy);
  }
  vector<double> a;
  vector<bool> fine_search(upDisparity-lowDisparity + 1);
  for(int i = 0; i < fine_search.size();i++)
  {
    fine_search[i] = false;
  }
  if(!epipolarSearchingTwoFrames(min_cost, bestMatch, bestDisp, patch_dst,
    lowDisparity, upDisparity, 1,
    x1, patch_src, dx, dy, emPair.costs_,
    fine_search, bUpDownConfiguration_))
  {
    coarseSearchingFailNum_++;
    return false;
  }

  // transfer best match to emPair
  if(min_cost <= ZNCC_Threshold_*3.00)
  {
    emPair.x_left_raw_ = EMP_lr.x_left_raw_;
    emPair.x_left_ = x_rect;
    emPair.lr_depth = 1/EMP_lr.invDepth_;
    emPair.x_last_ = Eigen::Vector2d((double)bestMatch(0), (double)bestMatch(1)) ;
    emPair.t_ = EMP_lr.t_;

    double disparity;
    if(bUpDownConfiguration_)
      disparity = sqrt(pow(emPair.x_left_(0) - bestMatch(0),2) + pow(emPair.x_left_(1) - bestMatch(1),2));
    else
      disparity = sqrt(pow(emPair.x_left_(0) - bestMatch(0),2) + pow(emPair.x_left_(1) - bestMatch(1),2));
    
    emPair.error_ = EMP_lr.error_;
    emPair.lr_cost = EMP_lr.cost_;
    emPair.ln_cost = min_cost;
    double depth = triangulatePoint(emPair.x_left_, emPair.x_last_, emPair.error_);
    if(depth < 0.8 || depth > 40)
      return false;

    emPair.x_right_<< EMP_lr.x_left_(0) - camSysPtr_->baseline_ * camSysPtr_->cam_left_ptr_->P_(0,0) / depth, EMP_lr.x_left_(1);
    emPair.trans_ = emPair.trans_ = pStampedTsObs_->second.tr_;
    emPair.invDepth_ = 1.0 / depth;
    emPair.cost_ = min_cost;
    emPair.disp_ = disparity;
    return true;
  }
  else
  {
    return false;
  }
}

bool esvo2_core::core::EventBM::epipolarSearching(
  double& min_cost, Eigen::Vector2i& bestMatch, size_t& bestDisp, Eigen::MatrixXd& patch_dst,
  size_t searching_start_pos, size_t searching_end_pos, size_t searching_step,
  Eigen::Vector2i& x1, Eigen::MatrixXd& patch_src, bool bUpDownConfiguration)
{
  bool bFoundOneMatch = false;
  std::map<size_t, double> mDispCost;

  for(size_t disp = searching_start_pos;disp <= searching_end_pos; disp+=searching_step)
  {
    Eigen::Vector2i x2;
    if(!bUpDownConfiguration)
      x2 << x1(0) - disp, x1(1);
    else
      x2 << x1(0), x1(1) - disp;
    Eigen::Vector2i x2_left_top;
    if(!isValidPatch(x2, x2_left_top, patch_size_Y_, patch_size_X_))
    {
      mDispCost.emplace(disp, ZNCC_MAX_);
      continue;
    }

    patch_dst = pStampedTsObs_->second.TS_right_.block(
      x2_left_top(1), x2_left_top(0), patch_size_Y_, patch_size_X_);
    double cost = ZNCC_MAX_;
    cost = zncc_cost(patch_src, patch_dst, false);
    mDispCost.emplace(disp, cost);

    if(cost <= min_cost)
    {
      min_cost = cost;
      bestMatch = x2;
      bestDisp = disp;
    }
  }

  if(searching_step > 1)// coarse
  {
    if(mDispCost.find(bestDisp - searching_step) != mDispCost.end() &&
       mDispCost.find(bestDisp + searching_step) != mDispCost.end())
    {

      if(mDispCost[bestDisp - searching_step] < ZNCC_MAX_ && mDispCost[bestDisp + searching_step] < ZNCC_MAX_ )
        if(min_cost < ZNCC_Threshold_)
          bFoundOneMatch = true;
    }
  }
  else// fine
  {
    if(min_cost < ZNCC_Threshold_)
      bFoundOneMatch = true;
  }
  return bFoundOneMatch;
}

  bool esvo2_core::core::EventBM::epipolarSearchingCoarse(
  double& min_cost, Eigen::Vector2i& bestMatch, size_t& bestDisp, Eigen::MatrixXd& patch_dst,
  size_t searching_start_pos, size_t searching_end_pos, size_t searching_step,
  Eigen::Vector2i& x1, Eigen::MatrixXd& patch_src, bool bUpDownConfiguration, vector<bool>& searching_or_not,
  Eigen::VectorXd &colSum, Eigen::VectorXd &colSquareSum, double &mean_l, double &Tl_square, double &Tr, double &Tr_square)
  {
    double var_l = 0;
    int zncc_num = 0;
    bool bFoundOneMatch = false;
    
    if(!use_fast_zncc_)
    {
      var_l = Tl_square - patch_src.rows() * patch_src.cols() * mean_l * mean_l;
    }

    for(size_t disp = searching_start_pos;disp <= searching_end_pos; disp+=searching_step)
    {
      Eigen::Vector2i x2;
      if(!bUpDownConfiguration)
        x2 << x1(0) - disp, x1(1);
      else
        x2 << x1(0), x1(1) - disp;
      Eigen::Vector2i x2_left_top;
      if(!isValidPatch(x2, x2_left_top, patch_size_Y_, patch_size_X_))
      {
        continue;
      }

      patch_dst = pStampedTsObs_->second.TS_right_.block(
        x2_left_top(1), x2_left_top(0), patch_size_Y_, patch_size_X_);

      double cost = ZNCC_MAX_;
      if(use_fast_zncc_)
      {
        int disp_to_rm = disp, step_to_rm = searching_step;
        cost = zncc_cost_fast(colSum, colSquareSum, patch_src, patch_dst, disp_to_rm, step_to_rm, mean_l, Tl_square, Tr, Tr_square);
      }
      else
      {
        cost = zncc_cost2(patch_src, patch_dst, var_l, mean_l);
      }
      zncc_num++;

      // add to preliminarily match list
      if(cost <= ZNCC_Threshold_*1.035)
      {
        for(int i = disp - searching_start_pos - searching_step; i < disp - searching_start_pos + searching_step + 1; i++)
        {
          if(i >= 0 && i < searching_or_not.size())
            searching_or_not[i] = true;
        }
      }
      if(cost <= min_cost)
      {
        
        min_cost = cost;
        bestMatch = x2;
        bestDisp = disp;
      }
    }

    if(min_cost < ZNCC_Threshold_*1.03)
      bFoundOneMatch = true;
    return bFoundOneMatch;
}


  bool esvo2_core::core::EventBM::epipolarSearchingFine(double& min_cost, Eigen::Vector2i& bestMatch, size_t& bestDisp, Eigen::MatrixXd& patch_dst,
  vector<size_t>& searching_radius,
  Eigen::Vector2i& x1, Eigen::MatrixXd& patch_src, bool bUpDownConfiguration)
  {
    bool bFoundOneMatch = false;
    double mean_l = patch_src.array().mean();
    double var_l = (patch_src.array() * patch_src.array()).sum() - patch_src.rows() * patch_src.cols() * mean_l * mean_l;
    for(int i = 0;i < searching_radius.size(); i++)
    {
      Eigen::Vector2i x2;
      size_t disp = searching_radius[i];
      if(!bUpDownConfiguration)
        x2 << x1(0) - disp, x1(1);
      else
        x2 << x1(0), x1(1) - disp;
      Eigen::Vector2i x2_left_top;
      if(!isValidPatch(x2, x2_left_top, patch_size_Y_, patch_size_X_))
      {
        continue;
      }
      patch_dst = pStampedTsObs_->second.TS_right_.block(
        x2_left_top(1), x2_left_top(0), patch_size_Y_, patch_size_X_);
      double cost = ZNCC_MAX_;
      cost = zncc_cost2(patch_src, patch_dst, var_l, mean_l);//zncc_cost(patch_src, patch_dst, true);
      if(cost <= min_cost)
      {
        min_cost = cost;
        bestMatch = x2;
        bestDisp = disp;
      }
    }
    if(min_cost < ZNCC_Threshold_*1.02)
        bFoundOneMatch = true;
    return bFoundOneMatch;
  }


bool esvo2_core::core::EventBM::epipolarSearchingTwoFrames(
  double& min_cost, Eigen::Vector2i& bestMatch, int & bestDisp, Eigen::MatrixXd& patch_dst,
  int searching_start_pos, int searching_end_pos, size_t searching_step,
  Eigen::Vector2i& x1, Eigen::MatrixXd& patch_src, double dx, double dy, std::vector<double>& costs,
  std::vector<bool>& fine_search, bool bUpDownConfiguration)
{
  //ssim
  bool use_zncc = (costs.size()<10);
  cv::Rect rect1, rect2;
  cv::Mat p1, p2;
  Eigen::Vector2i x1_left_top;
  if(!use_zncc)
  {
    isValidPatch(x1, x1_left_top, patch_size_X_2_, patch_size_Y_2_);
    rect1 = cv::Rect(x1_left_top(0), x1_left_top(1), patch_size_Y_2_, patch_size_X_2_);
    p1  = pStampedTsObs_->second.cvImagePtr_left_->image(rect1);
  }
  double mean_l = patch_src.array().mean();
  double var_l = (patch_src.array() * patch_src.array()).sum() - patch_src.rows() * patch_src.cols() * mean_l * mean_l;

  bool bFoundOneMatch = false;
  std::map<int, double> mDispCost;
  int curr = 0, bestCurr = 0;
  for(int disp = searching_start_pos;disp <= searching_end_pos; disp+=searching_step)
  {
    if(!use_zncc && fine_search[curr]==false)
    {
      curr++;
      continue;
    }
    Eigen::Vector2i x2;
    if(!bUpDownConfiguration)
      x2 << int(x1(0) + disp*dx), int(x1(1) + disp*dy);
    else
      x2 << int(x1(0) + disp*dy), int(x1(1) + disp*dx);
    
    Eigen::Vector2i x2_left_top;
    if(!isValidPatch(x2, x2_left_top, patch_size_X_2_, patch_size_Y_2_))
    {
      mDispCost.emplace(disp, ZNCC_MAX_);
      continue;
    }
    double cost = ZNCC_MAX_;
    
    if(use_zncc)
    {
      // zncc
      patch_dst = pStampedTsObs_->second.TS_last_.block(
      x2_left_top(1), x2_left_top(0), patch_size_X_2_, patch_size_Y_2_);
      cost = zncc_cost2(patch_src, patch_dst, var_l, mean_l);
    }
    else
    {
      // ssim
      rect2 = cv::Rect(x2_left_top(0), x2_left_top(1), patch_size_Y_2_, patch_size_X_2_);
      p2  = pStampedTsObs_->second.cvImagePtr_last_->image(rect2);
      cost = 1 - getMSSIM (p1, p2);
    }
    costs.push_back(cost);
    mDispCost.emplace(disp, cost);

    // todo reduce AA point by local num
    if(cost <= min_cost && cost>0.00001)
    {
      min_cost = cost;
      bestMatch = x2;
      bestDisp = disp;
      bestCurr = curr;
    }
    if(use_zncc && cost<ZNCC_Threshold_*3.00 && cost>0.00001)
    {
      int search_bound = 2;
      for(int i = curr-search_bound; i <= curr + search_bound; i++)
      {
        if(i>=0 && i<fine_search.size())
          fine_search[i]=true;
      }
    }
    curr++;
  }
  if(searching_step > 1)// coarse
  {
    if(mDispCost.find(bestDisp - searching_step) != mDispCost.end() &&
       mDispCost.find(bestDisp + searching_step) != mDispCost.end())
    {
      if(mDispCost[bestDisp - searching_step] < ZNCC_MAX_ && mDispCost[bestDisp + searching_step] < ZNCC_MAX_ )
        if(min_cost < ZNCC_Threshold_*3.00)
          bFoundOneMatch = true;
    }
  }
  else// fine
  {
    if(use_zncc && min_cost < ZNCC_Threshold_*3.00)
      bFoundOneMatch = true;
  }
  return bFoundOneMatch;
}

bool esvo2_core::core::EventBM::isNeedSSIM(std::vector<double>& costs)
{
  int start = -1;
  int end = -1;
  for(int i = 0; i<costs.size(); i++)
  {
    if(costs[i]<0.25)
    {
      end = i;
      if(start == -1)
      {
        start = i;
      }
    }
  }
  if(end - start > 5)
  {
    return true;
  }
  return false;
}

void esvo2_core::core::EventBM::match_all_SingleThread(
  std::vector<EventMatchPair> &vEMP)
{
  TicToc t;
  coarseSearchingFailNum_ = 0;
  fineSearchingFailNum_ = 0;
  infoNoiseRatioLowNum_ = 0;
  outsideNum_ = 0;
  vEMP.clear();
  vEMP.reserve(vEventsPtr_.size());
  for(size_t i = 0; i < vEventsPtr_.size(); i++)
  {
    EventMatchPair emp;
    std::pair<size_t, size_t> pDisparityBound = vpDisparitySearchBound_[i];
    if(match_an_event2(vEventsPtr_[i], pDisparityBound, emp))
      vEMP.emplace_back(emp);
  }
  // LOG(INFO) << "vEMP size is: " << vEMP.size();
  // LOG(INFO) << "Total number of events: " << vEventsPtr_.size();
  // LOG(INFO) << "Info-noise ratio low # " << infoNoiseRatioLowNum_;
  // LOG(INFO) << "outsideNum_ # " << outsideNum_;
  // LOG(INFO) << "coarse searching fails # " << coarseSearchingFailNum_;
  // LOG(INFO) << "fine searching fails # " << fineSearchingFailNum_;
}

void esvo2_core::core::EventBM::match_all_HyperThread(
  vector<EventMatchPair> &vEMP)
{
  TicToc t;
  t.tic();
  std::vector<EventBM::Job> jobs(NUM_THREAD_);
  for(size_t i = 0;i < NUM_THREAD_; i++)
  {
    jobs[i].i_thread_ = i;
    jobs[i].pvEventPtr_ = &vEventsPtr_;
    jobs[i].pvpDisparitySearchBound_ = &vpDisparitySearchBound_;
    jobs[i].pvEventMatchPair_ = std::make_shared<std::vector<EventMatchPair> >();
  }

  // use tbb to parallelize the process
  tbb::global_control c(tbb::global_control::max_allowed_parallelism, NUM_THREAD_);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, jobs.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i < range.end(); ++i) {
                            EventBM::match2(jobs[i]);
                        }
                    });
  size_t numPoints = 0;
  for(size_t i = 0;i < NUM_THREAD_;i++)
    numPoints += jobs[i].pvEventMatchPair_->size();
  vEMP.clear();
  vEMP.reserve(numPoints);
  for(size_t i = 0;i < NUM_THREAD_;i++)
    vEMP.insert(vEMP.end(), jobs[i].pvEventMatchPair_->begin(), jobs[i].pvEventMatchPair_->end());

  double time_hyperCroase = t.toc();
}



void esvo2_core::core::EventBM::match_all_SingleThreadTwoFrames(
  std::vector<EventMatchPair> &vEMP,
  std::vector<EventMatchPair> &vEMP_fail)
{
  TicToc t;
  t.tic();
  coarseSearchingFailNum_ = 0;
  fineSearchingFailNum_ = 0;
  infoNoiseRatioLowNum_ = 0;
  outsideNum_ = 0;
  dxdyBigNum_ = 0;
  vEMP.clear();
  vEMP.reserve(vEMP_lr_.size());
  for(size_t i = 0; i < vEMP_lr_.size(); i++)
  {
    EventMatchPair emp;
    std::pair<size_t, size_t> pDisparityBound = vpDisparitySearchBound_[i];
    if(match_an_eventTwoFrames(vEMP_lr_[i], pDisparityBound, emp))
      vEMP.emplace_back(emp);
  }
}


void esvo2_core::core::EventBM::match_all_HyperThreadTwoFrames(
  std::vector<EventMatchPair> &vEMP,
  std::vector<EventMatchPair> &vEMP_fail)
{
  TicToc t;
  t.tic();
  std::vector<EventBM::Job2> jobs(NUM_THREAD_);
  for(size_t i = 0;i < NUM_THREAD_; i++)
  {
    jobs[i].i_thread_ = i;
    jobs[i].pvEventPairPtr_ = &vEMP_lr_;
    jobs[i].pvpDisparitySearchBound_ = &vpDisparitySearchBound_;
    jobs[i].pvEventMatchPair_ = std::make_shared<std::vector<EventMatchPair> >();
  }

  // use tbb to parallelize the process
  tbb::global_control c(tbb::global_control::max_allowed_parallelism, NUM_THREAD_);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, jobs.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i < range.end(); ++i) {
                            EventBM::match2_TwoFrames(jobs[i]);
                        }
                    });
  size_t numPoints = 0;
  for(size_t i = 0;i < NUM_THREAD_;i++)
    numPoints += jobs[i].pvEventMatchPair_->size();
  vEMP.clear();
  vEMP.reserve(numPoints);
  for(size_t i = 0;i < NUM_THREAD_;i++)
    vEMP.insert(vEMP.end(), jobs[i].pvEventMatchPair_->begin(), jobs[i].pvEventMatchPair_->end());
}

bool esvo2_core::core::EventBM::isValidPatch(
  Eigen::Vector2i& x,
  Eigen::Vector2i& left_top,
  int size_y, int size_x)
{
  int wx = (size_x - 1) / 2;
  int wy = (size_y - 1) / 2;
  left_top = Eigen::Vector2i(x(0) - wx, x(1) - wy);
  Eigen::Vector2i right_bottom(x(0) + wx, x(1) + wy);
  // NOTE: The patch cannot touch the boundary row/col of the orginal image,
  // since in the nonlinear optimization, the interpolation would access
  // the neighbouring row/col!!!
  if(left_top(0) < 1 || left_top(1) < 1 ||
     right_bottom(0) >= camSysPtr_->cam_left_ptr_->width_ - 1 ||
     right_bottom(1) >= camSysPtr_->cam_left_ptr_->height_ - 1 )
    return false;
  return true;
}

void esvo2_core::core::EventBM::match2(
  EventBM::Job& job)
{
  size_t i_thread = job.i_thread_;
  size_t totalNumEvents = job.pvEventPtr_->size();
  job.pvEventMatchPair_->reserve(totalNumEvents / NUM_THREAD_ + 1);
  auto ev_it = job.pvEventPtr_->begin();
  std::advance(ev_it, i_thread);
  for(size_t i = i_thread; i < totalNumEvents; i+=NUM_THREAD_, std::advance(ev_it, NUM_THREAD_))
  {
    EventMatchPair emp;
    std::pair<size_t, size_t> pDisparityBound = (*job.pvpDisparitySearchBound_)[i];
    if(match_an_event2(*ev_it, pDisparityBound, emp))
      job.pvEventMatchPair_->push_back(emp);
  }
}

void esvo2_core::core::EventBM::match2_TwoFrames(
  EventBM::Job2& job)
{
  size_t i_thread = job.i_thread_;
  size_t totalNumEvents = job.pvEventPairPtr_->size();
  job.pvEventMatchPair_->reserve(totalNumEvents / NUM_THREAD_ + 1);
  auto ev_it = job.pvEventPairPtr_->begin();
  std::advance(ev_it, i_thread);
  for(size_t i = i_thread; i < totalNumEvents; i+=NUM_THREAD_, std::advance(ev_it, NUM_THREAD_))
  {
    EventMatchPair emp;
    std::pair<size_t, size_t> pDisparityBound = (*job.pvpDisparitySearchBound_)[i];
    if(match_an_eventTwoFrames(*ev_it, pDisparityBound, emp))
      job.pvEventMatchPair_->push_back(emp);
  }
}

void esvo2_core::core::EventBM::match(
  EventBM::Job& job)
{
  size_t i_thread = job.i_thread_;
  size_t totalNumEvents = job.pvEventPtr_->size();
  job.pvEventMatchPair_->reserve(totalNumEvents / NUM_THREAD_ + 1);

  auto ev_it = job.pvEventPtr_->begin();
  std::advance(ev_it, i_thread);
  for(size_t i = i_thread; i < totalNumEvents; i+=NUM_THREAD_, std::advance(ev_it, NUM_THREAD_))
  {
    EventMatchPair emp;
    std::pair<size_t, size_t> pDisparityBound = (*job.pvpDisparitySearchBound_)[i];
    if(match_an_event2(*ev_it, pDisparityBound, emp))
      job.pvEventMatchPair_->push_back(emp);
  }
}

double esvo2_core::core::EventBM::zncc_cost(
  Eigen::MatrixXd &patch_left,
  Eigen::MatrixXd &patch_right,
  bool normalized)
{
  double cost;
  if(!normalized)
  {
    Eigen::MatrixXd patch_left_normalized, patch_right_normalized;
    tools::normalizePatch(patch_left, patch_left_normalized);
    tools::normalizePatch(patch_right, patch_right_normalized);
    cost = 0.5 * (1 - (patch_left_normalized.array() * patch_right_normalized.array()).sum() / (patch_left.rows() * patch_left.cols()));
  }
  else
    cost = 0.5 * (1 - (patch_left.array() * patch_right.array()).sum() / (patch_left.rows() * patch_left.cols()));
  return cost;
}

double esvo2_core::core::EventBM::zncc_cost_fast(
  Eigen::VectorXd &colSum,
  Eigen::VectorXd &colSquareSum,
  Eigen::MatrixXd &patch_left,
  Eigen::MatrixXd &patch_right,
  int &disp_to_rm, 
  int &step_to_rm,
  double &mean_l, 
  double &Tl_square,
  double &Tr,
  double &Tr_square
)
{
  double cost, cov, var_l, var_r;

  double mean_r = Tr / (patch_right.rows() * patch_right.cols());
  if(mean_r == 0.0)
    mean_r = 1e-3;
  if(abs(mean_l - mean_r) / mean_l  > 5  || abs(mean_l - mean_r) / mean_r  > 5)
  {
    cost = 0;
    // return 1.0;
  }
  else
  {
    cov = (patch_left.array() * patch_right.array()).sum() - mean_l * Tr;
    var_l = Tl_square - patch_right.rows() * patch_right.cols() * mean_l * mean_l;
    var_r = Tr_square - Tr * Tr / (patch_right.rows() * patch_right.cols()) ;

    if(var_l * var_r == 0)
      cost = 0;
    else
      cost = cov / sqrt(var_l * var_r);
  }

  for(int i = 0; i < step_to_rm; i++)
  {
    if((int)colSum.size() - 1 - disp_to_rm - patch_right.cols() > 0)
    {
      Tr = Tr - colSum[colSum.size() - 1 - disp_to_rm] + colSum[colSum.size() - 1 - disp_to_rm - patch_right.cols()];
      Tr_square = Tr_square - colSquareSum[colSum.size() - 1 - disp_to_rm] + colSquareSum[colSum.size() - 1 - disp_to_rm - patch_right.cols()];
      disp_to_rm ++;
    }
  }
  
  return 0.5 * (1 - cost);
}

double esvo2_core::core::EventBM::zncc_cost2(
  Eigen::MatrixXd &patch_left,
  Eigen::MatrixXd &patch_right,
  double &var_l,
  double &mean_l
)
{
  double cost, cov, var_r, var_l_ori;
  double mean_r = patch_right.array().mean();
  cov = (patch_left.array() * patch_right.array()).sum() - patch_right.rows() * patch_right.cols() * mean_l * mean_r;
  var_r = (patch_right.array() * patch_right.array()).sum() - patch_right.rows() * patch_right.cols() * mean_r * mean_r;
  if(var_l * var_r == 0)
    cost = 0;
  else
    cost = cov / sqrt(var_l * var_r);
  return 0.5 * (1 - cost);
}

float esvo2_core::core::EventBM::getMSSIM( const cv::Mat& i1, const cv::Mat& i2)
{
  const double C1 = 6.5025, C2 = 58.5225;
  /***************************** INITS **********************************/
  int d     = CV_32F;
  cv::Mat I1, I2;
  i1.convertTo(I1, d);           // cannot calculate on one byte large values
  i2.convertTo(I2, d);
  cv::Mat I2_2   = I2.mul(I2);        // I2^2
  cv::Mat I1_2   = I1.mul(I1);        // I1^2
  cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2
  /*************************** END INITS **********************************/
  cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
  cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
  cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
  cv::Mat mu1_2   =   mu1.mul(mu1);
  cv::Mat mu2_2   =   mu2.mul(mu2);
  cv::Mat mu1_mu2 =   mu1.mul(mu2);
  cv::Mat sigma1_2, sigma2_2, sigma12;
  cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
  sigma1_2 -= mu1_2;
  cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
  sigma2_2 -= mu2_2;
  cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
  sigma12 -= mu1_mu2;
  cv::Mat t1, t2, t3;
  t1 = 2 * mu1_mu2 + C1;
  t2 = 2 * sigma12 + C2;
  t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
  t1 = mu1_2 + mu2_2 + C1;
  t2 = sigma1_2 + sigma2_2 + C2;
  t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
  cv::Mat ssim_map;
  divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
  cv::Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
  return mssim[0];
}

double esvo2_core::core::EventBM::triangulatePoint(Eigen::Vector2d &point0, Eigen::Vector2d &point1, double depth)
{
  Eigen::Matrix<double, 3, 4> Pose0 = Eigen::Matrix4d::Identity().block(0,0,3,4);
  Eigen::Matrix<double, 3, 4> Pose1 = T_last_now_.block(0,0,3,4);
  Eigen::Vector3d pointa, pointb;
  pointa << point0(0), point0(1), 1;
  pointb << point1(0), point1(1), 1;
  pointa = K_rtf_inv_ * pointa;
  pointb = K_rtf_inv_ * pointb;
	Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
	design_matrix.row(0) = pointa[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = pointa[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = pointb[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = pointb[1] * Pose1.row(2) - Pose1.row(1);
	Eigen::Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  triangulated_point = triangulated_point / triangulated_point(3);
  Eigen::Vector3d temp = triangulated_point.block(0,0,3,1);
	return temp(2);
}
