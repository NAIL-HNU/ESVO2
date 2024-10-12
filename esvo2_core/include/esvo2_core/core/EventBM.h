#ifndef ESVO2_CORE_CORE_EVENTBM_H
#define ESVO2_CORE_CORE_EVENTBM_H
#include <Eigen/Eigen>
#include <esvo2_core/tools/utils.h>
#include <esvo2_core/tools/sobel.h>
#include <esvo2_core/container/CameraSystem.h>
#include <esvo2_core/container/DepthMap.h>
#include <esvo2_core/container/EventMatchPair.h>

namespace esvo2_core
{
using namespace tools;
using namespace container;
namespace core
{
class EventBM
{
  struct Job
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t i_thread_;
    std::vector<dvs_msgs::Event*>* pvEventPtr_;
    std::vector<std::pair<size_t, size_t> >* pvpDisparitySearchBound_;
    std::shared_ptr<std::vector<EventMatchPair> > pvEventMatchPair_;
  };

  struct Job2
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t i_thread_;
    std::vector<EventMatchPair>* pvEventPairPtr_;
    std::vector<std::pair<size_t, size_t> >* pvpDisparitySearchBound_;
    std::shared_ptr<std::vector<EventMatchPair> > pvEventMatchPair_;
  };

public:
  EventBM(
    CameraSystem::Ptr camSysPtr,
    size_t numThread = 1,
    bool bSmoothTS = false,
    size_t patch_size_X = 25,
    size_t patch_size_Y = 25,
    size_t min_disparity = 1,
    size_t max_disparity = 40,
    size_t step = 1,
    double ZNCC_Threshold = 0.1,
    bool bUpDownConfiguration = false);// bUpDownConfiguration determines the epipolar searching direction (UpDown or LeftRight).
  virtual ~EventBM();

  void resetParameters(
    size_t patch_size_X,
    size_t patch_size_Y,
    size_t min_disparity,
    size_t max_disparity,
    size_t step,
    double ZNCC_Threshold,
    bool bDonwUpConfiguration, 
    size_t patch_size_X_2 = 5,
    size_t patch_size_Y_2 = 31);

  void createMatchProblem(
    constStampedTimeSurfaceObs * pStampedTsObs,
    StampTransformationMap * pSt_map,
    std::vector<dvs_msgs::Event *>* pvEventsPtr);
  void createMatchProblemTwoFrames(
    constStampedTimeSurfaceObs * pStampedTsObs,
    StampTransformationMap * pSt_map,
    std::vector<dvs_msgs::Event *>* pvEventsPtr,
    std::vector<EventMatchPair> * vEMP);

  bool match_an_event(
    dvs_msgs::Event *pEvent,
    std::pair<size_t, size_t>& pDisparityBound,
    EventMatchPair &emPair);
  bool match_an_event2(
    const dvs_msgs::Event *pEvent,
    std::pair<size_t, size_t>& pDisparityBound,
    EventMatchPair &emPair);
  bool match_an_event3(
    dvs_msgs::Event *pEvent,
    std::pair<size_t, size_t>& pDisparityBound,
    EventMatchPair &emPair,
    Eigen::MatrixXd& TS_left,
    Eigen::MatrixXd& TS_right);
  bool match_an_eventTwoFrames(
    EventMatchPair EMP_lr,
    std::pair<size_t, size_t>& pDisparityBound,
    EventMatchPair &emPair);

  void match_all_SingleThread(std::vector<EventMatchPair> &vEMP);
  void match_all_SingleThreadTwoFrames(std::vector<EventMatchPair> &vEMP, std::vector<EventMatchPair> &vEMP_fail);
  void match_all_HyperThread(std::vector<EventMatchPair> &vEMP);
  void match_all_HyperThreadTwoFrames(std::vector<EventMatchPair> &vEMP, std::vector<EventMatchPair> &vEMP_fail);

  static double zncc_cost(Eigen::MatrixXd &patch_left, Eigen::MatrixXd &patch_right, bool normalized = false);
  double zncc_cost_fast(Eigen::VectorXd &colSum, Eigen::VectorXd &varSum, Eigen::MatrixXd &patch_left,
                        Eigen::MatrixXd &patch_right, int &disp_to_rm, int &step_to_rm, double &mean_l, double &Tl_2, double &Tr, double &Tr_quare);
  double zncc_cost2(Eigen::MatrixXd &patch_left, Eigen::MatrixXd &patch_right, double &var_l, double &mean_l);
  float getMSSIM( const cv::Mat& i1, const cv::Mat& i2);
private:
  void match(EventBM::Job &job);
  void match2(EventBM::Job &job);
  void match2_TwoFrames(EventBM::Job2& job);
  bool epipolarSearching(double& min_cost, Eigen::Vector2i& bestMatch, size_t& bestDisp, Eigen::MatrixXd& patch_dst,
                         size_t searching_start_pos, size_t searching_end_pos, size_t searching_step,
                         Eigen::Vector2i& x1, Eigen::MatrixXd& patch_src, bool bDownUpConfiguration = false);
  bool epipolarSearchingTwoFrames(double& min_cost, Eigen::Vector2i& bestMatch, int & bestDisp, Eigen::MatrixXd& patch_dst,
                         int searching_start_pos, int searching_end_pos, size_t searching_step,
                         Eigen::Vector2i& x1, Eigen::MatrixXd& patch_src, double dx, double dy, std::vector<double>& costs,
                         std::vector<bool>& fine_search, bool bDownUpConfiguration = false);


  bool epipolarSearchingCoarse(
  double& min_cost, Eigen::Vector2i& bestMatch, size_t& bestDisp, Eigen::MatrixXd& patch_dst,
  size_t searching_start_pos, size_t searching_end_pos, size_t searching_step,
  Eigen::Vector2i& x1, Eigen::MatrixXd& patch_src, bool bUpDownConfiguration, vector<bool>& searching_or_not,
  Eigen::VectorXd &colSum, Eigen::VectorXd &varSum, double &mean_l, double &Tl_2, double &Tr, double &Tr_quare);

  bool epipolarSearchingFine(double& min_cost, Eigen::Vector2i& bestMatch, size_t& bestDisp, Eigen::MatrixXd& patch_dst,
  vector<size_t>& searching_radius,
  Eigen::Vector2i& x1, Eigen::MatrixXd& patch_src, bool bUpDownConfiguration);

  bool isNeedSSIM(std::vector<double>& costs);
  bool isValidPatch(Eigen::Vector2i& x, Eigen::Vector2i& left_top, int size_y, int size_x);
  double triangulatePoint(Eigen::Vector2d &point0, Eigen::Vector2d &point1, double depth);
private:
  CameraSystem::Ptr camSysPtr_;
  constStampedTimeSurfaceObs* pStampedTsObs_;
  StampTransformationMap * pSt_map_;
  std::vector<dvs_msgs::Event*> vEventsPtr_;
  std::vector<std::pair<size_t, size_t> > vpDisparitySearchBound_;
  Sobel sb_;

  size_t NUM_THREAD_;
  bool bSmoothTS_;
  size_t patch_size_X_;
  size_t patch_size_Y_;
  size_t min_disparity_;
  size_t max_disparity_;
  size_t step_;
  double ZNCC_Threshold_;
  double ZNCC_MAX_;
  bool bUpDownConfiguration_;

  size_t coarseSearchingFailNum_, fineSearchingFailNum_, infoNoiseRatioLowNum_, outsideNum_, dxdyBigNum_;
  int coarseSearchingNum_ = 0;

  //to rm
  cv::Mat DispMap_;
  std::vector<EventMatchPair> vEMP_lr_;
  Eigen::Matrix4d T_last_now_;
  Eigen::Matrix3d K_inv_;
  Eigen::Matrix3d K_rtf_, K_rtf_inv_, R_ori_rtf_;
  size_t patch_size_X_2_, patch_size_Y_2_;
  int num_of_need_ssim_, num_of_success_ssim_, avg_num_of_ssim_;
  double fine_searching_time_, croase_searching_time_;// croase_searching_time_1, croase_searching_time_2, croase_searching_time_3, croase_searching_time_4, pre_time_;
  int zncc_num_;
  bool use_fast_zncc_;
  std::mutex time_mutex_;
};
}// core
}// esvo2_core

#endif //ESVO2_CORE_CORE_EVENTBM_H
