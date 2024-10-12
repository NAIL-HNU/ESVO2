#ifndef ESVO2_CORE_MAPPING_H
#define ESVO2_CORE_MAPPING_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <events_repacking_helper/V_ba_bg.h>

#include <tf2_ros/transform_broadcaster.h>

#include <esvo2_core/container/CameraSystem.h>
#include <esvo2_core/container/DepthMap.h>
#include <esvo2_core/container/EventMatchPair.h>
#include <esvo2_core/core/DepthFusion.h>
#include <esvo2_core/core/DepthRegularization.h>
#include <esvo2_core/core/DepthProblem.h>
#include <esvo2_core/core/DepthProblemSolver.h>
#include <esvo2_core/core/EventBM.h>
#include <esvo2_core/tools/utils.h>
#include <esvo2_core/tools/Visualization.h>

#include <dynamic_reconfigure/server.h>
#include <esvo2_core/DVS_MappingStereoConfig.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <map>
#include <deque>
#include <mutex>
#include <future>

#include <cv_bridge/cv_bridge.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

#include <sensor_msgs/Imu.h>
#include <esvo2_core/factor/imu_integration.h>
#include <esvo2_core/core/BackendOptimization.h>
namespace esvo2_core
{
  using namespace core;

  class esvo2_Mapping
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    esvo2_Mapping(const ros::NodeHandle &nh,
                  const ros::NodeHandle &nh_private);
    virtual ~esvo2_Mapping();

    // mapping
    void MappingLoop(std::promise<void> prom_mapping, std::future<void> future_reset);
    void MappingAtTime(const ros::Time &t);
    bool InitializationAtTime(const ros::Time &t);
    bool dataTransferring();

    // callback functions
    void stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg);
    void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg, EventQueue &EQ);
    void timeSurfaceCallback(
        const sensor_msgs::ImageConstPtr &time_surface_left,
        const sensor_msgs::ImageConstPtr &time_surface_right,
        const sensor_msgs::ImageConstPtr &AA_map,
        const sensor_msgs::ImageConstPtr &time_surface_negative,
        const sensor_msgs::ImageConstPtr &time_surface_dx,
        const sensor_msgs::ImageConstPtr &time_surface_dy);
    void onlineParameterChangeCallback(DVS_MappingStereoConfig &config, uint32_t level);
    void AACallback(
        const sensor_msgs::ImageConstPtr &AA_left);
    // EventQueue& EQ);
    void refImuCallback(const sensor_msgs::ImuPtr &msg);
    // utils
    bool getPoseAt(const ros::Time &t, Transformation &Tr, const std::string &source_frame);
    void clearEventQueue(EventQueue &EQ);
    void reset();

    /*** publish results ***/
    void publishMappingResults(
        DepthMap::Ptr depthMapPtr,
        Transformation tr,
        ros::Time t);
    void publishPointCloud(
        DepthMap::Ptr &depthMapPtr,
        Transformation &tr,
        ros::Time &t);
    void publishImage(
        const cv::Mat &image,
        const ros::Time &t,
        image_transport::Publisher &pub,
        std::string encoding = "bgr8");

    /*** event processing ***/
    void createEdgeMask(
        std::vector<dvs_msgs::Event *> &vEventsPtr,
        PerspectiveCamera::Ptr &camPtr,
        cv::Mat &edgeMap,
        std::vector<std::pair<size_t, size_t>> &vEdgeletCoordinates,
        bool bUndistortEvents = true,
        size_t radius = 0);

    void createDenoisingMask(
        std::vector<dvs_msgs::Event *> &vAllEventsPtr,
        cv::Mat &mask,
        size_t row, size_t col); // reserve in this file

    void extractDenoisedEvents(
        std::vector<dvs_msgs::Event *> &vCloseEventsPtr,
        std::vector<dvs_msgs::Event *> &vEdgeEventsPtr,
        cv::Mat &mask,
        size_t maxNum = 5000);

    void getReprojection(std::vector<EventMatchPair> &vEMP, Eigen::Matrix4d T_last_now, std::vector<dvs_msgs::Event *> &vDenoisedEventsPtr_left_dy_);
    void selectPoint();
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                        vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);
    void processIMU(double t, double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);

    /************************ member variables ************************/
  private:
    ros::NodeHandle nh_, pnh_;

    // Subscribers
    ros::Subscriber stampedPose_sub_, AA_frequency_sub_, events_left_sub_;
    message_filters::Subscriber<sensor_msgs::Image> TS_left_sub_, TS_right_sub_;
    message_filters::Subscriber<sensor_msgs::Image> AA_map_sub_;
    message_filters::Subscriber<sensor_msgs::Image> TS_negative_sub_, TS_dx_sub_, TS_dy_sub_;

    ros::Subscriber imu_sub_;

    // Publishers
    ros::Publisher pc_pub_, gpc_pub_, pc_filtered_pub_;
    ros::Publisher V_ba_bg_pub_;
    image_transport::ImageTransport it_;
    double t_last_pub_pc_;

    // Time-Surface sync policy
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> ExactSyncPolicy;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image,
                                                            sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image>
        ApproxSyncPolicy2;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproxSyncPolicy;
    message_filters::Synchronizer<ApproxSyncPolicy> TS_sync_;
    message_filters::Synchronizer<ApproxSyncPolicy2> TS_AA_sync_;

    // dynamic configuration (modify parameters online)
    boost::shared_ptr<dynamic_reconfigure::Server<DVS_MappingStereoConfig>> server_;
    dynamic_reconfigure::Server<DVS_MappingStereoConfig>::CallbackType dynamic_reconfigure_callback_;

    // offline data
    std::string dvs_frame_id_;
    std::string world_frame_id_;
    std::string calibInfoDir_;
    CameraSystem::Ptr camSysPtr_;

    // imu data
    double prevTime;
    bool first_imu;

    // online data
    EventQueue events_left_, events_right_;
    TimeSurfaceHistory TS_history_;
    constStampedTimeSurfaceObs *TS_obs_ptr_;
    StampTransformationMap st_map_;
    std::shared_ptr<tf::Transformer> tf_;
    size_t TS_id_;
    ros::Time tf_lastest_common_time_;

    // system
    std::string ESVO2_System_Status_;
    DepthProblemConfig::Ptr dpConfigPtr_, dpConfigPtr_ln_;
    DepthProblemSolver dpSolver_, dpSolver_ln_;
    DepthFusion dFusor_, dFusor_ln_;
    DepthRegularization dRegularizor_, dRegularizor_ln_;
    Visualization visualizor_;
    EventBM ebm_;

    // data transfer
    std::vector<dvs_msgs::Event *> vALLEventsPtr_left_;          // for BM
    std::vector<dvs_msgs::Event *> vCloseEventsPtr_left_;        // for BM
    std::vector<dvs_msgs::Event *> vDenoisedEventsPtr_left_;     // for BM
    std::vector<dvs_msgs::Event *> vDenoisedEventsPtr_left_dx_;  // for BM
    std::vector<dvs_msgs::Event *> vDenoisedEventsPtr_left_dx2_; // for BM
    std::vector<dvs_msgs::Event *> vDenoisedEventsPtr_left_dy_;  // for BM
    size_t totalNumCount_;                                       // count the number of events involved
    std::vector<dvs_msgs::Event *> vEventsPtr_left_SGM_;         // for SGM

    // result
    PointCloud::Ptr pc_near_, pc_global_;
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pc_color_, pc_filtered_;
    DepthFrame::Ptr depthFramePtr_;
    bool blarge_scale_, bpoints_from_AA_;

    // std::deque<std::vector<DepthPoint> > dqvDepthPoints_,dqvDepthPoints_ln_;
    std::deque<DepthPointFrame> dqvDepthPoints_, dqvDepthPoints_ln_;
    // DepthPointFrame

    // inter-thread management
    std::mutex data_mutex_;
    std::promise<void> mapping_thread_promise_, reset_promise_;
    std::future<void> mapping_thread_future_, reset_future_;

    /**** mapping parameters ***/
    // range and visualization threshold
    double invDepth_min_range_;
    double invDepth_max_range_;
    double cost_vis_threshold_, cost_vis_threshold_ln_;
    size_t patch_area_;
    double residual_vis_threshold_, residual_vis_threshold_ln_;
    double stdVar_vis_threshold_, stdVar_vis_threshold_ln_;
    size_t age_max_range_;
    size_t age_vis_threshold_;
    int fusion_radius_;
    std::string FusionStrategy_;
    int maxNumFusionFrames_, maxNumFusionFrames_ln_;
    int maxNumFusionPoints_;
    size_t INIT_SGM_DP_NUM_Threshold_;
    // module parameters
    size_t PROCESS_EVENT_NUM_;
    size_t PROCESS_EVENT_NUM_AA_;
    size_t TS_HISTORY_LENGTH_;
    size_t mapping_rate_hz_;
    // options
    bool changed_frame_rate_;
    bool bRegularization_;
    bool resetButton_;
    bool bDenoising_;
    bool bVisualizeGlobalPC_;
    // visualization parameters
    double visualizeGPC_interval_;
    double visualize_range_;
    size_t numAddedPC_threshold_;
    // Event Block Matching (BM) parameters
    double BM_half_slice_thickness_, eta_for_select_points_;
    size_t BM_patch_size_X_, BM_patch_size_X_2_;
    size_t BM_patch_size_Y_, BM_patch_size_Y_2_;
    size_t BM_min_disparity_;
    size_t BM_max_disparity_;
    size_t BM_step_;
    double BM_ZNCC_Threshold_;
    bool BM_bUpDownConfiguration_;
    bool bUSE_IMU_;
    // Select points from AA
    int x_patchs_, y_patchs_;

    double distance_from_last_frame_;

    // SGM parameters (Used by Initialization)
    int num_disparities_;
    int block_size_;
    int P1_;
    int P2_;
    int uniqueness_ratio_;
    cv::Ptr<cv::StereoSGBM> sgbm_;

    BackendOptimization BackendOpt_;

    queue<pair<double, Eigen::Vector3d>> accBuf;
    queue<pair<double, Eigen::Vector3d>> gyrBuf;
    bool initFirstPoseFlag;
    Eigen::Vector3d acc_0, gyr_0;
    std::mutex mBuf;

    /**********************************************************/
    /******************** For test & debug ********************/
    /**********************************************************/
    image_transport::Publisher invDepthMap_pub_, stdVarMap_pub_, ageMap_pub_, costMap_pub_, invDepthMap_rel_pub_;
    std::string resultPath_;
    // For counting the total number of fusion
    size_t TotalNumFusion_;
    double data_trans_time;
  };
}

#endif // ESVO2_CORE_MAPPING_H
