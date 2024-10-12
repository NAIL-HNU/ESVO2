#ifndef ESVO2_CORE_TRACKING_H
#define ESVO2_CORE_TRACKING_H

#include <nav_msgs/Path.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Imu.h>

#include <tf2_ros/transform_broadcaster.h>

#include <esvo2_core/container/CameraSystem.h>
#include <esvo2_core/core/RegProblemLM.h>
#include <esvo2_core/core/RegProblemSolverLM.h>
#include <esvo2_core/tools/utils.h>
#include <esvo2_core/tools/Visualization.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>

#include <map>
#include <deque>
#include <mutex>
#include <future>
#include <vector>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

#include <esvo2_core/factor/imu_integration.h>
#include <events_repacking_tool/V_ba_bg.h>

namespace esvo2_core
{
  using namespace core;
  using namespace factor;
  enum TrackingStatus
  {
    IDLE,
    WORKING
  };

  class esvo2_Tracking
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    esvo2_Tracking(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);
    virtual ~esvo2_Tracking();

    // functions regarding tracking
    void TrackingLoop();
    bool refDataTransferring();
    bool curDataTransferring(); // These two data transferring functions are decoupled because the data are not updated at the same frequency.
    bool curImuTransferring();
    // topic callback functions
    void refMapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void refImuCallback(const sensor_msgs::ImuPtr &msg);
    void VBaBgCallback(const events_repacking_tool::V_ba_bg &msg);
    void groundTruthCallback(const geometry_msgs::PoseStampedConstPtr &msg);
    void timeSurface_NegaTS_Callback(
        const sensor_msgs::ImageConstPtr &time_surface_left,
        const sensor_msgs::ImageConstPtr &time_surface_negative,
        const sensor_msgs::ImageConstPtr &time_surface_dx,
        const sensor_msgs::ImageConstPtr &time_surface_dy);
    void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg);

    // results
    void publishPose(const ros::Time &t, Transformation &tr);
    void publishPath(const ros::Time &t, Transformation &tr);
    void saveTrajectory(std::string &resultDir);

    // utils
    void reset();
    void clearEventQueue();
    void stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &msg);
    bool getPoseAt(
        const ros::Time &t,
        esvo2_core::Transformation &Tr, // T_world_something
        const std::string &source_frame);
    void renameOldTraj();
    Eigen::Matrix3d fixRotationMatrix(const Eigen::Matrix3d &R);

  private:
    ros::NodeHandle nh_, pnh_;
    image_transport::ImageTransport it_;

    // subscribers and publishers
    // ros::Subscriber events_left_sub_;
    ros::Subscriber map_sub_, map_sub_for_tracking_visualization_;
    ros::Subscriber V_ba_bg_sub_;
    ros::Subscriber imu_sub_;
    ros::Subscriber gt_sub_;

    message_filters::Subscriber<sensor_msgs::Image> TS_left_sub_, TS_right_sub_;
    message_filters::Subscriber<sensor_msgs::Image> TS_negative_sub_, TS_dx_sub_, TS_dy_sub_;
    ros::Subscriber stampedPose_sub_;
    image_transport::Publisher reprojMap_pub_left_;

    // publishers
    ros::Publisher pose_pub_, path_pub_;

    // results
    nav_msgs::Path path_;
    std::list<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>> lPose_;
    std::list<std::string> lTimestamp_;

    // Time Surface sync policy
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproximateSyncPolicy;
    message_filters::Synchronizer<ApproximateSyncPolicy> TS_sync_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> ApproximateSyncPolicy_negaTS;
    message_filters::Synchronizer<ApproximateSyncPolicy_negaTS> TS_negaTS_sync_;

    // offline data
    std::string dvs_frame_id_;
    std::string world_frame_id_;
    std::string calibInfoDir_;
    CameraSystem::Ptr camSysPtr_;

    // inter-thread management
    std::mutex data_mutex_;
    std::mutex imu_mutex_;

    // imu data
    IntegrationBase imu_data_;
    Eigen::Vector3d g_optimal{0, 9.81, 0};
    Eigen::Matrix3d R_b_c_;
    Eigen::Vector3d ba_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    Eigen::Vector3d bg_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    Eigen::Quaterniond Imu_q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d Imu_t = Eigen::Vector3d::Zero();

    // online data
    EventQueue events_left_;
    TimeSurfaceHistory TS_history_;
    size_t TS_id_;
    std::shared_ptr<tf::Transformer> tf_;
    std::map<ros::Time, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr> refPCMap_;
    RefFrame ref_;
    CurFrame cur_;

    /**** offline parameters ***/
    size_t tracking_rate_hz_;
    size_t TS_HISTORY_LENGTH_;
    size_t REF_HISTORY_LENGTH_;
    bool bSaveTrajectory_;
    bool bVisualizeTrajectory_;
    bool bUseImu_;
    std::string resultPath_;

    Eigen::Matrix<double, 4, 4> T_world_ref_;
    Eigen::Matrix<double, 4, 4> T_world_cur_;

    Eigen::Vector3d t_world_cur_;
    Eigen::Vector3d last_t_world_cur_;
    Eigen::Vector3d last_t_;

    /*** system objects ***/
    RegProblemType rpType_;
    TrackingStatus ets_;
    std::string ESVO2_System_Status_;
    RegProblemConfig::Ptr rpConfigPtr_;
    RegProblemSolverLM rpSolver_;
    bool initVsFlag;

    /*** for test ***/

    std::vector<Eigen::Vector3d> qprevious_ts_;
  };
}

#endif // ESVO2_CORE_TRACKING_H
