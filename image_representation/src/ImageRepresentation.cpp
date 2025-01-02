#include <fstream>
#include <image_representation/ImageRepresentation.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <std_msgs/Float32.h>
#include <glog/logging.h>

#include <cmath>
#include <vector>

// #define ESVIO_REPRESENTATION_LOG

namespace image_representation
{
  ImageRepresentation::ImageRepresentation(ros::NodeHandle &nh, ros::NodeHandle nh_private) : nh_(nh)
  {
    // setup subscribers and publishers
    event_sub_ = nh_.subscribe("events", 0, &ImageRepresentation::eventsCallback, this);
    image_transport::ImageTransport it_(nh_);
    nh_private.param<bool>("is_left", is_left_, true);    // is left camera
    if (is_left_)   
    {
      image_representation_pub_TS_ = it_.advertise("image_representation_TS_", 5);                   // for block matching
      image_representation_pub_negative_TS_ = it_.advertise("image_representation_negative_TS_", 5); // negative OS-TS for 3D-2D regristration
      image_representation_pub_AA_frequency_ = it_.advertise("image_representation_AA_frequency_", 5);
      image_representation_pub_AA_mat_ = it_.advertise("image_representation_AA_mat_", 5); // for temporal stereo matching
      dx_image_pub_ = it_.advertise("dx_image_pub_", 5);                                   // gradient map for point sampling
      dy_image_pub_ = it_.advertise("dy_image_pub_", 5);
    }
    else
    {
      image_representation_pub_TS_ = it_.advertise("image_representation_TS_", 5);
    }
    nh_private.param<bool>("use_sim_time", bUse_Sim_Time_, true);

    // system variables
    int representation_mode;
    nh_private.param<int>("representation_mode", representation_mode, 0);
    nh_private.param<int>("median_blur_kernel_size", median_blur_kernel_size_, 1);
    nh_private.param<int>("blur_size", blur_size_, 7);
    nh_private.param<int>("max_event_queue_len", max_event_queue_length_, 20);
   
    representation_mode_ = (RepresentationMode)representation_mode;
       
    // rectify variables
    bCamInfoAvailable_ = false;
    bSensorInitialized_ = false;
    sensor_size_ = cv::Size(0, 0);

    // local parameters
    nh_private.param<bool>("use_stereo_cam", bUseStereoCam_, true);
    nh_private.param<double>("decay_ms", decay_ms_, 30);
    decay_sec_ = decay_ms_ / 1000.0;
    nh_private.param<int>("x_patches", x_patches_, 8); // patch of AA
    nh_private.param<int>("y_patches", y_patches_, 6);
    nh_private.param<int>("generation_rate_hz", generation_rate_hz_, 100);
    nh_private.param("calibInfoDir", calibInfoDir_, std::string("path is not given"));
    if (!loadCalibInfo(calibInfoDir_, is_left_))
    {
      ROS_ERROR("Load Calib Info Error!!!  Given path is: %s", calibInfoDir_.c_str());
    }

    if(is_left_)
      LOG(INFO) << "\33[32m" << "Left event representation node is up " << "\33[0m";
    else
      LOG(INFO) << "\33[32m" << "Right event representation node is up " << "\33[0m";

    // start generation
    std::thread GenerationThread(&ImageRepresentation::GenerationLoop, this);
    GenerationThread.detach();
  }

  ImageRepresentation::~ImageRepresentation()
  {
    dx_image_pub_.shutdown();
    dy_image_pub_.shutdown();
    image_representation_pub_TS_.shutdown();
    image_representation_pub_negative_TS_.shutdown();
    image_representation_pub_AA_frequency_.shutdown();
    image_representation_pub_AA_mat_.shutdown();
  }

  void ImageRepresentation::init(int width, int height)
  {
    sensor_size_ = cv::Size(width, height);
    bSensorInitialized_ = true;
    ROS_INFO("Sensor size: (%d x %d)", sensor_size_.width, sensor_size_.height);

    representation_TS_ = cv::Mat::zeros(sensor_size_, CV_32F);
    representation_AA_ = cv::Mat::zeros(sensor_size_, CV_8U);

    //Access to Eigen matrix is faster than cv::Mat
    TS_temp_map = Eigen::MatrixXd::Constant(sensor_size_.height, sensor_size_.width, -10);
    vEvents_.reserve(5000000);
  }

  void ImageRepresentation::GenerationLoop()
  {
    ros::Rate r(generation_rate_hz_);
    while (ros::ok())
    {
      sync_time_ = ros::Time::now();
      {
        createImageRepresentationAtTime(sync_time_);
      }

      r.sleep();
    }
  }

  void ImageRepresentation::AA_thread(std::vector<dvs_msgs::Event>::iterator &ptr_e, int distance, double external_t)
  {
    ros::Time external_sync_time(external_t);

    representation_AA_ = cv::Mat::zeros(sensor_size_, CV_8U);   //for temporal stereo matching
    cv::Mat AA_frequency = cv::Mat::zeros(sensor_size_, CV_8U);   //for point sampling

    std::vector<double> last_activity(x_patches_ * y_patches_, 0), event_activity(x_patches_ * y_patches_, 0), beta(x_patches_ * y_patches_, 0);
    std::vector<double> last_event_time(x_patches_ * y_patches_, 0);
    std::vector<bool> flag(x_patches_ * y_patches_, true);
    int flags = 0;
    double conv_thresh_ = 0.95; // convergence threshold
    std::vector<double> final_activity(x_patches_ * y_patches_, 0);
    std::vector<int> num(x_patches_ * y_patches_, 0);

    // std::vector<int> nums_temp(x_patches_ * y_patches_, 0);
    int nums_EQ = 0;
    // calculate the final activity by all events, also can be estimated by eq. 3 in the paper
    for (auto it = vEvents_.begin(); it != ptr_e; it++)
    {
      dvs_msgs::Event e = *it;
      int y = e.y / (int)ceil((double)sensor_size_.height / (double)y_patches_);
      int x = e.x / (int)ceil((double)sensor_size_.width / (double)x_patches_);
      beta[y * x_patches_ + x] = 1 / (1 + final_activity[y * x_patches_ + x] * abs(e.ts.toSec() - last_event_time[y * x_patches_ + x])); // eq. 2
      if (y * x_patches_ + x >= x_patches_ * y_patches_)
        exit(-1);
      final_activity[y * x_patches_ + x] = beta[y * x_patches_ + x] * final_activity[y * x_patches_ + x] + 1; // eq. 1
      last_event_time[y * x_patches_ + x] = e.ts.toSec();
      // nums_temp[y * x_patches_ + x]++;
    }
    // for(int i = 0; i < x_patches_ * y_patches_; i++)
    // final_activity[i] = std::sqrt(1 / (0.01 / nums_temp[i]));  // eq. 3

    std::fill(beta.begin(), beta.end(), 0);
    std::fill(last_event_time.begin(), last_event_time.end(), 0);
    for (auto it = ptr_e; it != vEvents_.begin(); it--) // traverse events in reverse to accumulate the latest events
    {
      dvs_msgs::Event e = *it;
      int y = e.y / (int)ceil((double)sensor_size_.height / (double)y_patches_);
      int x = e.x / (int)ceil((double)sensor_size_.width / (double)x_patches_);
      if (flag[y * x_patches_ + x] != true)
        continue;
      beta[y * x_patches_ + x] = 1 / (1 + event_activity[y * x_patches_ + x] * abs(e.ts.toSec() - last_event_time[y * x_patches_ + x])); // eq. 2
      event_activity[y * x_patches_ + x] = beta[y * x_patches_ + x] * event_activity[y * x_patches_ + x] + 1;                            // eq. 1
      last_event_time[y * x_patches_ + x] = e.ts.toSec();
      AA_frequency.at<uchar>(e.y, e.x)++;
      num[y * x_patches_ + x]++;
      if (AA_frequency.at<uchar>(e.y, e.x) >= 1)
        representation_AA_.at<uchar>(e.y, e.x) = 255;
      if (num[y * x_patches_ + x] >= 10) // each patch is checked for convergence once every ten events accumulated
      {
        if (last_activity[y * x_patches_ + x] != 0)
        {
          if ((abs(event_activity[y * x_patches_ + x] - final_activity[y * x_patches_ + x])) < conv_thresh_)
          {
            flag[y * x_patches_ + x] = false;
            flags++;
            if (flags == x_patches_ * y_patches_)
              break;
            else
              continue;
          }
        }
        last_activity[y * x_patches_ + x] = event_activity[y * x_patches_ + x];
        num[y * x_patches_ + x] = 0;
      }
    }

    //distortion correction
    cv::remap(representation_AA_, representation_AA_, undistort_map1_, undistort_map2_, CV_INTER_LINEAR);

    cv_bridge::CvImage cv_AA_frequency, cv_AA_mat;
    cv_AA_frequency.encoding = "mono8";
    cv_AA_mat.encoding = "mono8";
    cv_AA_frequency.image = AA_frequency.clone();
    cv_AA_mat.image = representation_AA_.clone();
    cv_AA_frequency.header.stamp = external_sync_time;
    cv_AA_mat.header.stamp = external_sync_time;
    image_representation_pub_AA_frequency_.publish(cv_AA_frequency.toImageMsg());
    image_representation_pub_AA_mat_.publish(cv_AA_mat.toImageMsg());
  }

  void ImageRepresentation::createImageRepresentationAtTime(const ros::Time &external_sync_time)
  {
    if (!bcreat_)
      return;
    else
      bcreat_ = false;
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!bSensorInitialized_ || !bCamInfoAvailable_)
      return;
    
    //for AA generation
    cv::Mat filiter_image = cv::Mat::zeros(sensor_size_, CV_64F);
    cv::Mat rectangle_image = cv::Mat::zeros(cv::Size(80, 80), CV_8U);
    cv::Mat AA_frequency = cv::Mat::zeros(sensor_size_, CV_8U);

    if (representation_mode_ == Fast)
    {
      if (vEvents_.size() == 0)
        return;
      double external_t = external_sync_time.toSec();
      std::vector<dvs_msgs::Event>::iterator ptr_e = EventVector_lower_bound(vEvents_, external_t);
      int distance = std::distance(vEvents_.begin(), ptr_e);

      if (is_left_)   // generate AA and TS in parallel, just for left camera
      {
        std::thread thread0(&ImageRepresentation::AA_thread, this, std::ref(ptr_e), distance, external_t);
        representation_TS_.setTo(cv::Scalar(0));
        cv::Mat TS_img = cv::Mat::zeros(sensor_size_, CV_64F);

        // if the event rate is too high, we need to downsample the events
        // step = 1 indicates that we use all the events
        // double step = static_cast<double>(distance) / 90000.0;

        double step = 1;
        std::vector<dvs_msgs::Event>::iterator it = vEvents_.begin();

        // generate TS map
        for (int i = 0; i < distance; i++)
        {
          int index = static_cast<int>(i * step);
          if (index > distance - 2)
            break;
          dvs_msgs::Event e = *(it + index);
          TS_temp_map(e.y, e.x) = e.ts.toSec() / decay_sec_;
        }

        cv::eigen2cv(TS_temp_map, representation_TS_);
        representation_TS_ = representation_TS_ - external_t / decay_sec_;
        cv::exp(representation_TS_, representation_TS_);

        TS_img = representation_TS_ * 255.0;
        TS_img.convertTo(TS_img, CV_8U);

        //distortion correction
        cv::remap(TS_img, TS_img, undistort_map1_, undistort_map2_, CV_INTER_LINEAR);

        // generate OS-TS
        cv::Mat TS_img_blur;
        cv::Mat OS_TS = TS_img.clone(); 
        cv::blur(TS_img, TS_img_blur, cv::Size(blur_size_, blur_size_));
        cv::Mat mask = (TS_img == 0);
        TS_img_blur.copyTo(OS_TS, mask);
        cv::medianBlur(TS_img, TS_img, 2 * median_blur_kernel_size_ + 1);

        // generate and publish gradient map in parallel
        if (thread_sobel.joinable())
          thread_sobel.join();
        negative_TS_img = cv::Mat::ones(sensor_size_, CV_8U);
        negative_TS_img = negative_TS_img * 255;
        negative_TS_img = negative_TS_img - OS_TS;

        cv_bridge::CvImage cv_TS_image, cv_negative_TS_image;

        cv_TS_image.encoding = "mono8";
        cv_negative_TS_image.encoding = "mono8";
        cv_dx_image.encoding = sensor_msgs::image_encodings::TYPE_16SC1;
        cv_dy_image.encoding = sensor_msgs::image_encodings::TYPE_16SC1;

        cv_TS_image.header.stamp = ros::Time(external_t);
        cv_negative_TS_image.header.stamp = ros::Time(external_t);
        cv_dx_image.header.stamp = ros::Time(external_t);
        cv_dy_image.header.stamp = ros::Time(external_t);

        cv_TS_image.image = TS_img.clone();
        cv_negative_TS_image.image = negative_TS_img.clone();

        thread_sobel = std::thread(&ImageRepresentation::sobel, this, external_t);

        cv_TS_image.header.stamp = external_sync_time;
        cv_negative_TS_image.header.stamp = external_sync_time;

        image_representation_pub_TS_.publish(cv_TS_image.toImageMsg());
        image_representation_pub_negative_TS_.publish(cv_negative_TS_image.toImageMsg());
        thread0.join();
      }
      else // generate TS, just for right camera
      {
        representation_TS_.setTo(cv::Scalar(0));
        cv::Mat TS_img = cv::Mat::zeros(sensor_size_, CV_64F);

        // double step = static_cast<double>(distance) / 90000.0;
        // if (step < 1)
        double step = 1;
        std::vector<dvs_msgs::Event>::iterator it = vEvents_.begin();
        for (int i = 0; i < distance; i++)
        {
          int index = static_cast<int>(i * step);
          if (index > distance - 2)
            break;
          dvs_msgs::Event e = *(it + index);
          TS_temp_map(e.y, e.x) = e.ts.toSec() / decay_sec_;
        }
        cv::eigen2cv(TS_temp_map, representation_TS_);

        representation_TS_ = representation_TS_ - external_t / decay_sec_;
        cv::exp(representation_TS_, representation_TS_);
        TS_img = representation_TS_ * 255.0;
        TS_img.convertTo(TS_img, CV_8U);

        cv::remap(TS_img, TS_img, undistort_map1_, undistort_map2_, CV_INTER_LINEAR);

        cv::medianBlur(TS_img, TS_img, 2 * median_blur_kernel_size_ + 1);

        cv_bridge::CvImage cv_TS_image;
        cv_TS_image.encoding = "mono8";
        cv_TS_image.header.stamp = ros::Time(external_t);
        cv_TS_image.image = TS_img.clone();
        image_representation_pub_TS_.publish(cv_TS_image.toImageMsg());
      }

      clearEvents(distance, ptr_e);
    }
  }

  void ImageRepresentation::clearEvents(int distance, std::vector<dvs_msgs::Event>::iterator ptr_e)
  {
    if (vEvents_.size() > distance + 2)
      vEvents_.erase(vEvents_.begin(), ptr_e);
    else
      vEvents_.clear();
  }

  void ImageRepresentation::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
  {
    TicToc t;
    std::lock_guard<std::mutex> lock(data_mutex_);
    double t1 = t.toc();
    if (!bSensorInitialized_)
      init(msg->width, msg->height);
    for (const dvs_msgs::Event &e : msg->events)
    {
      if (e.x > sensor_size_.width || e.y > sensor_size_.height)
        continue;
      vEvents_.push_back(e);

      int i = vEvents_.size() - 2;
      while (i >= 0 && vEvents_[i].ts > e.ts)
      {
        vEvents_[i + 1] = vEvents_[i];
        i--;
      }
      vEvents_[i + 1] = e;
    }
    clearEventQueue();
    bcreat_ = true;
  }

  void ImageRepresentation::clearEventQueue()
  {
    static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 5000000;
    if (vEvents_.size() > MAX_EVENT_QUEUE_LENGTH)
    {
      size_t remove_events = vEvents_.size() - MAX_EVENT_QUEUE_LENGTH;
      vEvents_.erase(vEvents_.begin(), vEvents_.begin() + remove_events);
    }
  }

  void ImageRepresentation::sobel(double external_t)
  {
    cv::Sobel(negative_TS_img, cv_dx_image.image, CV_16SC1, 1, 0);
    cv::Sobel(negative_TS_img, cv_dy_image.image, CV_16SC1, 0, 1);
    cv_dx_image.header.stamp = ros::Time(external_t);
    cv_dy_image.header.stamp = ros::Time(external_t);
    dx_image_pub_.publish(cv_dx_image.toImageMsg());
    dy_image_pub_.publish(cv_dy_image.toImageMsg());
  }

  bool ImageRepresentation::loadCalibInfo(const std::string &cameraSystemDir, bool &is_left)
  {
    bCamInfoAvailable_ = false;
    std::string cam_calib_dir;
    if (is_left)
      cam_calib_dir = cameraSystemDir + "/left.yaml";
    else
      cam_calib_dir = cameraSystemDir + "/right.yaml";
    if (!fileExists(cam_calib_dir))
      return bCamInfoAvailable_;
    YAML::Node CamCalibInfo = YAML::LoadFile(cam_calib_dir);

    // load calib (left)
    size_t width = CamCalibInfo["image_width"].as<int>();
    size_t height = CamCalibInfo["image_height"].as<int>();
    std::string cameraNameLeft = CamCalibInfo["camera_name"].as<std::string>();
    std::string distortion_model = CamCalibInfo["distortion_model"].as<std::string>();
    std::vector<double> vD, vK, vRectMat, vP;
    std::vector<double> vT_right_left, vT_b_c;

    vD = CamCalibInfo["distortion_coefficients"]["data"].as<std::vector<double>>();
    vK = CamCalibInfo["camera_matrix"]["data"].as<std::vector<double>>();
    vRectMat = CamCalibInfo["rectification_matrix"]["data"].as<std::vector<double>>();
    vP = CamCalibInfo["projection_matrix"]["data"].as<std::vector<double>>();

    vT_right_left = CamCalibInfo["T_right_left"]["data"].as<std::vector<double>>();
    vT_b_c = CamCalibInfo["T_b_c"]["data"].as<std::vector<double>>();

    cv::Size sensor_size(width, height);
    camera_matrix_ = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        camera_matrix_.at<double>(cv::Point(i, j)) = vK[i + j * 3];

    distortion_model_ = distortion_model;
    dist_coeffs_ = cv::Mat(vD.size(), 1, CV_64F);
    for (int i = 0; i < vD.size(); i++)
      dist_coeffs_.at<double>(i) = vD[i];

    if (bUseStereoCam_)
    {
      rectification_matrix_ = cv::Mat(3, 3, CV_64F);
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          rectification_matrix_.at<double>(cv::Point(i, j)) = vRectMat[i + j * 3];

      projection_matrix_ = cv::Mat(3, 4, CV_64F);
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
          projection_matrix_.at<double>(cv::Point(i, j)) = vP[i + j * 4];

      if (distortion_model_ == "equidistant")
      {
        cv::fisheye::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
                                             rectification_matrix_, projection_matrix_,
                                             sensor_size, CV_32FC1, undistort_map1_, undistort_map2_);
        bCamInfoAvailable_ = true;
        ROS_INFO("Camera information is loaded (Distortion model %s).", distortion_model_.c_str());
      }
      else if (distortion_model_ == "plumb_bob")
      {
        cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
                                    rectification_matrix_, projection_matrix_,
                                    sensor_size, CV_32FC1, undistort_map1_, undistort_map2_);
        bCamInfoAvailable_ = true;
        ROS_INFO("Camera information is loaded (Distortion model %s).", distortion_model_.c_str());
      }
      else
      {
        ROS_ERROR_ONCE("Distortion model %s is not supported.", distortion_model_.c_str());

        return bCamInfoAvailable_;
      }

      /* pre-compute the undistorted-rectified look-up table */
      precomputed_rectified_points_ = Eigen::Matrix2Xd(2, sensor_size.height * sensor_size.width);
      // raw coordinates
      cv::Mat_<cv::Point2f> RawCoordinates(1, sensor_size.height * sensor_size.width);
      for (int y = 0; y < sensor_size.height; y++)
      {
        for (int x = 0; x < sensor_size.width; x++)
        {
          int index = y * sensor_size.width + x;
          RawCoordinates(index) = cv::Point2f((float)x, (float)y);
        }
      }
      // undistorted-rectified coordinates
      cv::Mat_<cv::Point2f> RectCoordinates(1, sensor_size.height * sensor_size.width);
      if (distortion_model_ == "plumb_bob")
      {
        cv::undistortPoints(RawCoordinates, RectCoordinates, camera_matrix_, dist_coeffs_,
                            rectification_matrix_, projection_matrix_);
        ROS_INFO("Undistorted-Rectified Look-Up Table with Distortion model: %s", distortion_model_.c_str());
      }
      else if (distortion_model_ == "equidistant")
      {
        cv::fisheye::undistortPoints(
            RawCoordinates, RectCoordinates, camera_matrix_, dist_coeffs_,
            rectification_matrix_, projection_matrix_);
        ROS_INFO("Undistorted-Rectified Look-Up Table with Distortion model: %s", distortion_model_.c_str());
      }
      else
      {
        ROS_INFO("Unknown distortion model is provided.");
        return bCamInfoAvailable_;
      }
      // load look-up table
      for (size_t i = 0; i < sensor_size.height * sensor_size.width; i++)
      {
        precomputed_rectified_points_.col(i) = Eigen::Matrix<double, 2, 1>(
            RectCoordinates(i).x, RectCoordinates(i).y);
      }
      ROS_INFO("Undistorted-Rectified Look-Up Table has been computed.");
    }
    else
    {
      // TODO: calculate undistortion map
      bCamInfoAvailable_ = true;
    }
    return bCamInfoAvailable_;
  }

  bool ImageRepresentation::fileExists(const std::string &filename)
  {
    std::ifstream file(filename);
    return file.good();
  }

} // namespace image_representation
