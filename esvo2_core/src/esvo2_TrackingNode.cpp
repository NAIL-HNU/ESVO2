#include <esvo2_core/esvo2_Tracking.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "esvo2_Tracking");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  esvo2_core::esvo2_Tracking tracker(nh, nh_private);
  ros::spin();
  return 0;
}

