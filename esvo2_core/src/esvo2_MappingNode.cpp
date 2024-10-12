#include <esvo2_core/esvo2_Mapping.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "esvo2_Mapping");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  esvo2_core::esvo2_Mapping mapper(nh, nh_private);
  ros::spin();
  return 0;
}

