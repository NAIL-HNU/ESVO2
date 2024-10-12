#include <image_representation/ImageRepresentation.h>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "image_representation");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  bool left = true;
  image_representation::ImageRepresentation ts(nh, nh_private);

  ros::spin();

  return 0;
}
