#include <esvo2_core/container/ResidualItem.h>

esvo2_core::container::ResidualItem::ResidualItem() = default;

esvo2_core::container::ResidualItem::ResidualItem(
  const double x,
  const double y,
  const double z)
{
  initialize(x,y,z);
}

void esvo2_core::container::ResidualItem::initialize(
  const double x,
  const double y,
  const double z)
{
  p_ = Eigen::Vector3d(x,y,z);
//  bOutlier_ = false;
//  variance_ = 1.0;
}