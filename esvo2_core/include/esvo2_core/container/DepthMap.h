#ifndef ESVO2_CORE_DEPTHMAP_H
#define ESVO2_CORE_DEPTHMAP_H

#include <esvo2_core/container/DepthPoint.h>
#include <esvo2_core/container/SmartGrid.h>
#include <esvo2_core/tools/utils.h>

namespace esvo2_core
{
using namespace tools;
namespace container
{
using DepthMap = SmartGrid<DepthPoint>;

struct DepthFrame
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<DepthFrame> Ptr;

  DepthFrame(size_t row, size_t col)
  {
    dMap_ = std::make_shared<DepthMap>(row, col);
    id_ = 0;
    T_world_frame_.setIdentity();
  }

  void setId(size_t id)
  {
    id_ = id;
  }

  void setTransformation(Transformation &T_world_frame)
  {
    T_world_frame_ = T_world_frame;
  }

  void clear()
  {
    dMap_->reset();
    id_ = 0;
    T_world_frame_.setIdentity();
  }

  DepthMap::Ptr dMap_;
  size_t id_;
  Transformation T_world_frame_;
};
}
}
#endif //ESVO2_CORE_DEPTHMAP_H
