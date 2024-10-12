#ifndef ESVO2_CORE_CORE_DEPTHREGULARIZATION_H
#define ESVO2_CORE_CORE_DEPTHREGULARIZATION_H

#include <esvo2_core/container/DepthMap.h>
#include <esvo2_core/core/DepthProblem.h>
#include <memory>
namespace esvo2_core
{
using namespace container;
namespace core
{
class DepthRegularization
{
public:
  typedef std::shared_ptr<DepthRegularization> Ptr;

  DepthRegularization(std::shared_ptr<DepthProblemConfig> & dpConfigPtr);
  virtual ~DepthRegularization();

  void apply( DepthMap::Ptr & depthMapPtr );

private:
  std::shared_ptr<DepthProblemConfig> dpConfigPtr_;
  size_t _regularizationRadius;
  size_t _regularizationMinNeighbours;
  size_t _regularizationMinCloseNeighbours;
};
}// core
}// esvo2_core

#endif //ESVO2_CORE_CORE_DEPTHREGULARIZATION_H