#ifndef ESVO2_CORE_TOOLS_CAYLEY_H
#define ESVO2_CORE_TOOLS_CAYLEY_H

#include <Eigen/Eigen>
namespace esvo2_core
{
namespace tools
{
Eigen::Matrix3d cayley2rot( const Eigen::Vector3d & cayley);
Eigen::Vector3d rot2cayley( const Eigen::Matrix3d & R );
}// namespace tools
}// namespace esvo2_core


#endif //ESVO2_CORE_TOOLS_CAYLEY_H
