cmake_minimum_required(VERSION 2.8.3)
project(esvo2_core)
# explicitly set std=c++14 to remove errors from pcl library

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_CXX_FLAGS "-O3")

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++14")

# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")



find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(OpenCV REQUIRED)

FIND_PACKAGE(Sophus REQUIRED)
INCLUDE_DIRECTORIES(${Sophus_INCLUDE_DIRS})

find_package(fmt REQUIRED)
include_directories(${fmt_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

include_directories(
  ${PROJECT_SOURCE_DIR}/include)
###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)

set(HEADERS
  include/esvo2_core/core/DepthFusion.h
  include/esvo2_core/core/DepthRegularization.h
  include/esvo2_core/core/DepthProblem.h
  include/esvo2_core/core/DepthProblemSolver.h
  include/esvo2_core/core/EventBM.h
  include/esvo2_core/core/RegProblemLM.h
  include/esvo2_core/core/RegProblemSolverLM.h
  include/esvo2_core/core/BackendOptimization.h
  include/esvo2_core/factor/OptimizationFunctor.h
  include/esvo2_core/container/CameraSystem.h
  include/esvo2_core/container/DepthPoint.h
  include/esvo2_core/container/EventMatchPair.h
  include/esvo2_core/container/SmartGrid.h
  include/esvo2_core/container/DepthMap.h
  include/esvo2_core/container/ResidualItem.h
  include/esvo2_core/container/TimeSurfaceObservation.h
  include/esvo2_core/tools/Visualization.h
  include/esvo2_core/tools/utils.h
  include/esvo2_core/tools/TicToc.h
  include/esvo2_core/tools/sobel.h
  include/esvo2_core/tools/cayley.h
  include/esvo2_core/tools/params_helper.h

  include/esvo2_core/factor/utility.h

  include/esvo2_core/factor/pose_local_parameterization.h
  )

set(SOURCES
  src/core/DepthFusion.cpp
  src/core/DepthRegularization.cpp
  src/core/DepthProblem.cpp
  src/core/DepthProblemSolver.cpp
  src/core/EventBM.cpp
  src/core/RegProblemLM.cpp
  src/core/RegProblemSolverLM.cpp
  src/core/BackendOptimization.cpp
  src/container/CameraSystem.cpp
  src/container/DepthPoint.cpp
  src/container/ResidualItem.cpp
  src/tools/Visualization.cpp
  src/tools/sobel.cpp
  src/tools/cayley.cpp
  )

cs_add_library(${PROJECT_NAME}_LIB ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME}_LIB ${CERES_LIBRARIES})


# Node esvo2_Mapping
cs_add_executable(esvo2_Mapping src/esvo2_MappingNode.cpp
    src/esvo2_Mapping.cpp include/esvo2_core/esvo2_Mapping.h)
target_link_libraries(esvo2_Mapping ${PROJECT_NAME}_LIB
    ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES}  yaml-cpp)

# Node esvo2_Tracking
cs_add_executable(esvo2_Tracking src/esvo2_TrackingNode.cpp
    src/esvo2_Tracking.cpp include/esvo2_core/esvo2_Tracking.h)
target_link_libraries(esvo2_Tracking ${PROJECT_NAME}_LIB
    ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp)
