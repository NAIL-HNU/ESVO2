cmake_minimum_required(VERSION 2.8.3)
project(image_representation)

# To be consistent with the configuration in esvo2_core
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_BUILD_TYPE DEBUG)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
# set(CMAKE_CXX_FLAGS "-O0")

# set(catkin_simple_DIR ~/ros/rpg_eklt/build/catkin_simple/catkin_generated/installspace)
find_package(catkin_simple REQUIRED)

# set(IPP_DIR "/opt/intel/oneapi/ipp/2021.11/lib/cmake/ipp/")
# find_package(IPP REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(OpenCV REQUIRED)

include_directories(include ${catkin_INCLUDE_DIRS})

# make the executable
cs_add_executable(image_representation
    src/ImageRepresentation.cpp
    src/ImageRepresentation_node.cpp
)

# link the executable to the necesarry libs
target_link_libraries(image_representation
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
#    ${IPP_LIBRARIES}
   yaml-cpp
)

cs_install()