

cmake_minimum_required(VERSION 2.8.3)
project(week6homework)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_compile_options(-std=c++11)
# set(CMAKE_BUILD_TYPE "Release")

find_package(catkin REQUIRED COMPONENTS
pcl_ros
roscpp
sensor_msgs
std_msgs

# Eigen3

)




# set(PCL_DIR /home/s/pcl/release)
# set(PCL_DIR /usr/share/pcl-1.9) 
find_package(PCL 1.7 REQUIRED)   # 使用pcl 1.9在欧式聚类设置KdTree的时候会报错!!!
# find_package(OpenCV REQUIRED)

message(STATUS "PCL_DIR = ${PCL_DIR}")
message(STATUS "PCL_INCLUDE_DIRS = ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL_LIBS = ${PCL_LIBRARIES}")


catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros 
  DEPENDS PCL
)


include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
  /usr/lib/cmake/eigen3
)
link_directories(${PCL_LIBRARY_DIRS})


# 平面分割
add_executable(keypoint_node src/keypoint_node.cpp)
target_link_libraries(keypoint_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
















