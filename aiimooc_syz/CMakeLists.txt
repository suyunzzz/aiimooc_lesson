cmake_minimum_required(VERSION 2.8.3)
project(aiimooc_syz)

add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
pcl_ros
roscpp
sensor_msgs
cv_bridge
geometry_msgs
image_transport
)

## Generate dynamic reconfigure parameters in the 'cfg' folder
 generate_dynamic_reconfigure_options(
   cfg/RangeImage.cfg
#   cfg/DynReconf2.cfg
 )

set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(PCL 1.9 REQUIRED)
find_package(OpenCV REQUIRED)


catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros cv_bridge geometry_msgs  image_transport
  DEPENDS PCL
)


include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
 ${OpenCV_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})


add_executable(${PROJECT_NAME}_node
  src/pt2image_core.cpp
  src/pt2image_node.cpp
)

add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_gencfg)  # dynamic reconfigure 新加的依赖


target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenVC_LIBRARIES}
)


