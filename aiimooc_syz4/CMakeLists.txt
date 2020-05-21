# cmake_minimum_required(VERSION 2.8.3)
# project(aiimooc_syz4)

# add_compile_options(-std=c++11)


# find_package(catkin REQUIRED COMPONENTS
# pcl_ros
# roscpp
# sensor_msgs
# )

# ## Generate dynamic reconfigure parameters in the 'cfg' folder
# # generate_dynamic_reconfigure_options(
# #   cfg/RangeImage.cfg
# ##   cfg/DynReconf2.cfg
# # )


# find_package(PCL 1.9 REQUIRED)



# catkin_package(
#   INCLUDE_DIRS include
#   CATKIN_DEPENDS roscpp sensor_msgs pcl_ros    
#   DEPENDS PCL
# )


# include_directories(
#   include ${catkin_INCLUDE_DIRS}
#  include
#  ${catkin_INCLUDE_DIRS}
#  ${PCL_INCLUDE_DIRS}
#  # ${OpenCV_INCLUDE_DIRS}
# )
# link_directories(${PCL_LIBRARY_DIRS})
# # link_directories(${OpenCV_LIBRARY_DIRS})


# add_executable(${PROJECT_NAME}_node
#   #src/pt2image_core.cpp
#   src/pcl_node.cpp
# )

# # add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_gencfg)  # dynamic reconfigure 新加的依赖


# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
#  # ${OpenVC_LIBRARIES}
# )

cmake_minimum_required(VERSION 2.8.3)
project(aiimooc_syz4)

add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
pcl_ros
roscpp
sensor_msgs
# Eigen3

)

## Generate dynamic reconfigure parameters in the 'cfg' folder
 generate_dynamic_reconfigure_options(
   cfg/FitPlane.cfg
#   cfg/DynReconf2.cfg
 )

# set(PCL_DIR /home/s/pcl/release)
# set(PCL_DIR /usr/share/pcl-1.9) 
find_package(PCL 1.9 REQUIRED)
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
#  ${OpenCV_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
# link_directories(${OpenCV_LIBRARY_DIRS})


add_executable(${PROJECT_NAME}_node
  src/pcl_node.cpp
)

# add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_gencfg)  # dynamic reconfigure 新加的依赖


target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  # ${OpenVC_LIBRARIES}
)






