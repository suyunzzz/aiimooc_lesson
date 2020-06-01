cmake_minimum_required(VERSION 2.8.3)
project(aiimooc_syz5)

add_compile_options(-std=c++11)

find_package(PCL REQUIRED)
find_package(Eigen3 QUIET)

if (NOT EIGEN3_FOUND)
    find_package(cmake_modules REQUIRED)
    find_package(Eigen REQUIRED)
    set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
    set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  
else ()
    set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif ()

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        nav_msgs
        tf
        pcl_ros
        sensor_msgs
        pcl_conversions
        )

###################################
## catkin specific configuration ##
###################################
catkin_package(
        CATKIN_DEPENDS std_msgs  
        DEPENDS PCL
)

###########
## Build ##
###########

include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS}")


add_executable(lidarodom src/ndt_mapping_node.cpp src/ndt_mapping.cpp)
target_link_libraries(lidarodom ${catkin_LIBRARIES})
add_dependencies(lidarodom ${catkin_EXPORTED_TARGETS})





