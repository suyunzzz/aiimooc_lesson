cmake_minimum_required(VERSION 2.8)

project(segment)

set(CMAKE_BUILD_TYPE Release)
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#set(CMAKE_CXX_FLAGS "-std=c++11 -03")

find_package(PCL 1.9 REQUIRED COMPONENT common io filters visualization segmentation )
find_package(Eigen3)
if ( PCL_FOUND)
    message( "\n--------------------------------------------------------\n")
    message(PCL \t${PCL_VERSION} \tFOUND!)
    message(PCL_INCLUDE_DIRS:${PCL_INCLUDE_DIRS}\n)
    message(PCL_LIBRARY_DIRS:${PCL_LIBRARY_DIRS}\n)
    message(PCL_DEFINITIONS:${PCL_DEFINITIONS})
    message(PCL_LIBRARIES:${PCL_LIBRARIES}\n-------------------------------------------------------------)
endif ( PCL_FOUND)


include_directories( ${PCL_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS})
#include_directories("/opt/ros/kinetic/include/pcl_conversions")
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )


add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES}
        )

add_executable(seu_test seu_test.cpp)
target_link_libraries(seu_test ${PCL_LIBRARIES}
        )


# 地面滤除
add_executable(ground_plane_fit ground_plane_fit.cpp)
target_link_libraries(ground_plane_fit ${PCL_LIBRARIES})

