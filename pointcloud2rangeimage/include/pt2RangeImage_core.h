#pragma once
#include <iostream>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_spherical.h>
#include <pcl/point_types_conversion.h>
#include <pcl/conversions.h>
#include <pcl/visualization/pcl_visualizer.h>       // 可视化
#include <pcl/visualization/range_image_visualizer.h>       
#include <pcl/visualization/common/float_image_utils.h>         // 转化为矩阵


#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>



using namespace std;


typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloudType;


class pt2RI
{

public:
    pt2RI(ros::NodeHandle&);

    pt2RI(ros::NodeHandle&,float,float,float,float,float,float);


    ~pt2RI();


    // 回调函数
    void point_cb(const sensor_msgs::PointCloud2ConstPtr& in_cloud);   // 回调函数
    




private:
    // 订阅者
    ros::Subscriber _suber;

    // 发布者
    ros::Publisher _publisher;

    // 转换后的点云
    PointCloudType::Ptr _pCloud;

    // 转换后的rangeImage
    pcl::RangeImage::Ptr _pRI;

    // RangeImage frame
    pcl::RangeImage::CoordinateFrame _frame = pcl::RangeImage::LASER_FRAME;
    // RangeImage resolution
    float _ang_res_x;
    float _ang_res_y;
    // RangeImage angular FoV
    float _max_ang_w;
    float _max_ang_h;
    // Sensor min/max range
    float _min_range;
    float _max_range;
    Eigen::Affine3f _camera_pose = Eigen::Affine3f::Identity();  // 相机位姿，pcl生成range_image函数会用到



    pcl::visualization::RangeImageVisualizer _range_image_viser;

    float* p_rangeimage = NULL;
 



};  // class pt2RI