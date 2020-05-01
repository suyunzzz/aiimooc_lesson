// 库文件 Pcl2ImgCore类定义


#pragma once

#include <math.h>
#include <ros/ros.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_spherical.h>

#include <opencv2/core/core.hpp>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <image_transport/image_transport.h>

#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>

#include <dynamic_reconfigure/server.h>
#include <aiimooc_syz/RangeImageConfig.h>  //dynamic reconfigure生成的头文件

// namespace 
// {
//      typedef aiimooc_syz::RangeImageConfig conf;
//      typedef dynamic_reconfigure::Server<conf>               RangeImageReconfServer;
// }



class Pcl2ImgCore
{
    private:
        ros::Subscriber sub_point_cloud_;
        ros::Publisher pub_Img_;
        // boost::shared_ptr<RangeImageReconfServer> drsv_;
        // RangeImage frame
        pcl::RangeImage::CoordinateFrame _frame;
        // RangeImage resolution
        float _ang_res_x;
        float _ang_res_y;
        // RangeImage angular FoV
        float _max_ang_w;
        float _max_ang_h;
        // Sensor min/max range
        float _min_range;
        float _max_range;
        Eigen::Affine3f camera_pose;  // 相机位姿，pcl生成depth_image函数会用到

        pcl::PointCloud<pcl::PointXYZ>::Ptr current_pc_ptr;  //转换后的点云数据 
        pcl::RangeImageSpherical::Ptr range_image;   // pointcloud生成的range_image

        sensor_msgs::ImagePtr msg;  // 最后发布的消息格式
        cv::Mat _rangeImage;  // rangeimage转成图片才能以msg发送出去
        std::string encoding ="mono16" ;   // 编码格式




    public:
        Pcl2ImgCore(ros::NodeHandle &nh); //构造函数

    
        ~Pcl2ImgCore();  // 析构函数
        void point_cb(const sensor_msgs::PointCloud2ConstPtr& in_cloud);   // 回调函数

        void Spin();

    private:
    
        // dynamic reconfigure 的回调函数
        void dynamic_callback(aiimooc_syz::RangeImageConfig &config, uint32_t level); 

};