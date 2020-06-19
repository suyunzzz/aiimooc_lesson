//创建一了类 进行欧式聚类

#ifndef __KEYPOINT_CORE__
#define __KEYPOINT_CORE__

#include <iostream>
#include <vector>
#include <math.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_cloud.h>  // make_Shared()
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/kdtree/kdtree.h>//kd树搜索算法
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <time.h>
#include <pcl/keypoints/iss_3d.h>  // 关键点
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>





#include <pcl/filters/voxel_grid.h>    // 下采样

#include <std_msgs/Header.h> 


using pcl::NormalEstimation;
using pcl::search::KdTree;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
using namespace std;

class Keypoint_core
{
private:
    /* data */
    ros::Subscriber sub_point_cloud_;

    ros::Publisher pub_keypoints_;

    // 降采样的leaf_size
    double leaf_size = 0.3;

    // iss特征计算的邻域
    double iss_size = 0.3;

    void point_cb(const sensor_msgs::PointCloud2ConstPtr& in_cloud);

public:
    Keypoint_core(ros::NodeHandle &nh);
    ~Keypoint_core();
};

Keypoint_core::Keypoint_core(ros::NodeHandle &nh)
{   
    std::cout<<"-----------------keypoint_node start-----------------"<<std::endl;
    cout<<"leaf_size: "<<leaf_size<<", "<<"iss_size: "<<iss_size<<endl;

    sub_point_cloud_ = nh.subscribe("/rslidar_points",10, &Keypoint_core::point_cb, this);

    pub_keypoints_ = nh.advertise<sensor_msgs::PointCloud2>("/key_points", 10);

    ros::spin();

}

Keypoint_core::~Keypoint_core()
{
}

void Keypoint_core::point_cb(const sensor_msgs::PointCloud2ConstPtr& in_cloud_ptr)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_pc_ptr(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*in_cloud_ptr, *current_pc_ptr);


    clock_t start = clock();

    // 下采样
    PointCloud::Ptr cloud_src_out(new PointCloud);
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(current_pc_ptr);
    filter.setLeafSize(leaf_size,leaf_size,leaf_size);
    filter.filter(*cloud_src_out);

    //iss
    PointCloud::Ptr cloud_src_is(new PointCloud);
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_det;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_1(new pcl::search::KdTree<pcl::PointXYZ>());

    double model_solution = 0.2;

    //iss参数设置
    iss_det.setSearchMethod(tree_1);
    iss_det.setSalientRadius(iss_size);  // 0.5
    iss_det.setNonMaxRadius(0.5);
    iss_det.setThreshold21(0.975);
    iss_det.setThreshold32(0.975);
    iss_det.setMinNeighbors(5);
    iss_det.setNumberOfThreads(4);
    iss_det.setInputCloud(cloud_src_out);
    iss_det.compute(*cloud_src_is);

    clock_t end = clock();
    cout << "iss关键点提取时间：" << (double)(end - start) / CLOCKS_PER_SEC <<endl;
    cout << "iss关键点数量" << cloud_src_is->size() << endl;

    PointCloud::Ptr cloud_key(new PointCloud);
    pcl::copyPointCloud(*cloud_src_is, *cloud_key);

    sensor_msgs::PointCloud2 pub_pc;
    pcl::toROSMsg(*cloud_key, pub_pc);

    pub_pc.header = in_cloud_ptr->header;

    pub_keypoints_.publish(pub_pc);


}

#endif
