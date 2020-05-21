//
// Created by s on 2020/5/20.
//


// seu.pcd 地面分割


#include <iostream>
#include <pcl/io/pcd_io.h> //io
#include <pcl/point_types.h>  // pointtype
#include <pcl/filters/voxel_grid.h>  // downsample
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>   // 平面模型
#include <pcl/visualization/pcl_visualizer.h>   // 可视化
#include <pcl/segmentation/sac_segmentation.h>  // 随机一致性分割模块
#include <pcl/segmentation/extract_clusters.h>  // 欧式聚类模块
#include <pcl/segmentation/region_growing.h>    // 区域增长算法
#include <pcl/filters/extract_indices.h>  // 根据索引提取模块
//#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp> //共享指针
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/conditional_removal.h> //条件滤波器
#include <pcl/common/transforms.h> // 点云变换模块
#include <pcl/features/normal_3d.h> // 法线计算
#include <eigen3/Eigen/Core> // eigen3
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <cmath>



typedef   pcl::PointCloud<pcl::PointXYZI>  CloudI;
typedef   pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef   pcl::PointXYZ CloudType ;



const double side_rang[2] ={-200,200.0};
const double fwd_range[2] ={-50,50.0};
const double height_range[2] ={-2,-0};





boost::shared_ptr<pcl::visualization::PCLVisualizer> createViewer()
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> v(new pcl::visualization::PCLVisualizer("seu Viewer"));
    return(v);
}

// 随机产生颜色
int* rand_rgb() {
    int *rgb=new int[3];
    rgb[0]=rand()%255;  // 产生255内的伪随机数
    rgb[1]=rand()%255;
    rgb[2]=rand()%255;
    return rgb;
}

// 法向量估计
void
normalEst(Cloud::Ptr,pcl::PointCloud<pcl::Normal>::Ptr);

// 输入点云,返回地面,非地面,法向量
bool
segment(Cloud::Ptr  ,Cloud::Ptr ,Cloud::Ptr ,Eigen::Vector3f);

//// 地面校准
//// 地面法向量,标准的地面法向量
//    Eigen::Matrix4f CreateRoatationMatrix(Eigen::Vector3f angle_before, Eigen::Vector3f angle_after);




int main(int argc,char** argv)
{
    Cloud::Ptr cloud_in(new Cloud);
    Cloud::Ptr cloud_downsample(new Cloud);
    pcl::io::loadPCDFile(argv[1],*cloud_in);
    std::cout<<"原始点云: "<<cloud_in->points.size()<<std::endl;


    // 创建滤波器对象　Create the filtering object
    pcl::VoxelGrid<CloudType> vg;
    // pcl::ApproximateVoxelGrid<pcl::PointXYZ> avg;
    vg.setInputCloud (cloud_in);//设置输入点云
    vg.setLeafSize(0.15f, 0.15f, 0.15f);//　体素块大小　１cm
    vg.filter (*cloud_downsample);
    std::cout<<"downsample后的点云: "<<cloud_downsample->points.size()<<std::endl;

    // 统计滤波
    Cloud::Ptr cloud_filtered(new Cloud);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sta;//创建滤波器对象
    sta.setInputCloud (cloud_downsample);		    //设置待滤波的点云
    sta.setMeanK (30);	     			    //设置在进行统计时考虑查询点临近点数
    sta.setStddevMulThresh (1.0);	   		    //设置判断是否为离群点的阀值
    sta.filter (*cloud_filtered); 		    //存储内点
    std::cout<<"cloud filter "<<cloud_filtered->points.size()<<std::endl;



//    // 设置ROI,使用条件滤波器
//    Cloud::Ptr cloudROI(new Cloud);
//    pcl::ConditionAnd<CloudType>::Ptr range_cond(new pcl::ConditionAnd<CloudType>);
//
//    //为条件定义对象添加比较算子
//    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
//                                                                                      pcl::FieldComparison<pcl::PointXYZ> ("x", pcl::ComparisonOps::GT, fwd_range[0])));   //添加在x字段上大于-20的比较算子
//    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
//                                                                                      pcl::FieldComparison<pcl::PointXYZ> ("x", pcl::ComparisonOps::LT, fwd_range[1])));   //添加在x字段上小于20的比较算子
//    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
//                                                                                      pcl::FieldComparison<pcl::PointXYZ> ("y", pcl::ComparisonOps::GT, side_rang[0])));   //添加在y字段上大于-50的比较算子
//    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
//                                                                                      pcl::FieldComparison<pcl::PointXYZ> ("y", pcl::ComparisonOps::LT, side_rang[1])));   //添加在y字段上小于50的比较算子
//    range_cond->addComparison(pcl::FieldComparison<CloudType>::ConstPtr (new
//                                                                                      pcl::FieldComparison<CloudType>("z",pcl::ComparisonOps::GT,height_range[0])));
//    range_cond->addComparison(pcl::FieldComparison<CloudType>::ConstPtr (new
//                                                                                      pcl::FieldComparison<CloudType>("z",pcl::ComparisonOps::LT,height_range[1])));

//    // 创建滤波器并用条件定义对象初始化
//    pcl::ConditionalRemoval<pcl::PointXYZ> condrem;
//    condrem.setCondition (range_cond);
//    condrem.setInputCloud (cloud_filtered);                   //输入点云
////    condrem.setKeepOrganized(false);               //设置保持点云的结构
//    // 执行滤波
//    condrem.filter (*cloudROI);  //两个条件用于建立滤波
//    cout<<"ROI 点云: "<<cloudROI->points.size()<<endl;
//    pcl::io::savePCDFile("Roi.pcd",*cloudROI);


//    // 去除nan点
//    Cloud::Ptr cloudROI_noNan(new Cloud);
//    std::vector<int> mapping;
//    pcl::removeNaNFromPointCloud(*cloudROI,*cloudROI_noNan,mapping);


//    // 法线估计
//    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//    normalEst(cloud_filtered,normals);

    // 平面分割
    // 创建平面
    Cloud::Ptr cloud_plane (new Cloud);
    Cloud::Ptr cloud_noPlane(new Cloud);
    Eigen::Vector3f plane_normal;  // 存放法向量
    if(segment(cloud_filtered ,cloud_plane,cloud_noPlane,plane_normal))
    {
        std::cout<<"分割完成"<<std::endl;
    }

//    // 平面校准
//    Cloud::Ptr final_cloud(new Cloud);
//    Eigen::Vector3f real_normal={0,0,1};
//    Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity();
//    rotationMatrix=CreateRoatationMatrix(plane_normal,real_normal);
//    // 利用得到的旋转矩阵,对点云进行旋转
//    pcl::transformPointCloud(*cloudROI_noNan,*final_cloud,rotationMatrix );
//    std::cout<<"rotationMatrix \n"<<rotationMatrix<<std::endl;
//    std::cout<<"final_cloud "<<final_cloud->points.size()<<std::endl;









    // 对非平面点聚类





//    // 可视化
//    pcl::visualization::PCLVisualizer viewer("seu");
//    pcl::visualization::PointCloudColorHandlerCustom<CloudType> cloud_plane_clolor(cloudROI_noNan,255,255,255);
//    pcl::visualization::PointCloudColorHandlerCustom<CloudType> cloud_no_plane_clolor(final_cloud,255,0,0);
//    viewer.addPointCloud(cloudROI_noNan,cloud_plane_clolor,"cloudROI_noNan");
//    viewer.addPointCloud(final_cloud,cloud_no_plane_clolor,"final_cloud");
//    viewer.addCoordinateSystem();
//    viewer.spin();
////    if(!viewer.wasStopped())
////    {
////        viewer.spinOnce(100);
////    }



    return 0;
}


// normal Est
// 法向量估计
void
normalEst(Cloud::Ptr cloud_in,pcl::PointCloud<pcl::Normal>::Ptr normals)
{
    pcl::search::KdTree<CloudType>::Ptr tree (new pcl::search::KdTree<CloudType> ());

    pcl::NormalEstimation<CloudType , pcl::Normal> ne;
    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud_in);
    ne.setKSearch (20);
    ne.compute (*normals);
}


// 分割出地面返回true
bool segment(Cloud::Ptr cloud_in, Cloud::Ptr cloud_Plane,Cloud::Ptr cloud_noPlane,Eigen::Vector3f plane_normal)
{
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());  // 索引
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());  //创建一个PointIndices结构体指针

    //SACsegment
//    pcl::SACSegmentationFromNormals<CloudType,pcl::Normal> seg;
    pcl::SACSegmentation<CloudType> seg;

//    seg.setInputNormals(normals);
//    seg.setNormalDistanceWeight(0.1);
    seg.setOptimizeCoefficients(true);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setDistanceThreshold(0.3);
    seg.setInputCloud(cloud_in);
    seg.segment(*inliers,*coefficients);

    if (inliers->indices.size () == 0)
    {
        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
        return false;
    }

    // 提取地面
    pcl::ExtractIndices<CloudType> extract;
    extract.setInputCloud (cloud_in);
    extract.setIndices (inliers);
    extract.filter (*cloud_Plane);

//    std::cout << "Ground cloud after filtering: " << std::endl;
//    std::cout << *cloud_Plane << std::endl;
    std::cout<<"coefficients: "<<coefficients->values[0]<<","
    <<coefficients->values[1]<<","<<coefficients->values[2]<<","
    <<coefficients->values[3]<<std::endl;

    // 保存到vector3f
    plane_normal={coefficients->values[0],coefficients->values[1],coefficients->values[2]};

    // 提取非地面点
    extract.setNegative(true);
    extract.filter(*cloud_noPlane);

    // 可视化
    pcl::visualization::PCLVisualizer viewer("planeSeg");
    pcl::visualization::PointCloudColorHandlerCustom<CloudType> cloud_plane_clolor(cloud_Plane,255,255,255);
    pcl::visualization::PointCloudColorHandlerCustom<CloudType> cloud_no_plane_clolor(cloud_noPlane,rand_rgb()[0],rand_rgb()[1],rand_rgb()[2]);
    viewer.addPointCloud(cloud_Plane,cloud_plane_clolor,"plane");
    viewer.addPointCloud(cloud_noPlane,cloud_no_plane_clolor,"cloud_noPlane");
    viewer.spin();

    return true;

}


Eigen::Matrix4f CreateRoatationMatrix(Eigen::Vector3f angle_before, Eigen::Vector3f angle_after)
{

    angle_before.normalize();
    angle_after.normalize();
    float angle = acos(angle_before.dot(angle_after));
    Eigen::Vector3f p_rotate = angle_before.cross(angle_after);
    p_rotate.normalize();
    Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity();
    rotationMatrix(0, 0)=cos(angle)+ p_rotate[0] * p_rotate[0] * (1 - cos(angle));
    rotationMatrix(0, 1) = p_rotate[0] * p_rotate[1] * (1 - cos(angle) - p_rotate[2] * sin(angle));//这里跟公式比多了一个括号，但是看实验结果它是对的。
    rotationMatrix(0, 2) = p_rotate[1] * sin(angle) + p_rotate[0] * p_rotate[2] * (1 - cos(angle));
    rotationMatrix(1, 0) = p_rotate[2] * sin(angle) + p_rotate[0] * p_rotate[1] * (1 - cos(angle));
    rotationMatrix(1, 1) = cos(angle) + p_rotate[1] * p_rotate[1] * (1 - cos(angle));
    rotationMatrix(1, 2) = -p_rotate[0] * sin(angle) + p_rotate[1] * p_rotate[2] * (1 - cos(angle));
    rotationMatrix(2, 0) = -p_rotate[1] * sin(angle) + p_rotate[0] * p_rotate[2] * (1 - cos(angle));
    rotationMatrix(2, 1) = p_rotate[0] * sin(angle) + p_rotate[1] * p_rotate[2] * (1 - cos(angle));
    rotationMatrix(2, 2) = cos(angle) + p_rotate[2] * p_rotate[2] * (1 - cos(angle));
    return rotationMatrix;
}