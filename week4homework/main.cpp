//
// Created by s on 2020/5/19.
//

//实现点云的下采样-》SAC分割

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

#include <pcl/features/normal_3d.h> // 法线计算
#include <pcl/kdtree/kdtree.h>//kd树搜索算法




typedef pcl::PointCloud<pcl::PointXYZ> PointCloud ;

/*
输入点云
返回一个可视化的对象
*/
pcl::visualization::PCLVisualizer::Ptr
simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----打开3维可视化窗口 加入点云----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);//背景颜色 黑se
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");//添加点云
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");//点云对象大小
    //viewer->addCoordinateSystem (1.0, "global");//添加坐标系
    viewer->initCameraParameters ();//初始化相机参数
    return (viewer);
}

float  r_euclidean = 0.1;
float  max_euclidean=3000;
float  min_euclidean=50;
float  r_seg=0.05;

// 分割点云 保存到filename_1.pcd...
void
segment(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::string filename);

// 聚类点云  保存到filename_1.pcd...
void
cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr ,pcl::PointCloud<pcl::PointXYZ>::Ptr ,std::string filename);

// 随机产生颜色
int* rand_rgb() {
    int *rgb=new int[3];
    rgb[0]=rand()%255;  // 产生255内的伪随机数
    rgb[1]=rand()%255;
    rgb[2]=rand()%255;
    return rgb;
}

void
region_cluster(PointCloud::Ptr cloud_in,PointCloud::Ptr cluster_all,std::string filename );

int
main(int argc,char** argv)
{
    // 初始化点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);  //存储源点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVoxel (new pcl::PointCloud<pcl::PointXYZ>);  //存储源点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr final1 (new pcl::PointCloud<pcl::PointXYZ>);  //存储提取的局内点

    //  读取点云
    pcl::io::loadPCDFile<pcl::PointXYZ>("homework.pcd",*cloud);
    std::cout<<"原始点云大小:"<<cloud->points.size()<<std::endl;

    //    点云下采样
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.02f, 0.02f, 0.02f);//　体素块大小　１cm
    vg.filter (*cloudVoxel);
    std::cout<<"下采样后的点云大小:"<<cloudVoxel->points.size()<<std::endl;

   // segment
   // 最后保存到这个文件下
   // cloudVoxel是分割掉平面后的点云
    std::string segPCD("seg");
    segment(cloudVoxel,segPCD);
    //最后的外点
    std::cout<<"平面分割后的外点  cloudVoxel: "<<cloudVoxel->points.size()<<std::endl;


    // 对cloudVoxel进行聚类
    // cluster全部保存在cluster_all中
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_all(new pcl::PointCloud<pcl::PointXYZ>);

    //欧式聚类
//    std::string clusterPCD ("cluster");
//    cluster(cloudVoxel,cluster_all, clusterPCD);

    // 区域生长
    std::string clusterPCD ("cluster");
    region_cluster(cloudVoxel,cluster_all, clusterPCD);

    // 保存cloud_all
    pcl::io::savePCDFile("cluster_all.pcd",*cluster_all);

//    // 可视化
//    pcl::visualization::PCLVisualizer viewer ("3D Viewer");
//    viewer.setBackgroundColor(0,0,0);
//    viewer.initCameraParameters ();
//
//    viewer.addPointCloud (cloudVoxel);//添加点云
////    viewer.spin();   // 两种方法都可以
//    while(!viewer.wasStopped())
//    {
//        viewer.spinOnce();
//    }



    return 0;

}


// 分割平面的函数
// 输入:要分割的点云
// 输出:以filename相关的分割后的点云
void
segment(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::string filename)
{

    //可视化
    pcl::visualization::PCLVisualizer viewer("Plane");

    cout<<"\n-------------Seg Start!-------------\n"<<endl;
    pcl::PCDWriter writer;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());  //创建一个PointIndices结构体指针
    // 创建分割对象
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // 可选
    seg.setOptimizeCoefficients(true); //设置对估计的模型做优化处理
    // 必选
    seg.setModelType(pcl::SACMODEL_PLANE);//设置分割模型类别
    seg.setMethodType(pcl::SAC_RANSAC);//设置使用那个随机参数估计方法
//    seg.setMaxIterations(1000);//迭代次数
    seg.setDistanceThreshold(r_seg);//设置是否为模型内点的距离阈值
    // 创建滤波器对象
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    int i = 1, nr_points = (int) cloud_in->points.size();  // 总点数
    cout << "输入点云: " << cloud_in->size() <<"\n"<< endl;
    float PointNumber = cloud_in->size();
    // 当还多于30%原始点云数据时
    while (cloud_in->points.size() > 0.1 * nr_points) {

        std::stringstream iter;
        iter<<"-----------------SegPlane_"<<i<<"-----------------";

        std::cerr<<iter.str()<<std::endl;
        // 从余下的点云中分割最大平面组成部分
        cout<<"待分割输的点云: "<<cloud_in->points.size()<<endl;
        seg.setInputCloud(cloud_in);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        break;
        }

        // 分离内层
        extract.setInputCloud(cloud_in);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_p);//将分割点云根据inliers提取到cloud_p中


        //保存
        std::cerr << "分割的内点: " << cloud_p->points.size() << std::endl;
        std::stringstream ss;
        ss << filename<<"_"<<i<<".pcd"; //对每一次的提取都进行了文件保存
        writer.write<pcl::PointXYZ>(ss.str(), *cloud_p, false);
        cerr<<ss.str()<<" saved!"<<endl;


        // 再次创建滤波器对象,提取外点
        extract.setNegative(true);//提取外层
        extract.filter(*cloud_f);//将外层的提取结果保存到cloud_f
        cout<<"分割的外点: "<<cloud_f->points.size()<<endl;

//        cloud_in.swap(cloud_f);//经cloud_filtered与cloud_f交换 这样会导致队后一次分割出现问题,点数不对
        *cloud_in=*cloud_f;  // 直接讲cloud_f外点给cloud_in 进行下一次分割


        std::cerr << "-----------------end-----------------\n\n\n" << std::endl;

        // 可视化设置颜色
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_p_color(cloud_p,
                                                                                        rand_rgb()[0],
                                                                                        rand_rgb()[1],
                                                                                        rand_rgb()[2]);
        viewer.addPointCloud(cloud_p,cloud_p_color,std::to_string(i) );

        i++;
    }
    cout<<"--------------Seg End---------------"<<endl;

    while(!viewer.wasStopped())
    {
        viewer.spinOnce(100);
    }
}


// 欧式聚类点云  保存到filename_1.pcd...
void
cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in ,pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_all,std::string filename)
{
    cout<<"\n-------------Cluster Start!-------------\n"<<endl;

    // 可视化clusters
    pcl::visualization::PCLVisualizer viewer("cluster");


    // tree
    //    pcl::search::Kd
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_in);

    // 创建Indices的vector,每一个Indices就代表一个类
    std::vector<pcl::PointIndices> cluster_indices;// 点云团索引

    // 创建欧式聚类对象
    pcl::EuclideanClusterExtraction <pcl::PointXYZ> ec;
    ec.setClusterTolerance (r_euclidean);                    // 设置近邻搜索的搜索半径为2cm
    ec.setMinClusterSize (min_euclidean);                       // 设置一个聚类需要的最少的点数目为100
    ec.setMaxClusterSize (max_euclidean);                     // 设置一个聚类需要的最大点数目为25000
    ec.setSearchMethod (tree);                        // 设置点云的搜索机制
    ec.setInputCloud (cloud_in);
    ec.extract (cluster_indices);           //从点云中提取聚类，并将点云索引保存在cluster_indices中
    cout<<"一共有 "<<cluster_indices.size()<<" 类"<<endl;

    //Extract
    // 遍历vector,提取每一类的点云
    int i=1;
    for(auto it=cluster_indices.begin();it!=cluster_indices.end();it++)
    {
        std::stringstream i_num; //计数
        i_num<<"----------------cluster_"<<i<<"------------------";
        cout<<i_num.str()<<endl;

        std::stringstream ss;  // 保存
        ss<<filename<<"_"<<i<<".pcd";

        //临时变量,存放每一个cluster
        pcl::PointCloud<pcl::PointXYZ>::Ptr P_cluster_tmp(new pcl::PointCloud<pcl::PointXYZ>);

        // 遍历cluster的每一个点的索引
        for(auto i_=it->indices.begin();i_!=it->indices.end();i_++)
        {
            P_cluster_tmp->points.push_back(cloud_in->points[*i_]);
        }

        P_cluster_tmp->width=P_cluster_tmp->points.size();
        P_cluster_tmp->height=1;
        P_cluster_tmp->is_dense=true;

        // 输出这个cluster有多少点
        std::stringstream ss_name;
        ss_name<<"cluster "<<i<<" points: "<<P_cluster_tmp->points.size();
        cout<<ss_name.str()<<endl;

        //保存这个类
        pcl::io::savePCDFile(ss.str(),*P_cluster_tmp);
        cout<<ss.str()<<" saved!"<<endl;

        // 所有的保存一块
        *cluster_all+=*P_cluster_tmp;
        std::cout<<"-----------------end-----------------\n\n\n"<<std::endl;

//         可视化这个cluster
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(P_cluster_tmp,
                                                                                         rand_rgb()[0],
                                                                                         rand_rgb()[1],
                                                                                         rand_rgb()[2]);//赋予显示点云的颜色
        viewer.addPointCloud(P_cluster_tmp,cloud_color, std::to_string(i));

        i++;
    }

    cout<<"聚类后一共有 "<<cluster_all->points.size()<<" 个点"<<endl;
    while(!viewer.wasStopped())
    {
        viewer.spinOnce(100);
    }

//    pcl::PointCloud<pcl::PointXYZ>::Ptr remain_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//    *remain_cloud = *cloud_in - *cluster_all;
}

// 使用区域生长长算法 对非平面点进行提取
void
region_cluster(PointCloud::Ptr cloud_in,PointCloud::Ptr cluster_all,std::string filename )
{

    pcl::visualization::PCLVisualizer viewer("RegionGrow");
    pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
    //求法线　和　曲率　
    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (cloud_in);
    normal_estimator.setKSearch (50);//临近50个点
    normal_estimator.compute (*normals);

    // 区域增长
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize (50);
    reg.setMaxClusterSize (3000);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (cloud_in);
    //reg.setIndices (indices);
    reg.setInputNormals (normals);
    reg.setSmoothnessThreshold (10.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold (3.0);

    std::vector <pcl::PointIndices> cluster_indices;
    reg.extract (cluster_indices);

    std::cout << "区域生长后一共有 " << cluster_indices.size () <<"类"<< std::endl;
//    std::cout << "First cluster has " << cluster_indices[0].indices.size () << " points." << std::endl;
//    std::cout << "These are the indices of the points of the initial" <<
//              std::endl << "cloud that belong to the first cluster:" << std::endl;

//    int counter = 0;
//    while (counter < cluster_indices[0].indices.size ())
//    {
//        std::cout << cluster_indices[0].indices[counter] << ", ";
//        counter++;
//        if (counter % 10 == 0)
//            std::cout << std::endl;
//    }
//    std::cout << std::endl;

    // 直接使用区域生长类中的可视化
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colord_cloud = reg.getColoredCloud();
    viewer.addPointCloud(colord_cloud);
    while(!viewer.wasStopped())
    {
        viewer.spinOnce(100);
    }

    //Extract
    // 遍历vector,提取每一类的点云
    int i=1;
    for(auto it=cluster_indices.begin();it!=cluster_indices.end();it++)
    {
        std::stringstream i_num; //计数
        i_num<<"----------------cluster_"<<i<<"------------------";
        cout<<i_num.str()<<endl;

        std::stringstream ss;  // 保存
        ss<<"./RegionGrow/";
        ss<<filename<<"_"<<i<<".pcd";

        //临时变量,存放每一个cluster
        pcl::PointCloud<pcl::PointXYZ>::Ptr P_cluster_tmp(new pcl::PointCloud<pcl::PointXYZ>);

        // 遍历cluster的每一个点的索引
        for(auto i_=it->indices.begin();i_!=it->indices.end();i_++)
        {
            P_cluster_tmp->points.push_back(cloud_in->points[*i_]);
        }

        P_cluster_tmp->width=P_cluster_tmp->points.size();
        P_cluster_tmp->height=1;
        P_cluster_tmp->is_dense=true;

        // 输出这个cluster有多少点
        std::stringstream ss_name;
        ss_name<<"cluster "<<i<<" points: "<<P_cluster_tmp->points.size();
        cout<<ss_name.str()<<endl;

        //保存这个类
        pcl::io::savePCDFile(ss.str(),*P_cluster_tmp);
        cout<<ss.str()<<" saved!"<<endl;

        // 所有的保存一块
        *cluster_all+=*P_cluster_tmp;
        std::cout<<"-----------------end-----------------\n\n\n"<<std::endl;

////         可视化这个cluster
//        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(P_cluster_tmp,
//                                                                                    rand_rgb()[0],
//                                                                                    rand_rgb()[1],
//                                                                                    rand_rgb()[2]);//赋予显示点云的颜色
//        viewer.addPointCloud(P_cluster_tmp,cloud_color, std::to_string(i));
//        viewer.addCoordinateSystem();

        i++;
    }

    cout<<"区域生长聚类后一共有 "<<cluster_all->points.size()<<" 个点"<<endl;
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
    }
}




