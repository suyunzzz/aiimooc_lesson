//
// Created by s on 2020/5/21.
//

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/extract_clusters.h>  // 欧式聚类模块
#include <pcl/kdtree/kdtree.h>//kd树搜索算法


#include <Eigen/Dense>
#include <vector>
#include <ctime>
using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;

typedef pcl::PointXYZ VPoint ;

using namespace std;

// 用户排序的参数
bool point_cmp(VPoint a, VPoint b){
    return a.z<b.z;
}


// 随机产生颜色
int* rand_rgb() {
    int *rgb=new int[3];
    rgb[0]=rand()%255;  // 产生255内的伪随机数
    rgb[1]=rand()%255;
    rgb[2]=rand()%255;
    return rgb;
}

// 欧式聚类的参数
float  r_euclidean = 0.1;
float  max_euclidean=3000;
float  min_euclidean=50;

pcl::PointCloud<VPoint>::Ptr g_seeds_pc(new pcl::PointCloud<VPoint>());    // 待拟合的地面点
pcl::PointCloud<VPoint>::Ptr g_ground_pc(new pcl::PointCloud<VPoint>());   // 初始的待拟合的地面点
pcl::PointCloud<VPoint>::Ptr g_not_ground_pc(new pcl::PointCloud<VPoint>());   // 非地面点
pcl::PointCloud<VPoint>::Ptr cluster_all(new pcl::PointCloud<VPoint>);  // 聚类的总点

class GroundPlaneFit{

public:
    GroundPlaneFit(double sensor_height = 1.2 ,int num_seg=1,int num_iter=6,int num_lpr=20,
                   double th_seeds=0.2,double th_dist=0.3 ):
                   sensor_height_(sensor_height),num_seg_(num_seg),num_iter_(num_iter),num_lpr_(num_lpr),
                    th_seeds_(th_seeds),th_dist_(th_dist)
    {
        std::cout<<"-----------start--------------"<<std::endl;
    }

    void callback_(const pcl::PointCloud<VPoint> cloud_in);
    void cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in ,pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_all,std::string filename);

private:
    double sensor_height_;              // 传感器高度 去除地面下的点
    int num_seg_;
    int num_iter_;                           // 平面拟合迭代的次数
    int num_lpr_;                            // 最初是的最小的20个点
    double th_seeds_;  // 确定种子点集合的时候使用这个
    double th_dist_;   // 确定地面点

    void estimate_plane_(void);
    void extract_initial_seeds_(const pcl::PointCloud<VPoint>& p_sorted);

    // Model parameter for ground plane fitting
    // The ground plane model is: ax+by+cz+d=0
    // Here normal:=[a,b,c], d=d
    // th_dist_d_ = threshold_dist - d
    float d_;
    MatrixXf normal_;
    float th_dist_d_;


};




/*
    @brief The function to estimate plane model. The
    model parameter `normal_` and `d_`, and `th_dist_d_`
    is set here.
    The main step is performed SVD(UAV) on covariance matrix.
    Taking the sigular vector in U matrix according to the smallest
    sigular value in A, as the `normal_`. `d_` is then calculated
    according to mean ground points.

    @param g_ground_pc:global ground pointcloud ptr.

*/
void GroundPlaneFit::estimate_plane_(void){
    // Create covarian matrix in single pass.
    // TODO: compare the efficiency.
    Eigen::Matrix3f cov;
    Eigen::Vector4f pc_mean;   // 归一化坐标值
    pcl::computeMeanAndCovarianceMatrix(*g_ground_pc, cov, pc_mean); // 对地面点(最小的n个点)进行计算协方差和平均值
    // Singular Value Decomposition: SVD
    JacobiSVD<MatrixXf> svd(cov,Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));   // 取最小的特征值对应的特征向量作为法向量
    cout<<"normal_ \n"<<normal_<<endl;
    // mean ground seeds value
    Eigen::Vector3f seeds_mean = pc_mean.head<3>();   // seeds_mean 地面点的平均值
    cout<<"seeds_mean \n"<<seeds_mean<<endl;

    // according to normal.T*[x,y,z] = -d
    d_ = -(normal_.transpose()*seeds_mean)(0,0);  // 计算d   D=d
//    std::cout<<"d_: "<<d_<<std::endl;
    // set distance threhold to `th_dist - d`
    th_dist_d_ = th_dist_ - d_;   // ------------------------------? // 这里只考虑在拟合的平面上方的点 小于这个范围的点当做地面
//    std::cout<<"th_dist_d_=th_dist_ - d_ : "<<th_dist_d_<<std::endl;

    // return the equation parameters
}


/*
    @brief Extract initial seeds of the given pointcloud sorted segment.
    This function filter ground seeds points accoring to heigt.
    This function will set the `g_ground_pc` to `g_seed_pc`.
    @param p_sorted: sorted pointcloud

    @param ::num_lpr_: num of LPR points
    @param ::th_seeds_: threshold distance of seeds
    @param ::

*/
// 得到 g_seeds_pc
void GroundPlaneFit::extract_initial_seeds_(const pcl::PointCloud<VPoint>& p_sorted){   // 获得待拟合的地面点
    // LPR is the mean of low point representative
    double sum = 0;
    int cnt = 0;
    // Calculate the mean height value.
    for(size_t i=0;i<p_sorted.points.size() && cnt<num_lpr_;i++){  // 提取20个最低的点作为种子点
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double lpr_height = cnt!=0?sum/cnt:0;// in case divide by 0   // 最低的20个种子点的平均值
    g_seeds_pc->clear();
    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_   // 小于lpr_height + th_seeds_ 的点作为待拟合的地面点
    for(size_t i=0;i<p_sorted.points.size();i++){
        if(p_sorted.points[i].z < lpr_height + th_seeds_){
            g_seeds_pc->points.push_back(p_sorted.points[i]);
        } else {
            break;  // 因为是按顺序遍历的 所以可以直接退出循环
        }
    }
    // return seeds points
}

void GroundPlaneFit::callback_(pcl::PointCloud<VPoint> cloud_in) {
    // 1. 复制一份点云
    pcl::PointCloud<VPoint> cloud_org(cloud_in);  // 复制一份点云
    cout << "点云复制完成" << endl;

    // 2.Sort on Z-axis value
    sort(cloud_in.points.begin(), cloud_in.end(), point_cmp);  // 排序

    // 3.Error point removal
    // As there are some error mirror reflection under the ground,
    // here regardless point under 2* sensor_height
    // Sort point according to height, here uses z-axis in default
    // 清除 在地面下的点
    pcl::PointCloud<VPoint>::iterator it = cloud_in.points.begin();
    for (size_t i = 0; i < cloud_in.points.size(); i++) {    // 统计小于-1.5*sensor_height_的点数目
        if (cloud_in.points[i].z < -1.5 * sensor_height_) {
            it++;
        } else {
            break;
        }
    }
    cloud_in.points.erase(cloud_in.points.begin(), it);  // 清除 在地面下的点 ,因为之前是根据z排序了 所以很方便

    std::cout<<"清除地面下的点后的点数: " <<cloud_in.points.size()<<std::endl;

    // 4. Extract init ground seeds.
    extract_initial_seeds_(cloud_in);  // 获得待拟合的地面点
    g_ground_pc = g_seeds_pc;   // 待拟合的地面点

    // 5. Ground plane fitter mainloop
    for (int i = 0; i < num_iter_; i++) {    // 迭代3次

        cout<<" -----------iter"<<"["<<i+1<<"]"<<"------------" <<endl;
        estimate_plane_();     // g_ground_pc 进行平面拟合 得到法向量normal_ 和 th_dist_d_
        g_ground_pc->clear();
        g_not_ground_pc->clear();

        //pointcloud to matrix
        MatrixXf points(cloud_org.points.size(), 3);
        int j = 0;
        for (auto p:cloud_org.points) {
            points.row(j++) << p.x, p.y, p.z;
        }
//        cout << "点云--->矩阵: " << points << endl;

        // ground plane model
        VectorXf result = points * normal_;  // result=Ax+By+Cz
        // threshold filter
        for (int r = 0; r < result.rows(); r++) {

//            std::cout << "点到平面的距离:\n" << result[r] + d_ << std::endl;  // 按理说应该都是大于0的,都在拟合的平面上方

            if (result[r] < th_dist_d_) {      // 到拟合的平面的距离小于th_dist_的点 作为最后的地面点
//                g_all_pc->points[r].label = 1u;// means ground
                g_ground_pc->points.push_back(cloud_org[r]);
            } else {                          // 非地面点
//                g_all_pc->points[r].label = 0u;// means not ground and non clusterred
                g_not_ground_pc->points.push_back(cloud_org[r]);
            }
        }

        // 每次迭代输出结果
        cout<<"["<<i+1<<"]"<<" "<<"地面点: "<<g_ground_pc->points.size()<<", "<<"非地面点: "<<g_not_ground_pc->points.size()<<"\n\n\n"<<endl;
    }

}


int main(int argc,char** argv)
{

    clock_t start,end0,end;
    // 1.读入点云 2.创建对象 3. callback_
    pcl::PointCloud<VPoint>::Ptr cloud_read(new pcl::PointCloud<VPoint>);
    pcl::io::loadPCDFile(argv[1],*cloud_read);
    cout<<"原始点云数目: "<<cloud_read->points.size()<<endl;


    start = clock();
    GroundPlaneFit core;

    core.callback_(*cloud_read);


    end = clock();

    cout<<"分割地面花费时间: " << (double)(end-start)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
//    std::cout<<"g_ground_pc: "<<*g_ground_pc<<std::endl;
//    std::cout<<"g_not_ground_pc: "<<*g_not_ground_pc<<std::endl;

//    //可视化
//    pcl::visualization::PCLVisualizer viewer("seu");
//    pcl::visualization::PointCloudColorHandlerCustom<VPoint> cloud_plane_clolor(g_ground_pc,255,255,255);
//    pcl::visualization::PointCloudColorHandlerCustom<VPoint> cloud_no_plane_clolor(g_not_ground_pc,255,0,0);
//    viewer.addPointCloud(g_ground_pc,cloud_plane_clolor,"cloudROI_noNan");
//    viewer.addPointCloud(g_not_ground_pc,cloud_no_plane_clolor,"final_cloud");
//    viewer.addCoordinateSystem();

    //解决width*height=0的问题

    start = clock();

    g_ground_pc->resize(g_ground_pc->points.size());
    g_not_ground_pc->resize(g_not_ground_pc->points.size());

    cout<<"地面: "<<g_ground_pc->points.size()<<endl;
    cout<<"非地面: "<<g_not_ground_pc->points.size()<<endl;

    pcl::io::savePCDFile("g_ground_pc.pcd",*g_ground_pc);
    pcl::io::savePCDFile("g_not_ground_pc.pcd",*g_not_ground_pc);

    // 去除nan点
    std::vector<int> mapping;
    pcl::PointCloud<pcl::PointXYZ> no_ground_noNan(*g_not_ground_pc);
    no_ground_noNan.is_dense= false;
    cout<<"no_ground_noNan "<<no_ground_noNan.points.size()<<endl;
    pcl::removeNaNFromPointCloud(no_ground_noNan,no_ground_noNan,mapping);
    cout<<"no_ground_noNan: "<<no_ground_noNan.points.size()<<endl;
//    cout<<no_ground_noNan<<endl;

    end0=clock();



//     聚类
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_all (new pcl::PointCloud<pcl::PointXYZ>);
    core.cluster(no_ground_noNan.makeShared(),cluster_all,"./clusters/cluster");  // 保存到clusters文件夹下
    pcl::io::savePCDFile("./clusters/cluster_all.pcd",*cluster_all);



    end = clock();
    cout<<"去除nan点用的时间: "<<(double)(end0-start)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
    cout<<"聚类花费时间: "<<(double)(end-start)/CLOCKS_PER_SEC*1000<<"ms"<<endl;


//    viewer.spin();
    return 0;

}


// 欧式聚类点云  保存到filename_1.pcd...
void
GroundPlaneFit::cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in ,pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_all,std::string filename)
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
    ec.setClusterTolerance (0.3);                    // 设置近邻搜索的搜索半径为2cm
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

        // 可视化这个cluster
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


}


