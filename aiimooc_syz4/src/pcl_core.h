#ifndef __PCL_CORE__
#define __PCL_CORE__

#include <math.h>
#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/filters/filter.h> 
#include <pcl/common/centroid.h>  // 计算种子点中心
#include <pcl/filters/voxel_grid.h>

#include <eigen3/Eigen/Dense>
#include <vector>
#include <ctime>

#include <dynamic_reconfigure/server.h>
#include <aiimooc_syz4/FitPlaneConfig.h>  //dynamic reconfigure生成的头文件

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

pcl::PointCloud<VPoint>::Ptr g_seeds_pc(new pcl::PointCloud<VPoint>());    // 待拟合的地面点
pcl::PointCloud<VPoint>::Ptr g_ground_pc(new pcl::PointCloud<VPoint>());   // 初始的待拟合的地面点
pcl::PointCloud<VPoint>::Ptr g_not_ground_pc(new pcl::PointCloud<VPoint>());   // 非地面点


double sensor_height = 1.2 ;
int num_seg=1;
int num_iter=6;
int num_lpr=20;
double th_seeds=0.2;
double th_dist=0.3;

class GroundFit
{
public:
    GroundFit(ros::NodeHandle &nh):
        sub(nh.subscribe("rslidar_points",10,&GroundFit::call_back,this)),
        pub_plane(nh.advertise<sensor_msgs::PointCloud2>("ground_points",10)),
        pub_no_plane(nh.advertise<sensor_msgs::PointCloud2>("no_ground_points",10)),

        sensor_height_(sensor_height),
        num_seg_(num_seg),
        num_iter_(num_iter),
        num_lpr_(num_lpr),
        th_seeds_(th_seeds),
        th_dist_(th_dist)
        {   
            std::cout<<"-----------start--------------"<<std::endl;
            // dynamic reconfigure   将这个回调函数放在构造函数内部
            dynamic_reconfigure::Server<aiimooc_syz4::FitPlaneConfig> server;
            dynamic_reconfigure::Server<aiimooc_syz4::FitPlaneConfig>::CallbackType callback;
        
            callback = boost::bind(&GroundFit::dynamic_callback, this,_1,_2);   // 调用dynamic reconfigure的回调函数
            server.setCallback(callback);

            ros::spin();  // 构造函数最后加 ros::spin()
        }

    ~GroundFit()
    {}

    // void callback(const sensor_msgs::PointCloud2ConstPtr & in_cloud_ptr);
    
  


private:
    /* data */
    ros::Subscriber sub ;
    ros::Publisher pub_plane;
    ros::Publisher pub_no_plane;

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
    float d_;            // ax+by+cz+d=0
    MatrixXf normal_;    // normal:=[a,b,c]
    float th_dist_d_;   // th_dist_d_ = threshold_dist - d

    void call_back(const sensor_msgs::PointCloud2ConstPtr& in_cloud_ptr);
    // dynamic reconfigure 的回调函数
    void dynamic_callback(aiimooc_syz4::FitPlaneConfig &config, uint32_t level); 
    
};

void 
GroundFit::call_back(const sensor_msgs::PointCloud2ConstPtr& in_cloud_ptr)
{
    // std::cout<<"-----------------start-------------------"<<std::endl;
    pcl::PointCloud<pcl::PointXYZ> cloud_in;
    pcl::fromROSMsg(*in_cloud_ptr,cloud_in);

    // 1. 复制一份点云
    pcl::PointCloud<VPoint> cloud_org(cloud_in);  // 复制一份点云
    // cout << "点云复制完成" << endl;

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
    // std::cout<<"清除地面下的点后的点数: " <<cloud_in.points.size()<<std::endl;

    // 4. Extract init ground seeds.
    extract_initial_seeds_(cloud_in);  // 获得待拟合的地面点
    g_ground_pc = g_seeds_pc;   // 待拟合的地面点

    // 5. Ground plane fitter mainloop
    for (int i = 0; i < num_iter_; i++) {    // 迭代3次

        // cout<<" -----------iter"<<"["<<i+1<<"]"<<"------------" <<endl;
        estimate_plane_();     // g_ground_pc 进行平面拟合 得到法向量normal_ 和 th_dist_d_
        g_ground_pc->clear();
        g_not_ground_pc->clear();

        //pointcloud to matrix
        MatrixXf points(cloud_org.points.size(), 3);
        int j = 0;
        for (auto p:cloud_org.points) {
            points.row(j++) << p.x, p.y, p.z;
        }

        // 得到所有点到平面的距离相关的 result
        // ground plane model
        VectorXf result = points * normal_;  // result=Ax+By+Cz
        // threshold filter
        for (int r = 0; r < result.rows(); r++) {
            if (result[r] < th_dist_d_) {      // 到拟合的平面的距离小于th_dist_的点 作为最后的地面点
                g_ground_pc->points.push_back(cloud_org[r]);
            } else {                          // 非地面点
                g_not_ground_pc->points.push_back(cloud_org[r]);
            }
        }

        // 每次迭代输出结果
        // cout<<"["<<i+1<<"]"<<" "<<"地面点: "<<g_ground_pc->points.size()<<", "<<"非地面点: "<<g_not_ground_pc->points.size()<<"\n\n\n"<<endl;
    }
    

    // 发布
    sensor_msgs::PointCloud2 ground_pt,no_ground_pt;
    pcl::toROSMsg(*g_ground_pc, ground_pt);
    pcl::toROSMsg(*g_not_ground_pc,no_ground_pt);
    ground_pt.header = in_cloud_ptr->header;
    no_ground_pt.header = in_cloud_ptr->header;
    pub_plane.publish(ground_pt);
    pub_no_plane.publish(no_ground_pt);


    // std::cout<<"----------------end------------------"<<std::endl;
}




// 更新拟合平面的A B C D
void
GroundFit::estimate_plane_()
{
    // Create covarian matrix in single pass.
    // TODO: compare the efficiency.
    Eigen::Matrix3f cov;
    Eigen::Vector4f pc_mean;   // 归一化坐标值
    pcl::computeMeanAndCovarianceMatrix(*g_ground_pc, cov, pc_mean); // 对地面点(最小的n个点)进行计算协方差和平均值
    // Singular Value Decomposition: SVD
    JacobiSVD<MatrixXf> svd(cov,Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));   // 取最小的特征值对应的特征向量作为法向量
    // cout<<"normal_ \n"<<normal_<<endl;
    // mean ground seeds value
    Eigen::Vector3f seeds_mean = pc_mean.head<3>();   // seeds_mean 地面点的平均值
    // cout<<"seeds_mean \n"<<seeds_mean<<endl;

    // according to normal.T*[x,y,z] = -d
    d_ = -(normal_.transpose()*seeds_mean)(0,0);  // 计算d   D=d
//    std::cout<<"d_: "<<d_<<std::endl;
    // set distance threhold to `th_dist - d`
    th_dist_d_ = th_dist_ - d_;   // ------------------------------? // 这里只考虑在拟合的平面上方的点 小于这个范围的点当做地面
//    std::cout<<"th_dist_d_=th_dist_ - d_ : "<<th_dist_d_<<std::endl;

    // return the equation parameters

}


// 获得待拟合的地面点
void GroundFit::extract_initial_seeds_(const pcl::PointCloud<VPoint>& p_sorted)
{
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
    // return seeds points  g_seeds_pc
}

// dynamic reconfigure 回调函数
void GroundFit::dynamic_callback(aiimooc_syz4::FitPlaneConfig &config, uint32_t level) {

            // 从.cfg文件中获取,传递给成员变量
            sensor_height_ = config.sensor_height;
            num_iter_ = config.num_iter;
            num_lpr_ = config.num_lpr;
            th_seeds_ = config.th_seeds;
            th_dist_ = config.th_dist;


            // 打印
            ROS_INFO("Reconfigure Request: %f %d %d %f %f", 
                    config.sensor_height, 
                    config.num_iter, 
                    config.num_lpr,
                    config.th_seeds,
                    config.th_dist
                    );
        }



#endif


