#include <ros/ros.h>

//sensor_msgs
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>

//PCL
#include <pcl/range_image/range_image.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

//Dynamic reconfigure
// #include <dynamic_reconfigure/server.h>

//参数cfg生成的头文件
// #include "aiimooc_ccf/Dynamic_DaramsConfig.h"

ros::Publisher DepthImg_pub;//

float w_range=360.0;//视野范围
float h_range=180.0;
float r_x=1.0;//分辨率
float r_y=1.0;
float a_x=0.0;//相机位姿
float a_y=0.0;
float a_z=0.0;
float p_x=0.0;
float p_y=0.0;
float p_z=0.0;

ros::NodeHandle *n_ptr;
/* void dynamic_callback(aiimooc_ccf::Dynamic_DaramsConfig &config, uint32_t level)
{
	//动态获取参数
	n_ptr->setParam("/aiimooc_ccf/w_range", config.w_range);
	n_ptr->setParam("/aiimooc_ccf/h_range", config.h_range);
	n_ptr->setParam("/aiimooc_ccf/r_x", config.r_x);
	n_ptr->setParam("/aiimooc_ccf/r_y", config.r_y);
	
	n_ptr->setParam("/aiimooc_ccf/a_x", config.a_x);
	n_ptr->setParam("/aiimooc_ccf/a_y", config.a_y);
	n_ptr->setParam("/aiimooc_ccf/a_z", config.a_z);
	n_ptr->setParam("/aiimooc_ccf/p_x", config.p_x);
	n_ptr->setParam("/aiimooc_ccf/p_y", config.p_y);
	n_ptr->setParam("/aiimooc_ccf/p_z", config.p_z);
} */
pcl::visualization::RangeImageVisualizer range_image_widget("Range image");
//回调函数
void receive_pointsCallback(const sensor_msgs::PointCloud2ConstPtr& points)
{
	pcl::PointCloud<pcl::PointXYZ> points_temp;
	pcl::fromROSMsg(*points, points_temp);//sensor_msgs::PointCloud2转pcl::PointCloud<pcl::PointXYZ>

	Eigen::AngleAxisf rotation_z ( M_PI*a_z/180.0, Eigen::Vector3f ( 0,0,1 ) );
	Eigen::AngleAxisf rotation_x ( M_PI*a_x/180.0, Eigen::Vector3f ( 1,0,0 ) );
	Eigen::AngleAxisf rotation_y ( M_PI*a_y/180.0, Eigen::Vector3f ( 0,1,0 ) );
	Eigen::Translation3f translation(p_x,p_y,p_z);
	Eigen::Affine3f sensorPose = Eigen::Affine3f::Identity();
	sensorPose = translation*rotation_z.toRotationMatrix()*rotation_x.toRotationMatrix()*rotation_y.toRotationMatrix();
	
	pcl::RangeImage rangeImage;
	rangeImage.createFromPointCloud(points_temp, //点云数据
									pcl::deg2rad(r_x),//x方向精度
									pcl::deg2rad(r_y), //y方向精度
									pcl::deg2rad(w_range), //宽度视角
									pcl::deg2rad(h_range), //高度视角
									sensorPose, //模拟相机的位姿
									pcl::RangeImage::LASER_FRAME, 0.0, 0.0, 0.0);

		if(rangeImage.size() > 0)
		{
			range_image_widget.showRangeImage(rangeImage);//pcl深度可视化
			//获取数据转sensor_msgs::Image，并发布
			float *ranges = rangeImage.getRangesArray();
			unsigned char* rgb_image = pcl::visualization::FloatImageUtils::getVisualImage(ranges,rangeImage.width,rangeImage.height);
			
			sensor_msgs::Image image;
			image.header=points->header;
			image.height = rangeImage.height;
			image.width = rangeImage.width;
			image.step = rangeImage.width*3;   // step
			image.encoding = "rgb8";
			for(int i=0;i<rangeImage.width*rangeImage.height*3;i++)     // 遍历每一个像素
				image.data.push_back(rgb_image[i]);
			
			DepthImg_pub.publish(image);//发布
			
			delete []ranges;
			delete []rgb_image;
		}
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "aiimooc_ccf");
    ros::NodeHandle n;
	n_ptr = &n;

    //发布的话题
    DepthImg_pub = n.advertise<sensor_msgs::Image>("/ccf_pc/_img",1000);

	//订阅话题
	ros::Subscriber receive_points_sub = n.subscribe("/rslidar_points",100,receive_pointsCallback);

/* 	//dynamic_reconfigure
	dynamic_reconfigure::Server<aiimooc_ccf::Dynamic_DaramsConfig> server;
	dynamic_reconfigure::Server<aiimooc_ccf::Dynamic_DaramsConfig>::CallbackType f;
	f = boost::bind(&dynamic_callback, _1, _2);
	server.setCallback(f);
 */
	ros::Rate loop_rate(5);	
	while(ros::ok())
	{	
/* 		//获取参数的值
		n.getParam("/aiimooc_ccf/w_range", w_range);
		n.getParam("/aiimooc_ccf/h_range", h_range);
		n.getParam("/aiimooc_ccf/r_x", r_x);
		n.getParam("/aiimooc_ccf/r_y", r_y);
		
		n.getParam("/aiimooc_ccf/a_x", a_x);
		n.getParam("/aiimooc_ccf/a_y", a_y);
		n.getParam("/aiimooc_ccf/a_z", a_z);
		n.getParam("/aiimooc_ccf/p_x", p_x);
		n.getParam("/aiimooc_ccf/p_y", p_y);
		n.getParam("/aiimooc_ccf/p_z", p_z); */
		
		//打印参数
		ROS_INFO("-----------------------");
		ROS_INFO("w_range: %f", w_range);
		ROS_INFO("h_range: %f", h_range);
		ROS_INFO("r_x: %f", r_x);
		ROS_INFO("r_y: %f", r_y);
		
		ROS_INFO("a_x: %f", a_x);
		ROS_INFO("a_y: %f", a_y);
		ROS_INFO("a_z: %f", a_z);
		ROS_INFO("p_x: %f", p_x);
		ROS_INFO("p_y: %f", p_y);
		ROS_INFO("p_z: %f", p_z);

		ros::spinOnce();
		loop_rate.sleep();
	}

    return 0;
}