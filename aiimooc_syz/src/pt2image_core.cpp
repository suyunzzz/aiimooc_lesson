// Pcl2ImgCore类实现


#include "pt2image_core.h"

// 构造函数
Pcl2ImgCore::Pcl2ImgCore(ros::NodeHandle &nh):


    // 初始化成员
    _frame(pcl::RangeImage::LASER_FRAME),  // 坐标系

    _ang_res_x(0.5),  //水平角度分辨率
    _ang_res_y(0.7),
    _max_ang_w(360),  //水平角度范围
    _max_ang_h(360),

    _min_range(0.5),
    _max_range(50),
    camera_pose(Eigen::Affine3f::Identity()),  // 相机位姿，pcl生成depth_image函数会用到

    range_image(new pcl::RangeImageSpherical),
    current_pc_ptr(new pcl::PointCloud<pcl::PointXYZ>),
    msg(new sensor_msgs::Image)

{
    std::cout<<"初始化类Pcl2ImgCore"<<std::endl;

    // dynamic reconfigure   将这个回调函数放在构造函数内部
    dynamic_reconfigure::Server<aiimooc_syz::RangeImageConfig> server;
    dynamic_reconfigure::Server<aiimooc_syz::RangeImageConfig>::CallbackType callback;
 
    callback = boost::bind(&Pcl2ImgCore::dynamic_callback, this,_1,_2);   // 调用dynamic reconfigure的回调函数
    server.setCallback(callback);


    sub_point_cloud_=nh.subscribe("/rslidar_points",10,&Pcl2ImgCore::point_cb,this);   

    pub_Img_=nh.advertise<sensor_msgs::Image>("/range_image", 10);  //发布到/image话题
    ros::spin();
}

//析构函数
Pcl2ImgCore::~Pcl2ImgCore(){}

void Pcl2ImgCore::Spin(){}

// 回调函数
void Pcl2ImgCore::point_cb(const sensor_msgs::PointCloud2ConstPtr & in_cloud_ptr){

    std::cout<<"--------------start-------------------\nget Pointcloud"<<std::endl;

    pcl::fromROSMsg(*in_cloud_ptr, *current_pc_ptr);  //ros数据类型转为pcl中的数据类型，下面就使用current_pc_ptr了

    // rangeImageSpherial投影
    range_image->createFromPointCloud( *current_pc_ptr,
                                    pcl::deg2rad(_ang_res_x),pcl::deg2rad(_ang_res_y),
                                    pcl::deg2rad(_max_ang_w),pcl::deg2rad(_max_ang_h),
                                    camera_pose,          // 相机位姿参数，需要指定一个值
                                    pcl::RangeImage::LASER_FRAME,0.0, 0.0f, 0);

    std::cout<<*range_image<<std::endl;
    
    // 给range_image设置header
    range_image->header.seq = current_pc_ptr->header.seq;
    range_image->header.frame_id = current_pc_ptr->header.frame_id;
    range_image->header.stamp    = current_pc_ptr->header.stamp;


    int cols = range_image->width;
    int rows = range_image->height;

    // sensor_msgs::ImagePtr msg;     

    // 转换因子
    float factor = 1.0f / (_max_range - _min_range);
    float offset = -_min_range;
    // std::cout<<"factor:\t"<<factor<<std::endl;

    
    // cv::Mat _rangeImage; //最后的OpenCV格式的图像
    _rangeImage = cv::Mat::zeros(rows, cols, cv_bridge::getCvType(encoding));

    std::cout<<"cols: "<<cols<<","<<"rows: "<<rows<<std::endl;

    // 遍历每一个点 生成OpenCV格式的图像
    for (int i=0; i<cols; ++i)
    {
        for (int j=0; j<rows; ++j)
        {
            float r = range_image->getPoint(i, j).range;
            
            float range = (!std::isinf(r))?
            std::max(0.0f, std::min(1.0f, factor * (r + offset))) :
            0.0;

            _rangeImage.at<ushort>(j, i) = static_cast<ushort>((range) * std::numeric_limits<ushort>::max());
        }
    }
    // 转换为msg
    msg=cv_bridge::CvImage(std_msgs::Header(),encoding,_rangeImage).toImageMsg();    // 这里直接使用range_image的header为什么不行？？？
    pcl_conversions::fromPCL(range_image->header, msg->header);   // header的转变

    std::cout<<"in_cloud_ptr->header\n"<<in_cloud_ptr->header<<std::endl;
    std::cout<<"current_pc_ptr->header\n"<<current_pc_ptr->header<<std::endl;
    std::cout<<"range_image->header\n"<<range_image->header<<std::endl;
    std::cout<<"msg->header\n"<<msg->header<<std::endl;

    pub_Img_.publish(msg);

    std::cout<<"---------------------end----------------------"<<std::endl;
}

// dynamic reconfigure 回调函数
void Pcl2ImgCore::dynamic_callback(aiimooc_syz::RangeImageConfig &config, uint32_t level) {

            // 从.cfg文件中获取,传递给成员变量
            _ang_res_x = config.ang_res_x;
            _ang_res_y = config.ang_res_y;
            _max_ang_w = config.max_ang_w;
            _max_ang_h = config.max_ang_h;

            // 打印
            ROS_INFO("Reconfigure Request: %f %f %f %f ", 
                    config.ang_res_x, config.ang_res_y, 
                    config.max_ang_w,
                    config.max_ang_h 
                    );
        }

