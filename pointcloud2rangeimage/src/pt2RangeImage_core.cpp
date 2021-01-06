

#include "pt2RangeImage_core.h"

// 构造函数
pt2RI::pt2RI(ros::NodeHandle& nh)
:_ang_res_x(0.5),_ang_res_y(0.7),_max_ang_w(360),_max_ang_h(360),_min_range(0.5),_max_range(50),
_pCloud(new PointCloudType()),_pRI(new pcl::RangeImage())
{   

    printf("constructor\n");
    // 初始化指针
    _suber = nh.subscribe("/rslidar_points",10,&pt2RI::point_cb,this);
    _publisher = nh.advertise<sensor_msgs::Image>("/range_image",1000);


    
    ros::spin();



    
}


/* // 构造函数
pt2RI::pt2RI(ros::NodeHandle& nh,float res_x,float res_y,float w,float h,float max_range,float min_range)
:_ang_res_x(res_x),_ang_res_y(res_y),_max_ang_w(w),_max_ang_h(h),_min_range(min_range),_max_range(max_range),
_pCloud(new PointCloudType()),_pRI(new pcl::RangeImageSpherical)
{
    // 初始化指针
    _suber = nh.subscribe("/rslidar_points",10,&pt2RI::point_cb,this);



    ros::spin();

} */

pt2RI::~pt2RI()
{
    printf("---desconstructor---\n");
    // delete [] 
}

// 回调函数
void
pt2RI::point_cb(const sensor_msgs::PointCloud2ConstPtr& in_cloud_ptr)
{
    std::cout<<"--------------start-------------------\nget Pointcloud"<<std::endl;

    pcl::fromROSMsg(*in_cloud_ptr,*_pCloud);

    _pRI->createFromPointCloud(*_pCloud,
                                   pcl::deg2rad(_ang_res_x),pcl::deg2rad(_ang_res_y),
                                    pcl::deg2rad(_max_ang_w),pcl::deg2rad(_max_ang_h),
                                    _camera_pose,          // 相机位姿参数，需要指定一个值
                                    pcl::RangeImage::LASER_FRAME,0.0, 0.0f, 0);

    std::cout<<*_pRI<<std::endl;

    if(_pRI->size()>0)
    {
        _range_image_viser.showRangeImage(*_pRI);

        printf("---showRangeImage--\n");

        // 转为img格式
        p_rangeimage = _pRI->getRangesArray();       // 获得rangeimage的深度值

        
        unsigned char* rgb_image = pcl::visualization::FloatImageUtils::getVisualImage(p_rangeimage,_pRI->width,_pRI->height);
        printf("-----\n");

        sensor_msgs::Image image_msg;
        image_msg.header = in_cloud_ptr->header;
        image_msg.height = _pRI->height;
        image_msg.width = _pRI->width;
        image_msg.step = _pRI->width*3;
        image_msg.encoding = "rgb8";            // 8 bit
        for(int i = 0;i<image_msg.width*image_msg.height*3;i++)             // 注意数组访问不能越界
        {
            image_msg.data.push_back(*(rgb_image+i));
        }

        _publisher.publish(image_msg);      

        // delete [] p_rangeimage;
        delete [] rgb_image;

    
    }


}


