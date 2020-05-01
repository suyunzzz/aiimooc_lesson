//入口函数

#include "pt2image_core.h"


int main(int argc, char **argv)
{
    ros::init(argc, argv, "range_image");  // 节点名称

    ros::NodeHandle nh;


    

    
    // if (nh.getParam("/pt_to_image/res_x",_ang_res_x)){
    //     ROS_INFO("proportional gain set to %f",_ang_res_x);
    // }
    // else
    // {
    // ROS_WARN("could not find parameter value /pt_to_image/res_x on parameter server");
    // }

    // if (nh.getParam("/pt_to_image/res_y",_ang_res_y)){
    //     ROS_INFO("proportional gain set to %f",_ang_res_y);
    // }
    // else
    // {
    // ROS_WARN("could not find parameter value /pt_to_image/res_y on parameter server");
    // }

    // if (nh.getParam("/pt_to_image/width_range",_max_ang_w)){
    //     ROS_INFO("proportional gain set to %f",_max_ang_w);
    // }
    // else
    // {
    // ROS_WARN("could not find parameter value /pt_to_image/width_range on parameter server");
    // }

    // if (nh.getParam("/pt_to_image/height_range",_max_ang_h)){
    //     ROS_INFO("proportional gain set to %f",_max_ang_h);
    // }
    // else
    // {
    // ROS_WARN("could not find parameter value /pt_to_image/height_range on parameter server");
    // }

    // 先将dynamic reconfigure的参数设置到临时变量上

    
    // 创建对象
    Pcl2ImgCore core(nh);



    return 0;
}
