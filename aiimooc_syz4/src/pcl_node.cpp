
#include "pcl_core.h"





int main(int argc, char *argv[])
{
    ros::init(argc, argv, "lidar_pcl");  // 节点名称 launch中的 type="aiimooc_syz4_node"是可执行文件名称
    ros::NodeHandle nh;

    // 创建对象
    GroundFit core(nh);




    
    return 0;
}
