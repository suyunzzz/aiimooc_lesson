#include "pt2RangeImage_core.h"

int main(int argc, char *argv[])
{
    ros::init(argc,argv,"RangeImageCreater");

    ros::NodeHandle nh;

    pt2RI core(nh);


    return 0;
}
