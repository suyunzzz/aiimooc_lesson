//
// Created by s on 2020/5/20.
//

#ifndef SEGMENT_SEU_SEG_H
#define SEGMENT_SEU_SEG_H


#include <iostream>
#include <pcl/point_types.h>
//#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>//体素格滤波器VoxelGrid
#include <pcl/io/pcd_io.h>//点云文件pcd 读写
#include <pcl/visualization/cloud_viewer.h>//点云可视化

typename pcl::PointCloud<pcl::PointXYZI> CloudI;
typename pcl::PointCloud<pcl::PointXYZ> Cloud;

class SeuSeg
{
public:
    SeuSeg():

    {}

    ~SeuSeg(){}

private:
    CloudI cloud;



};

#endif //SEGMENT_SEU_SEG_H
