'''
@Author: Su Yunzheng
@Date: 2020-06-22 12:15:11
@LastEditTime: 2020-07-04 12:32:24
@LastEditors: Please set LastEditors
@Description: 开源代码GPFPython实现
@FilePath: \GPF_修改\ground_plane_filter.py
'''

import numpy as np
import open3d
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR:{}".format(BASE_DIR))

# data文件夹
DATA_DIR=os.path.join(BASE_DIR,'data')
print("DATA_DIR:{}".format(DATA_DIR))

# 输入文件
pcd_file="f0.pcd"   # 输入文件
data_file=os.path.join(DATA_DIR,pcd_file)
print("data_file:{}".format(data_file))
result_dir,ext=pcd_file.split(".")

# 结果文件，存放地面和非地面
result_dir=os.path.join(DATA_DIR,result_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
print("result_dir:{}".format(result_dir))

# 日志
LOG_DIR=os.path.join(result_dir, 'LOG')
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')    # 保存日志

# 写进日志+在终端打印
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()  # 立即刷新
    print(out_str)

def visPointCloudByOpen3d(Pt_ground,no_ground):
    """
    可视化点云
    :param Pt: n*3的矩阵
    :return:两个点云
    """
    # points = np.random.rand(10000, 3)
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points =open3d.utility.Vector3dVector(Pt_ground[:,0:3].reshape(-1,3))
    point_cloud.paint_uniform_color([0.5, 0.5, 0.5])

    point_cloud_no_ground = open3d.geometry.PointCloud()
    point_cloud_no_ground.points =open3d.utility.Vector3dVector(no_ground[:,0:3].reshape(-1,3))
    point_cloud_no_ground.paint_uniform_color([1, 0, 0])


    open3d.visualization.draw_geometries([point_cloud,point_cloud_no_ground],"GPF",width=800,height=600)
    log_string("-----------visualization done--------------")

    return point_cloud,point_cloud_no_ground


# 全局变量
g_seeds_pc=[]   # 提取的种子点
g_seeds_pc_array=np.array(g_seeds_pc)

g_ground_pc=[]
g_ground_pc_array=np.array(g_ground_pc)

g_not_ground_pc=[]
g_not_ground_pc_array=np.array(g_not_ground_pc)

cluster_all=[]

# 创建GPF类
class GPF:


    __d=0              # ax+by+cz+__d_=0
    __normal=np.zeros((3,1),float)             # 法向量
    __th_dist_d=0                  # th_dist_d_ = th_dist_ - d  # 距离阈值和d作差

    # 地面点 非地面点
    point_cloud_ground = open3d.geometry.PointCloud()
    point_cloud_ground.paint_uniform_color([0.5, 0.5, 0.5])

    point_cloud_no_ground = open3d.geometry.PointCloud()
    point_cloud_no_ground.paint_uniform_color([1, 0, 0.5])

    def __init__(self,num_iter=2,num_lpr=20,  th_seeds=0.2, th_dist=0.3):
        self.__num_iter = num_iter
        self.__num_lpr = num_lpr
        self.__th_seeds = th_seeds
        self.__th_dist = th_dist
        # print("---------Params--------------")
        # print("num_iter:{}\nnum_lpr:{}\nth_seeds:{}\nth_dist:{}".  \
        #     format(self.__num_iter, self.__num_lpr ,self.__th_seeds,self.__th_dist))
        log_string("---------Params--------------")
        log_string("num_iter:{}\nnum_lpr:{}\nth_seeds:{}\nth_dist:{}". \
                   format(self.__num_iter, self.__num_lpr ,self.__th_seeds,self.__th_dist))
    
    # 打印平面的A B C D以及推导出的距离阈值
    def GetParams(self):
        print("--------------------------")
        print ("d:{}\nnormal:{}\nth_dist_d:{}". \
        format(self.__d,self.__normal,self.__th_dist_d))

    # 提取seeds  私有方法
    def __extract_initial_seeds(self,cloud_sorted):
        """
        input:cloud_sorted，排序好的点云，根据z值从小到大  n*3
        output:g_seeds_pc，种子点
        """

        global g_seeds_pc_array
        global g_seeds_pc

        num = 0         # 最低点的总和
        count = 0       # 最低点的数量
        
        # 计算前__num_lpr个点的平均值
        cloud_lpr=cloud_sorted[0:self.__num_lpr+1,:]   # 最低的20个点，20*3的矩阵
        lpr_height=cloud_lpr.mean(axis=0)[-1]       # 最低点的平均高度

        # 种子点清空
        g_seeds_pc.clear()

        # 计算新的这种子点集合
        # TODO : 使用矩阵选取这些点，而不是使用循环，因为是按照顺序排列的，其实速度应该不会有太大差异
        for i in range(len(cloud_sorted)):
            if (cloud_sorted[i,-1]<lpr_height+self.__th_seeds):
                g_seeds_pc.append(cloud_sorted[i,:])
            else:
                break
        
        g_seeds_pc_array=np.array(g_seeds_pc)
        # print("g_seeds_pc:{}".format(g_seeds_pc))

        




    # 平面估计 私有方法，只能在类的内部调用
    def __estimate_plane(self):
        """
        input：g_ground_pc_array,n*3的点云，待估计的地面
        output：normals,d,th_dist_d_
        """
        global g_ground_pc_array
        global g_ground_pc
        

        # 1.计算协方差矩阵，平均值
        # g_ground_pc_array=np.array(g_ground_pc)
        g_ground_pc_T=g_ground_pc_array.T
        # print("g_ground_pc_T:{}".format(g_ground_pc_T))
        covMat=np.cov(g_ground_pc_T)   # 3*3的协方差矩阵
        pc_mean=g_ground_pc_array.mean(axis=0)  # 平均值 1*3
        # print("pc_mean:{}".format(pc_mean))

        # 2.对协方差矩阵求取特征值、特征向量,得到法向量
        eigVals,eigVects=np.linalg.eig(np.mat(covMat))
        eigValIndice=np.argsort(eigVals)          #对特征值从小到大排序
        self.__normal=eigVects[:,eigValIndice[0]]      # 法向量 3*1
        self.__normal=self.__normal/np.linalg.norm(self.__normal)  # 归一化
        # print("self.__normal shape:{}".format((self.__normal).shape))
        log_string("self.__normal shape:{}".format((self.__normal).shape))

        # print("normals:{}".format(np.transpose(self.__normal)))
        log_string("normals:{}".format(np.transpose(self.__normal)))

        # 3.计算ax+by+cz+d=0中的d
        self.__d=-(np.matmul(pc_mean,self.__normal))
        # print("d:{}".format(self.__d))

        # 4.set distance threhold to `th_dist - d`
        self.__th_dist_d=self.__th_dist-self.__d
        # print("th_dist_d:{}".format(self.__th_dist_d))


    
    # 回调函数，公有方法，程序入口
    def callback(self,cloud_in):
        """
        cloud_in:n*3的矩阵
        """
        global g_ground_pc_array
        global g_ground_pc
        
        global g_not_ground_pc
        global g_not_ground_pc_array
        

        # 1.点云复制
        cloud_org=cloud_in
        # print("PointCloud copy done")
        log_string("PointCloud copy done")

        # 2.按照z排序,小到大
        cloud_sorted=cloud_in[cloud_in[:,-1].argsort()]  # argsort 默认为小到大

        # 3.Extract init ground seeds.
        self.__extract_initial_seeds(cloud_sorted)
        g_ground_pc_array=g_seeds_pc_array     
        # print("g_seeds_pc_array:{}".format(g_seeds_pc_array))

        # print("g_ground_pc_array:{}".format(g_ground_pc_array))


        # 4.Ground plane fitter mainloop
        for i in range(self.__num_iter):
            # print(" -----------iter [{}]------------".format(i+1))
            log_string(" -----------iter [{}]------------".format(i+1))
            self.__estimate_plane()   # 估计平面
            g_ground_pc.clear()
            g_not_ground_pc.clear()

            # 计算点云中每个点到平面的距离n*3*3*1=n*1
            result_d=np.matmul(cloud_org,self.__normal)
            # print("result_d:{}".format(result_d))

            # 根据result_d选取ground
            for j in range(len(result_d)):
                if(result_d[j]<self.__th_dist_d):
                    g_ground_pc.append(cloud_org[j,:])
                else:
                    g_not_ground_pc.append(cloud_org[j,:])

            g_ground_pc_array=np.array(g_ground_pc)
            g_not_ground_pc_array=np.array(g_not_ground_pc)

            # 输出迭代结果
            # print("Iter [{}], ground:{}, no_ground:{}".format(i+1,len(g_ground_pc),len(g_not_ground_pc)))
            log_string("Iter [{}], ground:{}, no_ground:{}".format(i+1,len(g_ground_pc),len(g_not_ground_pc)))

            # 可视化
            # _=visualize(g_ground_pc)
            point_cloud_ground,point_cloud_no_ground=visPointCloudByOpen3d(g_ground_pc_array,g_not_ground_pc_array)

            # save
            ground_name=str("ground"+str(i+1)+".pcd")
            no_ground_name=str("no_ground"+str(i+1)+".pcd")

            ground_file=os.path.join(result_dir,ground_name)
            no_ground_file=os.path.join(result_dir,no_ground_name)
            open3d.io.write_point_cloud(ground_file,point_cloud_ground)
            open3d.io.write_point_cloud(no_ground_file,point_cloud_no_ground)
            log_string(ground_file+" saved")
            log_string(no_ground_file+" saved")






if __name__ == "__main__":
    # print(open3d.__version__)
    gpf=GPF(num_iter=3,num_lpr= 20,th_seeds=0.7,th_dist=0.45)


    # 数据读取
    # txtq
    # cloud_in=np.loadtxt("city.txt")

    # pcd
    log_string("Testing IO for point cloud ...")
    pcd = open3d.io.read_point_cloud(data_file)
    log_string("read data:{}".format(data_file))
    cloud_in=np.asarray(pcd.points)

    log_string("cloud_in shape:{}".format(cloud_in.shape))

    gpf.callback(cloud_in)

    # save

    # print("P:{}".format(P))
