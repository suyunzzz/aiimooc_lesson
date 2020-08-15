'''
Author: your name
Date: 2020-08-14 11:41:57
LastEditTime: 2020-08-14 11:49:38
LastEditors: your name
Description: In User Settings Edit
FilePath: \flownet3d\vis_kitti_test.py
'''
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def vis_kitti(f1,f2,flow):

    print('---vis_kitti---')
    # f1+flow
    pt1=o3d.geometry.PointCloud()
    f1_flow=f1+flow
    pt1.points=o3d.utility.Vector3dVector(f1_flow.reshape(-1,3))
    pt1.paint_uniform_color([1,0,0])

    # f2
    pt2=o3d.geometry.PointCloud()
    pt2.points=o3d.utility.Vector3dVector(f2.reshape(-1,3))
    pt2.paint_uniform_color([0,1,0])

    # f1
    pt3=o3d.geometry.PointCloud()
    pt3.points=o3d.utility.Vector3dVector(f1.reshape(-1,3))
    # pt3.paint_uniform_color([0,0,0])

    # 根据flow(n*3)的距离赋值
    length=np.linalg.norm(flow,axis=1)
    length=np.asarray(length)
    max_length=length.max()
    colors = plt.get_cmap("tab20")(length / (max_length if max_length > 0 else 1))
    pt3.colors = o3d.utility.Vector3dVector(colors[:, :3])


    # pt1和pt3画对应点的线段


    o3d.visualization.draw_geometries([pt1,pt2,pt3],window_name='pred+f2+f1',width=800,height=600)

def vis_kitti_4(f1,f2,flow,f1_reg):
    print('---vis_kitti_4---')

    # f1+flow  对应点, blue
    pt1=o3d.geometry.PointCloud()
    f1_flow=f1+flow
    pt1.points=o3d.utility.Vector3dVector(f1_flow.reshape(-1,3))
    pt1.paint_uniform_color([0,0,1])

    # f2 target    ,green
    pt2=o3d.geometry.PointCloud()
    pt2.points=o3d.utility.Vector3dVector(f2.reshape(-1,3))
    pt2.paint_uniform_color([0,1,0])

    # f1 source     red 
    pt3=o3d.geometry.PointCloud()
    pt3.points=o3d.utility.Vector3dVector(f1.reshape(-1,3))
    pt3.paint_uniform_color([1,0,0])

    # # 根据flow(n*3)的距离赋值
    # length=np.linalg.norm(flow,axis=1)
    # length=np.asarray(length)
    # max_length=length.max()
    # colors = plt.get_cmap("tab20")(length / (max_length if max_length > 0 else 1))
    # pt3.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # f1变换后的点云，黄色
    pt4 = o3d.geometry.PointCloud()
    pt4.points = o3d.utility.Vector3dVector(f1_reg.reshape(-1, 3))
    pt4.paint_uniform_color([1, 1, 0])

    # pt1和pt3画对应点的线段


    o3d.visualization.draw_geometries([pt1,pt2,pt3,pt4],window_name='pred+f2+f1',width=800,height=600)


def reg(f1,f2,flow):
    '''
    实现两帧点云(f1,,f1+flow)的配准，输出r,t
    f1_reg：为source(f1)变换后的点云，
    f1：为source点云,f1_cloud
    f1+flow：为target点云
    f2：
    '''
    print('---reg---')
    
    print('source:f1,target:f1+flow')

    # target
    pred_cloud = o3d.geometry.PointCloud()
    f1_flow = f1 + flow
    pred_cloud.points = o3d.utility.Vector3dVector(f1_flow.reshape(-1, 3))

    # source
    f1_cloud=o3d.geometry.PointCloud()
    f1_cloud.points=o3d.utility.Vector3dVector(f1.reshape(-1,3))

    # 为两个点云上上不同的颜色
    f1_cloud.paint_uniform_color([1, 0.706, 0])  # source 为黄色
    pred_cloud.paint_uniform_color([0, 0.651, 0.929])  # target 为蓝色

    # # ---------配准前------------
    # # 创建一个 o3d.visualizer class
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # vis.add_geometry(f1_cloud)
    # vis.add_geometry(pred_cloud)
    #
    # # 让visualizer渲染点云
    # vis.update_geometry(f1_cloud)
    # vis.poll_events()
    # vis.update_renderer()
    #
    # vis.run()


    # ----------配准后-------------
    threshold = 1.0  # 移动范围的阀值
    trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，这是一个转换矩阵，
                             [0, 1, 0, 0],  # 象征着没有任何位移，没有任何旋转，我们输入
                             [0, 0, 1, 0],  # 这个矩阵为初始变换
                             [0, 0, 0, 1]])

    reg_p2p=o3d.registration.registration_icp(f1_cloud,pred_cloud,      # source ,target
                                              threshold,trans_init,
                                              o3d.registration.TransformationEstimationPointToPoint()
                                              )
    # 将我们的矩阵依照输出的变换矩阵进行变换
    print(reg_p2p)
    f1_cloud.transform(reg_p2p.transformation)
    # f1_reg.paint_uniform_color([1, 0, 0])  # target 变换后的点云为红色



    # 将两个点云放入visualizer
    vis.add_geometry(f1_cloud)
    vis.add_geometry(pred_cloud)
    # vis.add_geometry(f1_reg)

    # 让visualizer渲染点云
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

    vis.run()
    print('***')
    return f1_cloud, reg_p2p.transformation   # 返回变换后的点云+变换矩阵


def reg2(f1, f2, flow):
    '''
    实现两帧点云(f1,,f2)的配准，输出r,t
    f1_reg：为source(f1)变换后的点云，
    f1：第一帧点云，为source点云,f1_cloud
    f1+flow：预测后的点云
    f2：第二帧点云，target
    '''

    print('source:f1,target:f2')
    # target -- f2
    f2_cloud = o3d.geometry.PointCloud()
    f2_cloud.points = o3d.utility.Vector3dVector(f2.reshape(-1, 3))

    # source
    f1_cloud = o3d.geometry.PointCloud()
    f1_cloud.points = o3d.utility.Vector3dVector(f1.reshape(-1, 3))

    # 为两个点云上上不同的颜色
    f1_cloud.paint_uniform_color([1, 0.706, 0])  # source 为黄色
    f2_cloud.paint_uniform_color([0, 0.651, 0.929])  # target 为蓝色

    threshold = 1.0  # 移动范围的阀值
    trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，这是一个转换矩阵，
                             [0, 1, 0, 0],  # 象征着没有任何位移，没有任何旋转，我们输入
                             [0, 0, 1, 0],  # 这个矩阵为初始变换
                             [0, 0, 0, 1]])

    reg_p2p = o3d.registration.registration_icp(f1_cloud, f2_cloud,  # source ,target
                                                threshold, trans_init,
                                                o3d.registration.TransformationEstimationPointToPoint()
                                                )
    # 将我们的矩阵依照输出的变换矩阵进行变换
    print(reg_p2p)
    f1_cloud.transform(reg_p2p.transformation)
    # f1_reg.paint_uniform_color([1, 0, 0])  # target 变换后的点云为红色

    # 创建一个 o3d.visualizer class
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 将两个点云放入visualizer
    vis.add_geometry(f1_cloud)      # source
    vis.add_geometry(f2_cloud)      # target
    # vis.add_geometry(f1_reg)

    # 让visualizer渲染点云
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

    vis.run()

    return f1_cloud, reg_p2p.transformation  # f1_reg,返回变换矩阵

if __name__ == "__main__":
    f1=np.loadtxt('seu_pcd/f0.txt')
    f2=np.loadtxt('seu_pcd/f1.txt')
    flow=np.loadtxt('seu_pcd/pred_val.txt')
    f1_flow=f1+flow
    np.savetxt('seu_pcd/f0_f1_flow.txt',f1_flow)
    vis_kitti(f1,f2,flow)

    # 配准f1,f1+flow
    f1_reg,T1=reg(f1,f2,flow)
    np.savetxt('seu_pcd/flow_T0_1.txt',T1)

    # 使用f1,f2配准
    # f1_reg,T2=reg2(f1,f2,flow)

    # 保存配准后的点云
    f1_reg=np.asarray(f1_reg.points)
    np.savetxt('seu_pcd/pc1_reg.txt',f1_reg)
    # 可视化
    vis_kitti_4(f1,f2,flow,f1_reg)

