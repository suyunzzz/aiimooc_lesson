import numpy as np 
import open3d as o3d 

def vis_cloud(a,b,c1,c2):
	# a:n*3的矩阵
	# b:n*3的矩阵
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(a.reshape(-1,3))
    # pt1.paint_uniform_color([1,0,0])
    pt1.colors=o3d.utility.Vector3dVector(c1[:,:3])

    pt2=o3d.geometry.PointCloud()
    pt2.points=o3d.utility.Vector3dVector(b.reshape(-1,3))
    # pt2.paint_uniform_color([0,1,0])
    pt2.colors=o3d.utility.Vector3dVector(c2[:,:3])


    o3d.visualization.draw_geometries([pt1,pt2],window_name='pc1 and pc2',width=800,height=600)


def vis_cloud1(a):
	# a:n*3的矩阵
	# b:n*3的矩阵
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(a.reshape(-1,3))
    pt1.paint_uniform_color([1,0,0])
    # pt1.colors=o3d.utility.Vector3dVector(c1[:,:3])

def vis_cloud1andcolor(a,color):
	# a:n*3的矩阵
	# b:n*3的矩阵
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(a.reshape(-1,3))
    # pt1.paint_uniform_color([1,0,0])
    pt1.colors=o3d.utility.Vector3dVector(color[:,:3])  


    o3d.visualization.draw_geometries([pt1],window_name='pc1',width=800,height=600)