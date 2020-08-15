import open3d as o3d 
import numpy as np

def vis_cloud(a,b):
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(a.reshape(-1,3))
    pt1.paint_uniform_color([1,0,0])

    pt2=o3d.geometry.PointCloud()
    pt2.points=o3d.utility.Vector3dVector(b.reshape(-1,3))
    pt2.paint_uniform_color([0,1,0])

    o3d.visualization.draw_geometries([pt1,pt2],window_name='cloud[0] and corr',width=800,height=600)

if __name__ == "__main__":
    a=np.loadtxt("raw.txt")
    b=np.loadtxt("src_corr.txt")
    vis_cloud(a,b)