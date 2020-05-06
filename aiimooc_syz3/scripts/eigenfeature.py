#!/usr/bin/env python3
import sys

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

# my msg
#from eigen.msg import eigen_features

import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import os

# 对于ros中和conda环境中都有的package，在导入前先将ros中的路径删掉，然后再导入
# sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages/")
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
# import cv2
from open3d.open3d.geometry import PointCloud
from open3d.open3d.utility import Vector3dVector


class eigen_feature:
	def __init__(self):
		# self.feature_pub = rospy.Publisher('/features',eigen_features,queue_size=1)
		self.pt_sub=rospy.Subscriber("/rslidar_points", PointCloud2, self.callback)
	
	
	def callback(self,lidar):
	
		# convert pointcloud2 to array
		lidar = pc2.read_points(lidar)
		points = np.array(list(lidar))
		
		# get a PointCloud
		point_cloud = PointCloud()
		point_cloud.points = Vector3dVector(points[:,0:3].reshape(-1,3))
    	
		# downsample
		point_cloud= point_cloud.voxel_down_sample(voxel_size=0.6)
		points=point_cloud.points
		points=np.asarray(points)
    	
		# build a kdtree
		pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    	
		eigenfeatures = []
		K=19
		
		save_filename='/home/s/Dataset/syz.txt'
		
		for i in range(points.shape[0]):
			[k,index,_]=pcd_tree.search_knn_vector_3d(point_cloud.points[i],K+1)
			
			eigenvalues,eigenvectors=self.PCA(points[index[:],:])
			
			tmp=self.get_6features(eigenvalues=eigenvalues)
			
			eigenfeatures.append(tmp)
			
		# print("eigenfeatures size:{}".format(len(eigenfeatures)))
		
		# convert list[] to array
		eigenfeatures=np.array(eigenfeatures)
		print('eigenfeatures shape:{}'.format(eigenfeatures.shape))
			
		np.savetxt(save_filename,eigenfeatures)
		print(save_filename+" save!")

		# 退出
		try:
			os._exit(0)
		except:
			print('Program is dead')
			
		
	
	def get_6features(self,eigenvalues):
		
		v1=np.float(eigenvalues[0]) 
		v2=np.float(eigenvalues[1])
		v3=np.float(eigenvalues[2])
		
		sum=v1+v2+v3
		if sum==0:
			sum=0.000000001
		e1=v1/sum
		e2=v2/sum
		e3=v3/sum

		# print('v1:{},v2:{},v3:{}'.format(v1,v2,v3))
		if v1==0:
			v1=0.000000001
		L=(v1-v2)/v1   # 线性
		P=(v2-v3)/v1   # 平面性
		S=v3/v1        # 散度性
		O=3*np.power(e1*e2*e3,1/3)   # 全方差

		if e1<=0:
			e1=0.000000001
		if e2<=0:
			e2=0.000000001
		if e3<=0:
			e3=0.000000001

		E=e1*np.log(e1)+e2*np.log(e2)+e3*np.log(e3)  # 特征熵
		E=E*(-1)
		C=3*v3/sum       # 曲率变化

		return [L,P,S,O,E,C]

		
	
	def PCA(self,k_neighbor,sort=True):
		k_neighbor_T=k_neighbor.T
		C=np.matmul(k_neighbor_T,k_neighbor) # 3x3
		eigenvalues,eigenvectors=np.linalg.eig(C)

		if sort:
			sort=eigenvalues.argsort()[::-1]
			eigenvalues = eigenvalues[sort]
			eigenvectors = eigenvectors[:, sort]
		return eigenvalues, eigenvectors
		
	
if __name__ == '__main__':
	print('------------------start-----------------')
	print("open3d:{}".format(o3d.__version__))
	eigen=eigen_feature()
	rospy.init_node('eigenfeature')
	rospy.spin()
		
