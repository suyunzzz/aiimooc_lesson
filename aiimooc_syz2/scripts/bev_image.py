#!/usr/bin/env python3
# use python3 in current conda env!!!

# import cv2 in conda rather than in ros
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


# import tf
# print(tf.__file__)
import cv_bridge
print(cv_bridge.__file__)
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2,Image
import sensor_msgs.point_cloud2 as pc2
import open3d
import os


class pt2brid_eye:
	def __init__(self):
		self.image_pub = rospy.Publisher('/bird_eyes', Image, queue_size=10)
		self.pt_sub=rospy.Subscriber("/rslidar_points", PointCloud2, self.callback)
		self.bridge = CvBridge()

	def callback(self,lidar):
		lidar = pc2.read_points(lidar)
		points = np.array(list(lidar))
		im = self.point_cloud_2_birdseye(points)  # im is a numpy array
    
  

    	
		self.image_pub.publish(self.bridge.cv2_to_imgmsg(im))
		print("convert!")
         


	def point_cloud_2_birdseye(self,points):
		x_points = points[:, 0]
		y_points = points[:, 1]
		z_points = points[:, 2]

		f_filt = np.logical_and((x_points > -50), (x_points < 50))
		s_filt = np.logical_and((y_points > -50), (y_points < 50))
		filter = np.logical_and(f_filt, s_filt)
		indices = np.argwhere(filter)

		x_points = x_points[indices]
		y_points = y_points[indices]
		z_points = z_points[indices]

		x_img = (-y_points*10).astype(np.int32)+500
		y_img = (-x_points *10).astype(np.int32)+500      

		pixel_values = np.clip(z_points,-2,2)
		pixel_values = ((pixel_values +2) / 4.0) * 255
		im=np.zeros([1001,1001],dtype=np.uint8)
		im[y_img, x_img] = pixel_values
		return im


# def cloud_subscribe():
#    rospy.init_node('cloud_subscribe_node')
#    pub = rospy.Publisher('/bird_eyes', Image, queue_size=10)
#    rospy.Subscriber("/rslidar_points", PointCloud2, callback)
    
#    rospy.spin()



if __name__ == '__main__':
	print("opencv: {}".format(cv2.__version__))
	print("open3d:{}".format(open3d.__version__))
	# cloud_subscribe()
	pt2img=pt2brid_eye()
	rospy.init_node('bev_image')
	rospy.spin()

