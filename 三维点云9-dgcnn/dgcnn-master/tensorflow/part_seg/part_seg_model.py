import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../'))
import tf_util          # edge-conv等操作
from transform_nets import input_transform_net  # t-net

def get_model(point_cloud, input_label, is_training, cat_num, part_num, \
    batch_size, num_point, weight_decay, bn_decay=None):

  batch_size = point_cloud.get_shape()[0].value  # B
  num_point = point_cloud.get_shape()[1].value      # N
  input_image = tf.expand_dims(point_cloud, -1)   # B*N*3*1

  k = 20

  adj = tf_util.pairwise_distance(point_cloud)   # B*N*N
  nn_idx = tf_util.knn(adj, k=k)                  # B*N*K
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)          # B*N*K*6

  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, bn_decay, K=3, is_dist=True)         # B*3*3
  point_cloud_transformed = tf.matmul(point_cloud, transform)       # B*N*3
  
  input_image = tf.expand_dims(point_cloud_transformed, -1)  # B*N*3*1
  adj = tf_util.pairwise_distance(point_cloud_transformed)      # B*N*N
  nn_idx = tf_util.knn(adj, k=k)                                 # B*N*K
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)          # B*N*K*6

  out1 = tf_util.conv2d(edge_feature, 64, [1,1],                            # B*N*K*64
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv1', bn_decay=bn_decay, is_dist=True)
  
  out2 = tf_util.conv2d(out1, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

  net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)                  # B*N*1*64



  adj = tf_util.pairwise_distance(net_1)                # B*N*N
  nn_idx = tf_util.knn(adj, k=k)                        # B*N*20
  edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)               # B*N*K*128

  out3 = tf_util.conv2d(edge_feature, 64, [1,1],                # B*N*K*64
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

  out4 = tf_util.conv2d(out3, 64, [1,1],                    # B*N*K*64
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv4', bn_decay=bn_decay, is_dist=True)
  
  net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)          # B*N*1*64
  
  

  adj = tf_util.pairwise_distance(net_2)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

  out5 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv5', bn_decay=bn_decay, is_dist=True)

  # out6 = tf_util.conv2d(out5, 64, [1,1],
  #                      padding='VALID', stride=[1,1],
  #                      bn=True, is_training=is_training, weight_decay=weight_decay,
  #                      scope='adj_conv6', bn_decay=bn_decay, is_dist=True)

  net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)          # B*N*1*64



  out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1],        # B*N*1*1024
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

  out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')  # B*1*1*1024


  one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])           # B*1*1*cat_num
  one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 64, [1, 1],   # B*1*1*64
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='one_hot_label_expand', bn_decay=bn_decay, is_dist=True)
  out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])       # B*1*1*1088
  expand = tf.tile(out_max, [1, num_point, 1, 1])           # B*N*1*1088

  concat = tf.concat(axis=3, values=[expand,        # B*N*1*1088
                                     net_1,         # B*N*1*64
                                     net_2,         # B*N*1*64
                                     net_3])        # B*N*1*64

  net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,       # B*N*1*256
            bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
  net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,         # B*N*1*256
            bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
  net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,         # B*N*1*128
            bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.conv2d(net2, part_num, [1,1], padding='VALID', stride=[1,1], activation_fn=None,   # B*N*1*part_num = 4*2048*1*50
            bn=False, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)

  net2 = tf.reshape(net2, [batch_size, num_point, part_num])            # B*N*part_num = 4*2048*50

  return net2


def get_loss(seg_pred, seg):
  per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
  seg_loss = tf.reduce_mean(per_instance_seg_loss)
  per_instance_seg_pred_res = tf.argmax(seg_pred, 2)
  
  return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res

