"""
描述：预测test0中的前10个点云，并可视化

运行：
(tf) F:\睿慕课\8\suyunzzz_8\pointnet>python pred_myData.py --model_path log/model_20.ckpt --visu --batch_size 10

"""

import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
# import scipy.misc  # 被抛弃了
import imageio   # 替换scipy.misc.imsave
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
from utils import pc_util

import open3d


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='dgcnn', help='Model name: dgcnn [default: dgcnn]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 4]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='My_dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()    # 命令行参数解析


BATCH_SIZE = FLAGS.batch_size  # 每一个batch中有几个点云
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir               ## dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_inference.txt'), 'w')    # 创建日志
LOG_FOUT.write(str(FLAGS)+'\n')

fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')      # 创建文件记录预测的label


NUM_CLASSES = 40
# Python rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
SHAPE_NAMES = [line.rstrip() for line in \
               open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]

print("SHAPE_NAMES:{}".format(SHAPE_NAMES))

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt')) # 得到list[]
TEST_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))     # list[]

# 日志输出
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()  # 立刻刷新
    print(out_str)

def evaluate(num_votes):   #  控制对每一个batch进行几次评价，进行几次旋转，数据的数目
    is_training = False

    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)  # 占位
        is_training_pl = tf.placeholder(tf.bool, shape=())  #  占位

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)   # 得到预测
        loss = MODEL.get_loss(pred, labels_pl, end_points)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    # 载入变量
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    # eval_one_epoch(sess, ops, num_votes)
    eval_my_data(sess,ops,num_votes)   # 预测test0中的0-9的点云

def visPointCloudByOpen3d(Pt):
    """
    可视化点云
    :param Pt: n*3的矩阵
    :return:
    """
    # points = np.random.rand(10000, 3)
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points =open3d.utility.Vector3dVector(Pt[:,0:3].reshape(-1,3))
    open3d.visualization.draw_geometries([point_cloud],width=400,height=300)
    print("-------------------------")

# 可视化
def visAnd_pred(BatchPointCloud,pred,label):
    """
    :param BatchPointCloud: 要预测的点云 B*N*3的矩阵，这里取10*1024*3
    :param pred: 预测的类别，10个pred值，
    :param label: 标签，10个label值
    :return: 如果可视化成功，返回True
    """
    num_pointcloud=BatchPointCloud.shape[0]

    # 一个循环，进行可视化，并打印预测值和label
    for i in range(num_pointcloud):
        print('------------num:{}-------------'.format(i))

        print("pred：{}, label:{}".format(pred[i],label[i]))
        print("pred: {}, label:{}".format(SHAPE_NAMES[pred[i]],SHAPE_NAMES[label[i]]) )

        # print("BatchPointCloud[i,:,:].reshape(-1,3):{}".format(BatchPointCloud[i,:,:].reshape(-1,3).shape))
        visPointCloudByOpen3d(BatchPointCloud[i,:,:].reshape(-1,3) )
        fout.write('pred: %d,  label: %d\n' % (pred[i], label[i]))         # 在 pred_label.txt  中记录下 <预测值，label>





def eval_my_data(sess,ops,num_votes=1):

    log_string('----'+str("Predict ply_data_test0.h5[0:10]")+'----')
    current_data, current_label = provider.loadDataFile(TEST_FILES[0])
    current_data = current_data[:,0:NUM_POINT,:]    # current_data (2048, 1024, 3)
    current_label = np.squeeze(current_label)       # 去掉维数为1的维度   current_label（2048）

    start_idx=0
    end_idx=10
    vote_idx=0
    is_training = False
    rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
                                                        vote_idx/float(num_votes) * np.pi * 2)      # B*N*3的点云
    feed_dict = {ops['pointclouds_pl']: rotated_data,
                 ops['labels_pl']: current_label[start_idx:end_idx],
                 ops['is_training_pl']: is_training}
    loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                  feed_dict=feed_dict)

    # 得到预测的类别
    pred_val = np.argmax(pred_val, 1)     # 预测的类别
    print("pred_val:{}".format(pred_val))  # 应该是一个10*1的矩阵
    print("current_label:{}".format(current_label[start_idx:end_idx]))   # 10*1的矩阵

    # 可视化点云并且打印结果
    visAnd_pred(current_data[start_idx:end_idx,:,:],pred_val,current_label[start_idx:end_idx] )

    print("--------------END---------------")




if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)    # 每个batch就评估一次
    LOG_FOUT.close()
