'''
使用模型对seu数据集进行配准
(py36tensorflow13) ➜  flownet3d git:(master) ✗ python evaluate_seu.py --model model_concat_upsa_eval_kitti --gpu 0 --dataset kitti_dataset \
    --data kitti_rm_ground \
    --log_dir log_evaluate_seu \
    --model_path log_train/model.ckpt \
    --num_point 16384 \
    --batch_size 1 \


'''

import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pickle
from utils.vis import vis_cloud,vis_cloud1,vis_cloud1andcolor
from vis_kitti_test import *

file1='seu_pcd/data/f0.txt'
file2='seu_pcd/data/f1.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_concat_upsa', help='Model name [default: model_concat_upsa]')
parser.add_argument('--dataset', default='flying_things_dataset', help='Dataset name [default: flying_things_dataset]')
parser.add_argument('--data', default='data_preprocessing/data_processed_maxcut_35_20k_2k_8192', help='Dataset directory [default: /data_preprocessing/data_processed_maxcut_35_20k_2k_8192]')
parser.add_argument('--model_path', default='log_train/model.ckpt', help='model checkpoint file path [default: log_train/model.ckpt]')
parser.add_argument('--log_dir', default='log_evaluate', help='Log dir [default: log_evaluate]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
# DATA = FLAGS.data
GPU_INDEX = FLAGS.gpu

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
MODEL_PATH = FLAGS.model_path
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# DATASET = importlib.import_module(FLAGS.dataset)
# TEST_DATASET = DATASET.SceneflowDataset(DATA, npoints=NUM_POINT, train=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, masks_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            print("--- Get model and loss")
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=None)       # pred,B*n*3 
            loss = MODEL.get_loss(pred, labels_pl, masks_pl, end_points)
            tf.summary.scalar('loss', loss)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        
        # 读入模型
        saver.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'masks_pl': masks_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,        # 预测值
               'loss': loss}        # loss 

        eval_one_epoch(sess, ops)


def load_txt(file1,file2):
    '''
    读取txt数据,随机选择NUM_POINT个点
    最终返回的数据:(1,16384*2,6)
    '''
    f1=np.loadtxt(file1)
    f1=f1[:,:3]

    f1_color=np.zeros((f1.shape[0],3))
    print('color:{}'.format(f1_color.shape))
    print('f1:{}'.format(f1.shape))


    f1_cloud=np.hstack((f1,f1_color))       # n*6
    f1_cloud=f1_cloud[:NUM_POINT,:]         # 选择前16384个点,NUM_POINT*6

    f2=np.loadtxt(file2)
    f2=f2[:,:3]

    f2_color=np.zeros((f2.shape[0],3))

    f2_cloud=np.hstack((f2,f2_color))       # n*6
    f2_cloud=f2_cloud[:NUM_POINT,:]         # 选择前16384个点,NUM_POINT*6   

    cloud=np.vstack((f1_cloud,f2_cloud))        # (16384*2,6)

    # 扩充维度
    cloud=cloud[np.newaxis,:]               # (1,16384*2,6)
    return cloud 
    

def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False


    log_string(str(datetime.now()))
    log_string('---- EVALUATION ----')


    # 读取数据
    cloud=load_txt(file1,file2)         # (1,NUM_POINTS,6)


    # ---------------------------------------------------------------------
    # ---- INFERENCE BELOW ----
    SHUFFLE_TIMES = 10
    for shuffle_cnt in range(SHUFFLE_TIMES):
        shuffle_idx = np.arange(NUM_POINT)
        np.random.shuffle(shuffle_idx)
        batch_data_new=np.zeros((1,NUM_POINT*2,6))
        batch_data_new[:,0:NUM_POINT,:] = cloud[:,shuffle_idx,:]       # 前一帧        # (1, 16384, 6) (x,y,z,color,color,color)
        batch_data_new[:,NUM_POINT:,:] = cloud[:,NUM_POINT+shuffle_idx,:]      # 后一帧
        feed_dict = {ops['pointclouds_pl']: batch_data_new,         # (1,16384*2,6)
                        ops['is_training_pl']: is_training}
        pred_val = sess.run([ops['pred']], feed_dict=feed_dict)      # 运行模型，进行推理
        pred_val=np.asarray(pred_val)
        print('pred_val:{}'.format(pred_val.shape))
        # 保存预测的flow
        # print("batch_data_new[:,0:NUM_POINT,:]:{}".format(batch_data_new[:,0:NUM_POINT,:].shape))
        np.savetxt('seu_pcd/f1.txt',batch_data_new[:,0:NUM_POINT,:3].reshape(-1,3))     # f1
        np.savetxt('seu_pcd/f3.txt',batch_data_new[:,NUM_POINT:,:3].reshape(-1,3))      # f2
        np.savetxt('seu_pcd/pred_val.txt',pred_val.reshape(-1,3))                                         # flow_pred
        vis_kitti(batch_data_new[:,0:NUM_POINT,:3].reshape(-1,3),
                batch_data_new[:,NUM_POINT:,:3].reshape(-1,3),
                pred_val.reshape(-1,3))
        

            
        # ---- INFERENCE ABOVE ----


    print('---------')




if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    evaluate()
    LOG_FOUT.close()
