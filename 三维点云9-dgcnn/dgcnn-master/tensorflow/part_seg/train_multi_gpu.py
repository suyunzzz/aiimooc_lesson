import argparse
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider
import part_seg_model as model

TOWER_NAME = 'tower'

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int, default=1, help='The number of GPUs to use [default: 2]')
parser.add_argument('--batch', type=int, default=4, help='Batch Size per GPU during training [default: 32]')
parser.add_argument('--epoch', type=int, default=11, help='Epoch to run [default: 50]')
parser.add_argument('--point_num', type=int, default=2048, help='Point Number [256/512/1024/2048]')
parser.add_argument('--output_dir', type=str, default='train_results', help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
FLAGS = parser.parse_args()

hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')

# MAIN SCRIPT
point_num = FLAGS.point_num
batch_size = FLAGS.batch
output_dir = FLAGS.output_dir

if not os.path.exists(output_dir):
  os.mkdir(output_dir)

# color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
# color_map = json.load(open(color_map_file, 'r'))

all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
fin = open(all_obj_cats_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]   # list [[Airplane,02691156],[],[],...[]]
all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]  # [(Airplane,02691156),(),...()]
fin.close()

all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))   # list
NUM_CATEGORIES = 16     # 一共有16类
NUM_PART_CATS = len(all_cats)   # 16类的物体，一共可以分为50个part

print('#### Batch Size Per GPU: {0}'.format(batch_size))
print('#### Point Number: {0}'.format(point_num))
print('#### Using GPUs: {0}'.format(FLAGS.num_gpu))

DECAY_STEP = 16881 * 20
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-5

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.003
MOMENTUM = 0.9
TRAINING_EPOCHES = FLAGS.epoch
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'val_hdf5_file_list.txt')

# 训练的模型文件夹
MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
  os.mkdir(MODEL_STORAGE_PATH)

# logs文件夹
LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
  os.mkdir(LOG_STORAGE_PATH)

# summaries文件夹
SUMMARIES_FOLDER =  os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
  os.mkdir(SUMMARIES_FOLDER)

# 打印并保存日志
def printout(flog, data):
  print(data)
  flog.write(data + '\n')

# labels转换为onehot编码
def convert_label_to_one_hot(labels):
  label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))   # batch_size*16
  for idx in range(labels.shape[0]):
    label_one_hot[idx, labels[idx]] = 1
  return label_one_hot

# 计算平均梯度
def average_gradients(tower_grads):
  """Calculate average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been 
     averaged across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      if g is None:
        continue
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  with tf.Graph().as_default(), tf.device('/cpu:0'):

    batch = tf.Variable(0, trainable=False)
    
    learning_rate = tf.train.exponential_decay(
            BASE_LEARNING_RATE,     # base learning rate
            batch * batch_size,     # global_var indicating the number of steps
            DECAY_STEP,             # step size
            DECAY_RATE,             # decay rate
            staircase=True          # Stair-case or continuous decreasing
            )
    learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)
  
    bn_momentum = tf.train.exponential_decay(
          BN_INIT_DECAY,
          batch*batch_size,
          BN_DECAY_DECAY_STEP,
          BN_DECAY_DECAY_RATE,
          staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

    lr_op = tf.summary.scalar('learning_rate', learning_rate)
    batch_op = tf.summary.scalar('batch_number', batch)
    bn_decay_op = tf.summary.scalar('bn_decay', bn_decay)

    trainer = tf.train.AdamOptimizer(learning_rate)

    # store tensors for different gpus
    tower_grads = []
    pointclouds_phs = []
    input_label_phs = []
    seg_phs =[]
    is_training_phs =[]

    # 变量
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(FLAGS.num_gpu):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
            pointclouds_phs.append(tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))) # for points  4*2048*3
            input_label_phs.append(tf.placeholder(tf.float32, shape=(batch_size, NUM_CATEGORIES))) # for one-hot category label  # 4*16
            seg_phs.append(tf.placeholder(tf.int32, shape=(batch_size, point_num))) # for part labels     # labels  4*2048
            is_training_phs.append(tf.placeholder(tf.bool, shape=()))

            # 预测值
            # seg_pred : B*N*part_num=4*2048*50
            seg_pred = model.get_model(pointclouds_phs[-1], input_label_phs[-1], \
                is_training=is_training_phs[-1], bn_decay=bn_decay, cat_num=NUM_CATEGORIES, \
                part_num=NUM_PART_CATS, batch_size=batch_size, num_point=point_num, weight_decay=FLAGS.wd)      # cat_num=16,part_num=50

            # loss:这个batch的总的loss
            # per_instance_loss:每个点云的loss,shape=4
            # per_instance_seg_pred_res : 每个点云的分割结果，由seg_pred取max得到的，shape=4*2048
            loss, per_instance_seg_loss, per_instance_seg_pred_res  \
              = model.get_loss(seg_pred, seg_phs[-1])

            # placeholder
            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            total_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            seg_training_acc_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_acc_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_acc_avg_cat_ph = tf.placeholder(tf.float32, shape=())

            # scalar
            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            total_test_loss_sum_op = tf.summary.scalar('total_testing_loss', total_testing_loss_ph)

            seg_train_acc_sum_op = tf.summary.scalar('seg_training_acc', seg_training_acc_ph)
            seg_test_acc_sum_op = tf.summary.scalar('seg_testing_acc', seg_testing_acc_ph)
            seg_test_acc_avg_cat_op = tf.summary.scalar('seg_testing_acc_avg_cat', seg_testing_acc_avg_cat_ph)

            tf.get_variable_scope().reuse_variables()

            grads = trainer.compute_gradients(loss)

            tower_grads.append(grads)

    grads = average_gradients(tower_grads)   # 计算平均梯度

    train_op = trainer.apply_gradients(grads, global_step=batch)

    saver = tf.train.Saver(tf.global_variables(), sharded=True, max_to_keep=20)     # 保存变量

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    
    init = tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer())
    sess.run(init)

    train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)  # 保存graph
    test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

    train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)    # [train0.h5,train1.h5,...train5.h5]
    num_train_file = len(train_file_list)               # 6
    test_file_list = provider.getDataFiles(TESTING_FILE_LIST)     # [ply_data_val0.h5]
    num_test_file = len(test_file_list)                     # 1

    # 保存终端输入
    fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
    fcmd.write(str(FLAGS))
    fcmd.close()

    # write logs to the disk
    flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')

    def train_one_epoch(train_file_idx, epoch_num):     # 输入为打乱后的顺序
      is_training = True

      # 遍历每一个train文件
      for i in range(num_train_file):
        cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[train_file_idx[i]])  #  获取当前的train的点云文件
        printout(flog, 'Loading train file ' + cur_train_filename)

        cur_data, cur_labels, cur_seg = provider.load_h5_data_label_seg(cur_train_filename)
        cur_data, cur_labels, order = provider.shuffle_data(cur_data, np.squeeze(cur_labels))
        cur_seg = cur_seg[order, ...]

        cur_labels_one_hot = convert_label_to_one_hot(cur_labels)

        num_data = len(cur_labels)
        num_batch = num_data // (FLAGS.num_gpu * batch_size) # For all working gpus  num——batch代表这个train文件分几个batch

        total_loss = 0.0
        total_seg_acc = 0.0

        # 对每一个batch
        for j in range(num_batch):
          begidx_0 = j * batch_size       # 第一个gpu
          endidx_0 = (j + 1) * batch_size
          begidx_1 = (j + 1) * batch_size   # 第二个gpu
          endidx_1 = (j + 2) * batch_size

          feed_dict = {
              # For the first gpu
              pointclouds_phs[0]: cur_data[begidx_0: endidx_0, ...],      # 4*2048*3
              input_label_phs[0]: cur_labels_one_hot[begidx_0: endidx_0, ...],    # 4*16
              seg_phs[0]: cur_seg[begidx_0: endidx_0, ...],     # 4*2048  ，每一个数都在0-49之间
              is_training_phs[0]: is_training, 
              # # For the second gpu
              # pointclouds_phs[1]: cur_data[begidx_1: endidx_1, ...],
              # input_label_phs[1]: cur_labels_one_hot[begidx_1: endidx_1, ...],
              # seg_phs[1]: cur_seg[begidx_1: endidx_1, ...],
              # is_training_phs[1]: is_training,
              }


          # train_op is for both gpus, and the others are for gpu_1
          # 每一个batch的平均损失、每一个点云的损失、分割预测值
          _, loss_val, per_instance_seg_loss_val, seg_pred_val, pred_seg_res \
              = sess.run([train_op, loss, per_instance_seg_loss, seg_pred, per_instance_seg_pred_res], \
              feed_dict=feed_dict)

          # per_instance_part_acc = np.mean(pred_seg_res == cur_seg[begidx_1: endidx_1, ...], axis=1)
          per_instance_part_acc = np.mean(pred_seg_res == cur_seg[begidx_0: endidx_0, ...], axis=1)     # 每一个点云的分割精度

          average_part_acc = np.mean(per_instance_part_acc)        # 当前batch的平均精度

          total_loss += loss_val
          total_seg_acc += average_part_acc
          # 至此，一个train文件遍历完成

        total_loss = total_loss * 1.0 / num_batch     # 每一个train文件都得到一个loss和seg_acc
        total_seg_acc = total_seg_acc * 1.0 / num_batch

        # 绘制图
        lr_sum, bn_decay_sum, batch_sum, train_loss_sum, train_seg_acc_sum = sess.run(\
            [lr_op, bn_decay_op, batch_op, total_train_loss_sum_op, seg_train_acc_sum_op], \
            feed_dict={total_training_loss_ph: total_loss, seg_training_acc_ph: total_seg_acc})

        train_writer.add_summary(train_loss_sum, i + epoch_num * num_train_file)   # epoch_num是一个不断变化的值
        train_writer.add_summary(lr_sum, i + epoch_num * num_train_file)
        train_writer.add_summary(bn_decay_sum, i + epoch_num * num_train_file)
        train_writer.add_summary(train_seg_acc_sum, i + epoch_num * num_train_file)
        train_writer.add_summary(batch_sum, i + epoch_num * num_train_file)

        printout(flog, '\tTanin_file: {},Training Total Mean_loss: {}'.format(i,total_loss))    # 每一个train文件的loss和acc
        printout(flog, '\t\tTanin_file: {} ,Training Seg Accuracy: {}'.format(i,total_seg_acc))

    def eval_one_epoch(epoch_num):
      is_training = False

      total_loss = 0.0
      total_seg_acc = 0.0
      total_seen = 0

      total_seg_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)  # [0. 0. ....0.]  16个0
      total_seen_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.int32)  # [0 0 0 0 ...0]

      for i in range(num_test_file):   # i= 0  num_test_file=1
        cur_test_filename = os.path.join(hdf5_data_dir, test_file_list[i])    # ply_data_val0.h5
        printout(flog, 'Loading test file ' + cur_test_filename)  # ply_data_val0.h5

        # 读取数据
        # data = f['data'][:] # (1870, 2048, 3)
        # label = f['label'][:] # (1870，1)  1870个点云的类别0-15
        # seg = f['pid'][:] # (1870, 2048)  ，表示每一个点属于的类别0-49，一共50类点
        cur_data, cur_labels, cur_seg = provider.load_h5_data_label_seg(cur_test_filename)
        cur_labels = np.squeeze(cur_labels)         # shape:(1870,)

        cur_labels_one_hot = convert_label_to_one_hot(cur_labels)   # 1870*16

        num_data = len(cur_labels)      # 1870个点云
        num_batch = num_data // batch_size    # 467个batch

        # Run on gpu_1, since the tensors used for evaluation are defined on gpu_1
        for j in range(num_batch):      # 0-466
          begidx = j * batch_size
          endidx = (j + 1) * batch_size
          # feed_dict = {
          #     pointclouds_phs[1]: cur_data[begidx: endidx, ...],
          #     input_label_phs[1]: cur_labels_one_hot[begidx: endidx, ...],
          #     seg_phs[1]: cur_seg[begidx: endidx, ...],
          #     is_training_phs[1]: is_training}

          feed_dict = {
            pointclouds_phs[0]: cur_data[begidx: endidx, ...],      # 4*2048*3
            input_label_phs[0]: cur_labels_one_hot[begidx: endidx, ...],    # 4*16
            seg_phs[0]: cur_seg[begidx: endidx, ...],                # 4*2048
            is_training_phs[0]: is_training}

          # pred_seg_res：每个点的分割结果 shape:4*2048
          # seg_pred_val:分割结果，概率编码 shape:4*2048*50
          # per_instance_seg_loss_val: 每一个点云的损失 shape:(4,)
          # loss_val：当前batch的平均损失 shape:()
          loss_val, per_instance_seg_loss_val, seg_pred_val, pred_seg_res \
              = sess.run([loss, per_instance_seg_loss, seg_pred, per_instance_seg_pred_res], \
              feed_dict=feed_dict)

          per_instance_part_acc = np.mean(pred_seg_res == cur_seg[begidx: endidx, ...], axis=1)  # 4个点云的每一个的分割精度  shape:(4,)
          average_part_acc = np.mean(per_instance_part_acc)     # 4个点云的平均分割精度

          total_seen += 1   # 以batch为单位
          total_loss += loss_val
          
          total_seg_acc += average_part_acc     # average_part_acc：当前batch的分割精度

          for shape_idx in range(begidx, endidx):   # 对当前batch中的每一个点云
            total_seen_per_cat[cur_labels[shape_idx]] += 1      # 统计看过的每一类点云的数量
            total_seg_acc_per_cat[cur_labels[shape_idx]] += per_instance_part_acc[shape_idx - begidx]   # 统计每一类点云的精度

      total_loss = total_loss * 1.0 / total_seen            # 总的分割损失
      total_seg_acc = total_seg_acc * 1.0 / total_seen      # 总的平均分割精度，以batch为单位

      # 绘制图
      test_loss_sum, test_seg_acc_sum = sess.run(\
          [total_test_loss_sum_op, seg_test_acc_sum_op], \
          feed_dict={total_testing_loss_ph: total_loss, \
          seg_testing_acc_ph: total_seg_acc})

      test_writer.add_summary(test_loss_sum, (epoch_num+1) * num_train_file-1)
      test_writer.add_summary(test_seg_acc_sum, (epoch_num+1) * num_train_file-1)


      printout(flog, '\t\tTesting Total Mean_loss: %f' % total_loss)
      printout(flog, '\t\tTesting Seg Accuracy: %f' % total_seg_acc)

      for cat_idx in range(NUM_CATEGORIES):     # 0-15
        if total_seen_per_cat[cat_idx] > 0:  # 如果看过这类物体，就打印看得数量
          printout(flog, '\n\t\tCategory %s Object Number: %d' % (all_obj_cats[cat_idx][0], total_seen_per_cat[cat_idx]))
          printout(flog, '\t\tCategory %s Seg Accuracy: %f' % (all_obj_cats[cat_idx][0], total_seg_acc_per_cat[cat_idx]/total_seen_per_cat[cat_idx]))

    if not os.path.exists(MODEL_STORAGE_PATH):
      os.mkdir(MODEL_STORAGE_PATH)

    for epoch in range(TRAINING_EPOCHES):
      printout(flog, '\n<<< Testing on the test dataset ...')
      eval_one_epoch(epoch)     # 评估

      printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))

      train_file_idx = np.arange(0, len(train_file_list))
      np.random.shuffle(train_file_idx)

      train_one_epoch(train_file_idx, epoch)      # 训练

      if epoch % 5 == 0:
        cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch)+'.ckpt'))
        printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

      flog.flush()

    flog.close()

if __name__=='__main__':
  train()
