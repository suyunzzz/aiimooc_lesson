'''
@Description: cnn MNIST手写数字体识别
@Author: Su Yunzheng
@Date: 2020-06-18 16:14:39
@LastEditTime: 2020-06-18 16:14:39
@LastEditors: Su Yunzheng
'''

### 查看tensorboard:
# (tf) C:\Users\11604>tensorboard --logdir F:\睿慕课\7\苏云征-三维点云7\logs  --host=127.0.0.1

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt



mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)

logs_path = './log/example/' # log存放位置
batch_size = 50
n_batch=mnist.train.num_examples//batch_size    # 有多少个batch

def weight_variable(shape,namew='w'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=namew)


def bias_variable(shape,nameb='b'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=nameb)


def conv2d(x, W,namec='c'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',name=namec)


def max_pool_2x2(x,namep='p'):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name=namep)


x = tf.placeholder(tf.float32, [None, 784],name='xinput')
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Conv1 Layer
W_conv1 = weight_variable([5, 5, 1, 32],'w1')
b_conv1 = bias_variable([32],'b1')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1,'c1')
h_pool1 = max_pool_2x2(h_conv1,'p1')

# Conv2 Layer
W_conv2 = weight_variable([5, 5, 32, 64],'w2')
b_conv2 = bias_variable([64],'b2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024],'wf1')
b_fc1 = bias_variable([1024],'bf1')
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32,name="prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10],'wfull2')
b_fc2 = bias_variable([10],'bf2')
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init=tf.global_variables_initializer()

# tensorboard相关的
# Create a summary to monitor cost tensor
mse_summary=tf.summary.scalar("loss", cross_entropy)
# Create a summary to monitor accuracy tensor
acc_summary=tf.summary.scalar("accuracy", accuracy)
file_writer = tf.summary.FileWriter('./logs',tf.get_default_graph())
# Merge all summaries into a single op
# merged_summary_op = tf.summary.merge_all()


#开始训练
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    # summary_writer = tf.summary.FileWriter("logs/", graph=tf.get_default_graph())

    saver = tf.train.Saver()
    for i in range(2):
        for batch_i in range(n_batch):
            batch = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            summary_str = mse_summary.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            accuracy_str=acc_summary.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            # 记录下来每一次训练的数据
            file_writer.add_summary(summary_str,i*n_batch+batch_i)
            file_writer.add_summary(accuracy_str,i*n_batch+batch_i)


        # 训练集精度
            if (batch_i%500==0):
                train_accuracy = sess.run(accuracy,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("Iter %d, batch_i %d,training accuracy %g" % (i,batch_i, train_accuracy))


        # 一次遍历完整个数据集，打印测试集精度
        test_acc=sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print("Iter %d, test accuracy %g" % (i, test_acc))

    saver.save(sess, "Model/model.ckpt")
    print("model saved!")

    # 测试
    #训练完模型后，开始测试

    pred_pic=mnist.test.images[0].reshape(1,-1)
    # print("pred_pic:{}".format(pred_pic.shape))  # pred_pic:(1, 784)
    pred_num=sess.run(y_conv,feed_dict={x:pred_pic,keep_prob: 1.0})   # 这是一个1*10的概率矩阵 , 这里feed_dict中要有keep_prob,否则会报错
    # print("概率矩阵softmax:{}".format(pred_num.shape))
    print("预测结果：{}".format( np.argmax(pred_num) ))


# 读取图片
    # print(mnist.test.images.shape)          #(10000, 784)
    plt.imshow(mnist.test.images[0].reshape(28,28))
    plt.show()


