'''
@Description: mlp MNIST手写数字体识别
@Author: Su Yunzheng
@Date: 2020-06-18 16:14:32
@LastEditTime: 2020-06-18 17:18:14
@LastEditors: Su Yunzheng
'''

### 查看tensorboard:
# (tf) C:\Users\11604>tensorboard --logdir F:\睿慕课\7\苏云征-三维点云7\logs  --host=127.0.0.1


import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA",one_hot=True)    #onehot对标签的标注，非onehot是1,2,3.onehot就是只有一个1其余全是0



#超参数（学习率，batch的大小，训练的轮数，多少轮展示一下loss）
learning_rate = 0.2     # 学习率
num_step = 20          # 迭代次数
batch_size = 128        # 一个batch的size
n_batch=mnist.train.num_examples//batch_size    # 有多少个batch
display_step =100     # 多少轮打印一次

#网络参数（有多少层网络，每层有多少个神经元，整个网络的输入是多少维度的，输出是多少维度的）
num_input = 784     #(28*28)
h1=1000             # 隐藏层 这个是没用到的
num_class = 10      # 输出

#图的输入
X = tf.placeholder(tf.float32,[None,num_input])
Y = tf.placeholder(tf.float32,[None,num_class])

#网络的权重和偏向,如果是两个隐层的话需要定义三个权重，包括输出层
weights={
    'out':tf.Variable(tf.zeros([num_input,num_class]))

}

biase = {
    'out':tf.Variable(tf.zeros([num_class]))
}


#定义网络结构
def neural_net(x):
    # h1_out = tf.matmul(x,weights['h1'])+biase['h1']
    out_layer = tf.matmul(x,weights['out'])+biase['out']
    return out_layer


#模型输出处理
logits = neural_net(X)
prediction = tf.nn.softmax(logits)    # 得到batch_size*10的概率矩阵

# 定义损失和优化器
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_step = optimizer.minimize(loss_op)


#评估模型准确率
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))    #结果存放在一个布尔型列表中(argmax函数返回一维张量中最大的值所在的位置)
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))   #求准确率(tf.cast将布尔值转换为float型)

#初始化变量
init = tf.global_variables_initializer()

mse_summary=tf.summary.scalar("loss", loss_op)
# Create a summary to monitor accuracy tensor
acc_summary=tf.summary.scalar("accuracy", accuracy)
file_writer = tf.summary.FileWriter('./logs',tf.get_default_graph())

#开始训练
with tf.Session() as sess:
    sess.run(init)
    for step in range(num_step):   # 总迭代
        for batch in range(n_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,los,ac=sess.run([train_step,mse_summary,acc_summary],feed_dict={X:batch_x,Y:batch_y})
            # 记录下来每一次训练的数据
            file_writer.add_summary(los,step*n_batch+batch)   # loss
            file_writer.add_summary(ac,step*n_batch+batch)    # acc
            # if step % display_step == 0 or step == 1:
                # print("step:{},loss:{},acc:{}".format(step,loss,acc))
        acc=sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels})
        print("Iter:{},testing Accuracy:{}".format(step,acc))
    print("优化完成!")

    #训练完模型后，开始测试

    pred_pic=mnist.test.images[0].reshape(1,-1)
    pred_num=sess.run(prediction,feed_dict={X:pred_pic})
    print("预测结果：{}".format( np.argmax(pred_num) ))
    # 读取图片
    # print(mnist.test.images.shape)          #(10000, 784)
    plt.imshow(mnist.test.images[0].reshape(28,28))
    plt.show()


