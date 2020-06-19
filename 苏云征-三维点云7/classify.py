'''
@Description:  三层神经网络实现分类
@Author: Su Yunzheng
@Date: 2020-06-17 15:40:53
@LastEditTime: 2020-06-18 16:12:21
@LastEditors: Su Yunzheng
'''


import numpy as np
# import sklearn
import matplotlib 
import matplotlib.pyplot as plt

from getData import gen_classify_data 




# 可视化
# input:sample_x(2*n), sample_y(n,)
def visualization(sample_x, sam_type, save=False, path='inference/sample-data.png'):
    samp = plt.figure()
    samp_ax = samp.add_subplot(1, 1, 1)
    samp_ax.set_title('sample data')
    type = sam_type                 # 给定一个label(n,)
    samp_ax.scatter(sample_x[0], sample_x[1], c=type, linewidths=type)
    if save:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.show()

# 可视化对比图
# 可视化
# input:sample_x(2*1000), sam_type(1000,), pred_type(1000,)
def compare_visual(sample_x, sam_type, pred_type, save=False, path='inference/inference-compare.png',
                   title='inference comparison', loss=0):

    # 查看输出的格式  
    # sample_x:(2, 1000),sample_y:(3, 1000),predict_y:(3, 1000)
    # print('sample_x:{},sample_y:{},predict_y:{}'.format(sample_x.shape, sample_y.shape, predict_y.shape))
    
    comp = plt.figure()
    comp.suptitle(title)
    samp_ax = plt.subplot2grid((2, 2), (0, 0))              # 划分位置
    pred_ax = plt.subplot2grid((2, 2), (0, 1))
    comp_ax = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    samp_ax.set_title('sample results')
    pred_ax.set_title('predict results')
    comp_ax.set_title('comparison loss= %.5f' %loss)
    # sam_type = np.where(sample_y.T == sample_y.max())[1]        # 1000个点的类别 （1000，）
    # pred_type = np.where(predict_y.T == predict_y.max())[1]     # 1000个点的类别 0 1 2 

    # print("predict_y:{}".format(predict_y.shape))   # 3*1000
    # print("pred_type :{},sam_type shape:{}".format(pred_type,sam_type.shape))
    # print('sam_type:{}'.format(sam_type))

    d_type = pred_type - sam_type     # label和predict的差异
    # print('type:{}'.format(type))  
    samp_ax.scatter(sample_x[0], sample_x[1], c=sam_type, linewidths=sam_type)
    pred_ax.scatter(sample_x[0], sample_x[1], c=pred_type, linewidths=pred_type)
    comp_ax.scatter(sample_x[0], sample_x[1], c=d_type)                # 根据预测结果的不同，差异越大，其余颜色的数目就越多
    if save:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.show()



# 生成作业数据数据 X为坐标，y为标签
X,y=gen_classify_data(200)
y=np.argmax(y,axis=0)         # [1,0,0]-->0;        [0,1,0]-->1;        [0,0,1]-->2;
X=np.transpose(X)  
# print("y :\n{}".format(y))
# print("y shape :\n{}".format(y.shape))
print("X :{},y :{}".format(X.shape,y.shape))
print("----------------------------------------")



num_examples = len(X) # 样本数   本例题为200
nn_input_dim = 2 # 输入的维度   本例题有两个特征向量
# nn_output_dim = 2 # 输出的类别个数  分为两类
nn_output_dim = 3 # 输出的类别个数  分为三类
 
# 梯度下降参数
epsilon = 0.01 # 学习率
reg_lambda = 0.01 # 正则化参数

# 定义损失函数，计算损失(才能用梯度下降啊...)
def calculate_loss(model,X=X,y=y,num_examples=num_examples):
    '''
    input: model,  X,  y,  num_examples样本的数量(200或者1000)
    '''
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
#     print("W1:{}\nb1:{}\nW2:{}\nb2:{}\n".format(W1,b1,W2,b2))
#     print("X:{}\n".format(X.shape))
    # 向前推进，前向运算
    z1 = X.dot(W1) + b1  #200*2*2*10 =200*10,  200*10+200*10=200*10 ，z1:200*3
    # print("X.dot(W1):{}".format(X.dot(W1).shape))
    # print("z1:{}\n".format(z1.shape))
    a1 = np.tanh(z1)  #a1:200*10
    # print("a1:{}".format(a1.shape))  # 激活


    z2 = a1.dot(W2) + b2  #200*10*10*3=200*3  , b2:1*3->200*3 , z2:200*3
    exp_scores = np.exp(z2)  #200*3                 # 全部取指数   softmax函数
    # print("exp_scores.shape: {} ".format(exp_scores.shape)) 
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  #np.sum()  将矩阵按列相加，全部加到第一列 , 200*3/200*1->200*3/200*3
#     print("np.sum:{}".format(np.sum(exp_scores, axis=1, keepdims=True).shape))
    # print("probs:{}".format(probs))    # 概率值  200*3
    # 计算损失   交叉熵函数
    corect_logprobs = -np.log(probs[range(num_examples), y])  #取prob这个200*3的概率矩阵的每一行，具体是第几列是靠对应的y来确定的 #200*1
#     print("y:{}".format(y))
    data_loss = np.sum(corect_logprobs)    #200行加在一起->1*1  一个数
#     print("corect_logprobs:{}".format(corect_logprobs.shape))  #200*1
#     print("data_loss:{}".format(data_loss))  #200行加在一起->1*1  一个数
    
    # 也得加一下正则化项
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))  #W1：2*3 W2:3*2  data_loss是一个数
#     print("data_loss:{}".format(data_loss))
#     print("1./num_examples * data_loss:{}".format(1.*data_loss/num_examples  ))
    return 1./num_examples * data_loss   #返回一个数，作为损失值

# 完整的训练建模函数定义
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    '''
    参数：
    1) nn_hdim: 隐层节点个数   作业为10
    2）num_passes: 梯度下降迭代次数
    3）print_loss: 设定为True的话，每1000次迭代输出一次loss的当前值
    '''
    # 随机初始化一下权重呗
    np.random.seed(0)  #seed只对第一组随机数起作用
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim) # nn.sqrt打印nn_input_dim=2的开方，也就是1.414
#     print("nn.sqrt:{}".format(np.sqrt(nn_input_dim)))
#     print("W1:{}".format(W1))
#     print(" np.random.randn(nn_input_dim, nn_hdim):{}",format( np.random.randn(nn_input_dim, nn_hdim)))
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)  
    b2 = np.zeros((1, nn_output_dim))
 
    # 这是咱们最后学到的模型
    model = {}
    loss=[]    # 损失
    # 开始梯度下降...
    for i in range(0, num_passes):
 
        # 前向运算计算loss
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)   # 激活函数 tanh
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)    # softmax 得到概率
        # print("probs shape: {}".format(probs.shape))     #  概率值  200*3
 
        # 反向传播
        delta3 = probs             #### 200*2  每一列表示对应的概率
        delta3[range(num_examples), y] -= 1    # 200
        # print("delta3:{}".format(delta3.shape))   ################################
        # print("delta3:{}".format(delta3))   ################################
        
        # 增量
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        # 加上正则化项
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
 
        # 梯度下降 更新参数
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
         
        # 得到的模型实际上就是这些权重
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        loss.append(calculate_loss(model))              # 将所有的loss放在一个list中
        # 如果设定print_loss了，那我们汇报一下中间状况
        if print_loss and i % 1000 == 0:
          print("Loss after iteration {}: {} ".format(i,calculate_loss(model)))

    # print("loss:{}".format(loss))     # 输出总的loss  一个list个格式

    # 绘制train-loss曲线
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('train-loss')
    ax.set_ylim([0, 1])
    ax.plot(loss)
    plt.savefig('./inference/train-loss')
    plt.show()
    


    return model


# 判定结果的函数
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 前向运算
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)  #200*3
    # 计算概率输出最大概率对应的类别
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #200*3
    # print('probs:{}'.format(probs))    # 输出概率矩阵
    return np.argmax(probs, axis=1)  #返回200行中，每一行的最大的值，得到的矩阵是一个200*1的矩阵，表示200个元素对应的类别


# 计算精度
def getAcc(pred,y):
    ac=sum(pred==y)/len(pred)   ## 统计两个数组相同元素的数目
    # ac=sum(pred==y)   ## 统计两个数组相同元素的数目
    return ac*100.0



# 建立隐层有10个节点(神经元)的神经网络
model = build_model(10, print_loss=True)
print("-------------------------")
# print("model:{}".format(model))
# 评价训练集
pred=predict(model,X)   # 得到预测值

# 计算损失
train_loss=calculate_loss(model,X,y)
print("train_loss:{}".format(train_loss))

# print("pred:{}".format(pred))
# print("y :{}".format(y))

# 计算精度
# print("y shape:{},pred shape:{}".format(y.shape,pred.shape))         #  y shape:(200,),pred shape:(200,)
print("Train Acc:{}%".format(getAcc(pred,y)))


# 可视化训练集
visualization(np.transpose(X),y,True,path='./inference/train.png')
compare_visual(np.transpose(X),y,pred,True,path='./inference/compare_train.png',loss=train_loss)



print("------------------Test------------------")
##### 生成测试集数据 1000个点
X_test,y_test=gen_classify_data(1000)
y_test=np.argmax(y_test,axis=0)         # [1,0,0]-->0;        [0,1,0]-->1;        [0,0,1]-->2;
X_test=np.transpose(X_test)             # 2*1000 ---> 1000*2
# print("y :\n{}".format(y))
# print("y shape :\n{}".format(y.shape))
print("X_test :{},y_test :{}".format(X_test.shape,y_test.shape))        # X_test :(1000, 2),y_test :(1000,)


# 得到预测值
pred_test = predict(model,X_test)   # 得到预测值

# 计算损失
test_loss = calculate_loss(model,X_test,y_test,len(X_test))
print("test_loss:{}".format(test_loss))

# 计算精度
print("Test Acc:{}%".format(getAcc(pred_test,y_test)))


# 可视化训练集
visualization(np.transpose(X_test),y_test,save=True,path='inference/test.png')
compare_visual(np.transpose(X_test),y_test,pred_test,save=True,path='inference/compare_test.png',loss=test_loss)

