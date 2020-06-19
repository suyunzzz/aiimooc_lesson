""""
Date:2017.4.28
Neural Network design homework
Using 3-layers-BP NN to classification and regression
author:Suo Chuanzhe
email: suo_ivy@foxmail.com

run_inference
load the training model and infer the output
visualize the training and prediction data.

"""

import numpy as np
import os
import time
from BPModel import BPModel

import matplotlib.pyplot as plt

plt.rcParams['agg.path.chunksize'] = 50000


# generate dataset to classify
def gen_classify_data(numbers):
    sample_input = (np.random.rand(2, numbers) - 0.5) * 4
    sample_output = np.array([[], [], []])

    for i in range(numbers):
        sample = sample_input[:, i]
        x = sample[0]
        y = sample[1]

        if ((x > -1) & (x < 1)) == 1:
            if ((y > x / 2 + 1 / 2) & (y < 1)) == 1:
                sample_output = np.append(sample_output, np.array([[0], [1], [0]]), axis=1)
            elif ((y < -0.5) & (y > -1.5)) == 1:
                sample_output = np.append(sample_output, np.array([[0], [0], [1]]), axis=1)
            else:
                sample_output = np.append(sample_output, np.array([[1], [0], [0]]), axis=1)
        else:
            sample_output = np.append(sample_output, np.array([[1], [0], [0]]), axis=1)

    return sample_input, sample_output


def visualization(sample_x, sample_y, save=False, path='inference/sample-data.png'):
    samp = plt.figure()
    samp_ax = samp.add_subplot(1, 1, 1)
    samp_ax.set_title('sample data')
    type = np.where(sample_y.T == sample_y.max())[1]
    samp_ax.scatter(sample_x[0], sample_x[1], c='g', linewidths=type)
    if save:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.show()

# 可视化
def compare_visual(sample_x, sample_y, predict_y, save=False, path='inference/inference-compare.png',
                   title='inference comparison', loss=0):

    # 查看输出的格式
    print('sample_x:{},sample_y:{},predict_y:{}'.format(sample_x.shape, sample_y.shape, predict_y.shape))
    
    comp = plt.figure()
    comp.suptitle(title)
    samp_ax = plt.subplot2grid((2, 2), (0, 0))              # 划分位置
    pred_ax = plt.subplot2grid((2, 2), (0, 1))
    comp_ax = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    samp_ax.set_title('sample results')
    pred_ax.set_title('predict results')
    comp_ax.set_title('comparison loss= %.5f' %loss)
    sam_type = np.where(sample_y.T == sample_y.max())[1]        # 1000个点的类别 （1000，）
    pred_type = np.where(predict_y.T == predict_y.max())[1]     # 1000个点的类别 0 1 2 

    # print("predict_y:{}".format(predict_y.shape))   # 3*1000
    # print("pred_type :{},sam_type shape:{}".format(pred_type,sam_type.shape))
    # print('sam_type:{}'.format(sam_type))

    type = pred_type - sam_type     # label和predict的差异
    # print('type:{}'.format(type))  
    samp_ax.scatter(sample_x[0], sample_x[1], c='g', linewidths=sam_type)
    pred_ax.scatter(sample_x[0], sample_x[1], c='g', linewidths=pred_type)
    comp_ax.scatter(sample_x[0], sample_x[1], c=type)                # 根据预测结果的不同 如果差异为0，有一个颜色，代表相同
    if save:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.show()


# generate dataset to regression
# def gen_regress_data():


# main function
def main():
    ### Optimizers ###
    # BGD_optimizer
    # def BGD_optimizer(self, param, hyper_param={'learn_rate': 0.01}):
    def BGD():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.BGD_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.2)

    # Momentum_optimizer
    # def Momentum_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):
    def Momentum():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Momentum_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.1, momentum_rate=0.9)

    # Nesterov Accelerated Gradient(NAG_optimizer)
    # def NAG_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):
    def NAG():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.NAG_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.1, momentum_rate=0.9)

    # Adagrad_optimizer
    # def Adagrad_optimizer(self, param, hyper_param={'learn_rate': 0.01}):
    def Adagrad():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Adagrad_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.1)

    # Adadelta_optimizer
    # def Adadelta_optimizer(self, param, hyper_param={'decay_rate': 0.9}):
    def Adadelta():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Adadelta_optimizer, 0.01,
                                iteration, evaluate, decay_rate=0.9)

    # RMSProp
    # def RMSProp_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay_rate': 0.9}):
    def RMSProp():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.RMSProp_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.01, decay_rate=0.9)

    # RMSProp_with_Nesterov
    # def RMSProp_Nesterov_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9, 'decay_rate': 0.9}):
    def RMSProp_Nesterov():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.RMSProp_Nesterov_optimizer,
                                0.01, iteration, evaluate, learn_rate=0.01, momentum_rate=0.9, decay_rate=0.9)

    # Adam
    # def Adam_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay1_rate': 0.9, 'decay2_rate': 0.999}):
    def Adam():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Adam_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.01, decay1_rate=0.9, decay2_rate=0.999)

    ### Activation ###
    # sigmoid
    def sigmoid():
        classifier.set_activation(classifier.sigmoid_activation, classifier.sigmoid_gradient,
                                  classifier.sigmoid_activation, classifier.sigmoid_gradient)

    # tanh
    def tanh():
        classifier.set_activation(classifier.tanh_activation, classifier.tanh_gradient,
                                  classifier.tanh_activation, classifier.tanh_gradient)

    # ReLU
    def ReLU():
        classifier.set_activation(classifier.ReLU_activation, classifier.ReLU_gradient,
                                  classifier.ReLU_activation, classifier.ReLU_gradient)

    eval_samples_x, eval_samples_y = gen_classify_data(1000)
    visualization(eval_samples_x, eval_samples_y)

    classifier = BPModel()

    evaluate = True  # Train model and evaluate with evaluate_samples simultaneously

    optimizer = {'BGD': BGD, 'Momentum': Momentum, 'NAG': NAG, 'Adagrad': Adagrad, 'Adadelta': Adadelta,
                 'RMSProp': RMSProp, 'RMSProp_Nesterov': RMSProp_Nesterov, 'Adam': Adam}
    activation = {'sigmoid': sigmoid, 'tanh': tanh, 'ReLU': ReLU}

    line = {'BGD': 'b-', 'Momentum': 'r-', 'NAG': 'g-', 'Adagrad': 'k-', 'Adadelta': 'y-', 'RMSProp': 'c-',
            'RMSProp_Nesterov': 'm-', 'Adam': 'k--'}

    if not os.path.exists('inference'):
        os.mkdir('inference')
    if not os.path.exists('models'):
        os.mkdir('models')

    filelist = []
    for file in os.listdir(os.getcwd() + '/models'):
        file_path = os.path.join(os.getcwd() + '/models', file)
        if os.path.isdir(file_path):
            pass
        else:
            name, extension = os.path.splitext(file)
            if extension == '.npy':
                filelist.append(file_path)

    for file in filelist:
        filename = os.path.basename(file)

        act_function = filename.split('-')[0]
        opti = filename.split('-')[1]
        
        classifier.load_model(file)
        activation[act_function]()

        results, loss = classifier.evaluate(eval_samples_x, eval_samples_y)
        predict_y = (list(map(lambda x: x == max(x), results.T)) * np.ones_like(results.T)).T
        compare_visual(eval_samples_x, eval_samples_y, predict_y, save=True,
                       path='inference/' + str(act_function) + '-' + str(opti) + '-inference.png',
                       title=str(act_function) + '-' + str(opti) + '-comparison', loss=loss)

        print(loss)


if __name__ == "__main__":
    main()
