""""
Date:2017.4.2
Neural Network design homework
Using 3-layers-BP NN to classification and regression
author:Suo Chuanzhe

"""

import numpy as np
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


# generate dataset to regression
# def gen_regress_data():


# main function
def main():

    train_samples_x, train_samples_y = gen_classify_data(200)
    classifier = BPModel(train_samples_x, train_samples_y, 10)
    # classifier.initialize_parameters(classifier.sigmoid_activation, classifier.sigmoid_gradient, classifier.sigmoid_activation, classifier.sigmoid_gradient)

    # tanh
    classifier.initialize_parameters(classifier.tanh_activation, classifier.tanh_gradient, classifier.tanh_activation, classifier.tanh_gradient)
    loss = classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Adadelta_optimizer, 0.01, 100000, decay_rate=0.9)

    # ReLU
    # classifier.initialize_parameters(classifier.ReLU_activation, classifier.ReLU_gradient, classifier.ReLU_activation, classifier.ReLU_gradient)

    learn_rate = 0.1


    # BGD_optimizer
    # def BGD_optimizer(self, param, hyper_param={'learn_rate': 0.01}):

    # loss = classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.BGD_optimizer, 0.01, 100000, learn_rate=0.1)


    # Momentum_optimizer
    # def Momentum_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):

    # loss = classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Momentum_optimizer, 0.01, 500000, learn_rate=0.1, momentum_rate=0.9)


    # Nesterov Accelerated Gradient(NAG_optimizer)
    # def NAG_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):

    # loss = classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.NAG_optimizer, 0.01, 100000, learn_rate=0.1, momentum_rate=0.9)


    # Adagrad_optimizer
    # def Adagrad_optimizer(self, param, hyper_param={'learn_rate': 0.01}):

    # loss = classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Adagrad_optimizer, 0.01, 100000, learn_rate=0.1)


    # Adadelta_optimizer
    # def Adadelta_optimizer(self, param, hyper_param={'decay_rate': 0.9}):



    # RMSProp
    # def RMSProp_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay_rate': 0.9}):

    # loss = classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.RMSProp_optimizer, 0.01, 1000000, learn_rate=0.01, decay_rate=0.9)


    # RMSProp_with_Nesterov
    # def RMSProp_Nesterov_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9, 'decay_rate': 0.9}):

    # loss = classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.RMSProp_Nesterov_optimizer, 0.01, 100000, learn_rate=0.01, momentum_rate=0.9, decay_rate=0.9)


    # Adam
    # def Adam_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay1_rate': 0.9, 'decay2_rate': 0.999}):

    # loss = classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Adam_optimizer, 0.001, 100000, learn_rate= 0.01, decay1_rate=0.9, decay2_rate=0.999)

    plt.figure()
    plt.plot(loss, 'g-')
    plt.show()


if __name__ == "__main__":
    main()
