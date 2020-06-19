""""
Date:2017.4.2
Neural Network design homework
Using 3-layers-BP NN to classification and regression
author:Suo Chuanzhe

"""

import numpy as np
import time

import matplotlib.pyplot as plt


class BPModel():
    # config model value
    """
     Input: data_input(array(IN_value_num,data_num))
            data_output(array(OUT_value_num,data_num)
            hidden_unit_number(N)
    """

    def __init__(self, data_input = np.array([[0],[0]]), data_output = np.array([[0],[0]]), hidden_units_number = 1):

        self.input = data_input
        self.output = data_output
        self.data_number = self.input.shape[1]    # 200个

        self.input_units_number = self.input.shape[0]   # 输入的数据维度 2 代表(x,y)
        self.hidden_units_number = hidden_units_number   # 隐藏层单元的数目  10个隐藏层单元
        self.output_units_number = self.output.shape[0]   # 输出的数据维度 3 代表类别

        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = np.array([[], [], [], []])

        self.hidden_activation = self.sigmoid_activation     # 隐层激活函数
        self.hidden_activation_gradient = self.sigmoid_gradient  #  隐层梯度
        self.output_activation = self.sigmoid_activation   #  输出层激活函数
        self.output_activation_gradient = self.sigmoid_gradient   # 输出层梯度

        self.loss_function = self.L2_loss             # 损失函数
        self.loss_function_gradient = self.L2_loss_gradient     # 损失函数梯度

        self.optimizer = self.BGD_optimizer           # 优化器

        self.eval_input = np.array([])    # 评估输出
        self.eval_output = np.array([])     # 评估输出

    # Initialize values  #3-Layers BP
    """
     input: activation_function(function) activation_gradient(function)
    """

    def initialize_parameters(self, hidden_activation, hidden_activation_gradient,
                              output_activation, output_activation_gradient):

        self.weight_1 = 0.2 * np.random.rand(self.hidden_units_number, self.input_units_number) - 0.1    # 10*2 w1
        self.bias_1 = 0.2 * np.random.rand(self.hidden_units_number, 1) - 0.1                              # 10*1 b1
        self.weight_2 = 0.2 * np.random.rand(self.output_units_number, self.hidden_units_number) - 0.1   # 3*10  w2
        self.bias_2 = 0.2 * np.random.rand(self.output_units_number, 1) - 0.1                            # 3*1   b2

        self.set_activation(hidden_activation, hidden_activation_gradient, output_activation,       # 设置激活函数
                            output_activation_gradient)

    # Set activation function
    def set_activation(self, hidden_activation, hidden_activation_gradient, output_activation,
                       output_activation_gradient):

        self.hidden_activation = hidden_activation
        self.hidden_activation_gradient = hidden_activation_gradient
        self.output_activation = output_activation
        self.output_activation_gradient = output_activation_gradient

    # Set evaluate dataset
    def set_evaluate_dataset(self, samp_input, samp_output):
        self.eval_input = samp_input
        self.eval_output = samp_output

    # train model
    """ 
     Input: loss_function(function)     损失函数
            loss_gradient(function)     损失函数的梯度
            optimizer(function)     优化器
            learn_error(float64)   迭代截至条件
            max_iteration(int64)   最大迭代次数
    """

    def train(self, loss_function, loss_gradient, optimizer, learn_error, iteration, evaluate=False,
              **option_hyper_param):

        self.loss_function = loss_function
        self.loss_function_gradient = loss_gradient
        self.optimizer = optimizer

        train_losses = []
        eval_losses = []
        param = []
        elapsed_time = 0

        # plt.ion()
        # train_fig = plt.figure()
        # loss_plt = train_fig.add_subplot(1, 1, 1)
        # loss_plt.set_title('train_loss')
        # loss_plt.set_xlable('train_iter')
        # loss_plt.set_ylable('loss')

        for iter in range(iteration):

            last_time = time.time()

            # Back propagation and Optimizer
            loss = self.optimizer(param, option_hyper_param)

            iter_time = time.time() - last_time
            elapsed_time = elapsed_time + iter_time
            train_losses.append(loss)
            # loss_plt.plot(loss, 'b-')
            # plt.draw()

            if evaluate:
                results, eval_loss = self.evaluate(self.eval_input, self.eval_output)
                eval_losses.append(eval_loss)

            if iter % 100 == 0:
                print('train iteration:%d, train loss:%f, iter time:%f, elapsed time:%f' % (
                    iter, loss, iter_time, elapsed_time))

            if loss < learn_error:
                break

        # plt.ioff()
        # plt.show()
        if evaluate:
            return train_losses, eval_losses, results
        else:
            return train_losses

    # Forward graph configure network output
    def _forward(self, input_=np.array([])):

        if len(input_) == 0:
            input_ = self.input

        hidden_output = self.hidden_activation(self.weight_1.dot(input_) + self.bias_1)    # 隐含层输出
        network_output = self.output_activation(self.weight_2.dot(hidden_output) + self.bias_2)   # 总输出

        return hidden_output, network_output

    # Back graph configure network gradient
    def _backward(self, hidden_output, network_output):

        hidden_gradient = self.loss_function_gradient(network_output, self.output) * self.output_activation_gradient(
            network_output)
        input_gradient = self.weight_2.T.dot(hidden_gradient) * self.hidden_activation_gradient(hidden_output)

        delta_weight_2 = hidden_gradient.dot(hidden_output.T) / self.data_number
        delta_bias_2 = hidden_gradient.dot(np.ones((200, 1))) / self.data_number
        delta_weight_1 = input_gradient.dot(self.input.T) / self.data_number
        delta_bias_1 = input_gradient.dot(np.ones((200, 1))) / self.data_number

        return delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2

    # evaluate model
    def evaluate(self, input, output, ):

        hidden_output, network_output = self._forward(input)

        loss = self.loss_function(network_output, output)

        return network_output, loss

    # predict
    def predict(self, input):

        output = self._forward(input)

        return output

    ### Optimizer Functions ###
    # Batch Gradient Descent (BGD)
    def BGD_optimizer(self, param, hyper_param={'learn_rate': 0.01}):

        # Initialize variables
        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('BGD_optimizer have no "learn_rate" hyper-parameter')
            return

        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2 = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2)
        extended_variables = self.extend_variables(self.weight_1, self.bias_1, self.weight_2, self.bias_2)

        # Updata variables
        extended_delta = learn_rate * extended_gradient
        extended_variables = extended_variables - extended_delta

        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.split_weights(extended_variables)

        return loss_

    # Momentum
    def Momentum_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):

        # Initialize variables
        if len(param) == 0:
            param.append(np.zeros(1))  # last delta_weights and delta_biases

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('Momentum_optimizer have no "learn_rate" hyper-parameter')
            return
        try:
            momentum_rate = hyper_param['momentum_rate']
        except:
            print('Momentum_optimizer have no "momentum_rate" hyper-parameter')
            return
        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2 = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2)
        extended_variables = self.extend_variables(self.weight_1, self.bias_1, self.weight_2, self.bias_2)

        # Updata variables
        extended_delta = param[0]
        extended_delta = momentum_rate * extended_delta + learn_rate * extended_gradient
        param[0] = extended_delta
        extended_variables = extended_variables - extended_delta
        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.split_weights(extended_variables)
        return loss_

    # Nesterov Accelerated Gradient(NAG)
    def NAG_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):

        # Initialize variables
        if len(param) == 0:
            param.append(np.zeros(1))  # last delta_weights and delta_biases

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('NAG_optimizer have no "learn_rate" hyper-parameter')
            return
        try:
            momentum_rate = hyper_param['momentum_rate']
        except:
            print('NAG_optimizer have no "momentum_rate" hyper-parameter')
            return

        # Forward propagation
        extended_variables = self.extend_variables(self.weight_1, self.bias_1, self.weight_2, self.bias_2)
        extended_delta = param[0]
        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.split_weights(
            extended_variables - momentum_rate * extended_delta)
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2 = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2)

        # Updata variables
        extended_delta = momentum_rate * extended_delta + learn_rate * extended_gradient
        param[0] = extended_delta
        extended_variables = extended_variables - extended_delta

        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.split_weights(extended_variables)

        return loss_

    # Adagrad
    def Adagrad_optimizer(self, param, hyper_param={'learn_rate': 0.01}):

        # Initialize variables
        delta = 10e-7
        if len(param) == 0:
            param.append(np.zeros(1))  # accumulated square gradient

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('Adagrad_optimizer have no "learn_rate" hyper-parameter')
            return

        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2 = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2)
        extended_variables = self.extend_variables(self.weight_1, self.bias_1, self.weight_2, self.bias_2)

        # Updata variables
        accumulated_gradient = param[0]
        accumulated_gradient = accumulated_gradient + extended_gradient * extended_gradient
        param[0] = accumulated_gradient
        extended_delta = learn_rate / np.sqrt(accumulated_gradient + delta) * extended_gradient
        extended_variables = extended_variables - extended_delta

        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.split_weights(extended_variables)

        return loss_

    # Adadelta
    def Adadelta_optimizer(self, param, hyper_param={'decay_rate': 0.9}):   # 衰减率

        # Initialize variables
        delta = 10e-7
        if len(param) == 0:
            param.append(np.zeros(1))  # accumulated square gradient
            param.append(np.zeros(1))  # accumulated square delta

        try:
            decay_rate = hyper_param['decay_rate']
        except:
            print('Adadelta_optimizer have no "decay_rate" hyper-parameter')
            return

        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2 = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2)
        extended_variables = self.extend_variables(self.weight_1, self.bias_1, self.weight_2, self.bias_2)

        # Updata variables
        accumulated_gradient = param[0]
        accumulated_delta = param[1]
        accumulated_gradient = decay_rate * accumulated_gradient + (
                1 - decay_rate) * extended_gradient * extended_gradient
        extended_delta = np.sqrt(accumulated_delta + delta) / np.sqrt(accumulated_gradient + delta) * extended_gradient
        accumulated_delta = decay_rate * accumulated_delta + (1 - decay_rate) * extended_delta * extended_delta
        param[0] = accumulated_gradient                     ## 更新参数
        param[1] = accumulated_delta                        ## 更新参数
        extended_variables = extended_variables - extended_delta

        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.split_weights(extended_variables)

        return loss_

    # RMSProp
    def RMSProp_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay_rate': 0.9}):

        # Initialize variables
        delta = 10e-6
        if len(param) == 0:
            param.append(np.zeros(1))  # accumulated square gradient

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('RMSProp_optimizer have no "learn_rate" hyper-parameter')
            return
        try:
            decay_rate = hyper_param['decay_rate']
        except:
            print('RMSProp_optimizer have no "decay_rate" hyper-parameter')
            return

        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2 = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2)
        extended_variables = self.extend_variables(self.weight_1, self.bias_1, self.weight_2, self.bias_2)

        # Updata variables
        accumulated_gradient = param[0]
        accumulated_gradient = decay_rate * accumulated_gradient + (
                1 - decay_rate) * extended_gradient * extended_gradient
        param[0] = accumulated_gradient
        extended_delta = learn_rate / np.sqrt(accumulated_gradient + delta) * extended_gradient
        extended_variables = extended_variables - extended_delta

        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.split_weights(extended_variables)

        return loss_

    # RMSProp_with_Nesterov
    def RMSProp_Nesterov_optimizer(self, param,
                                   hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9, 'decay_rate': 0.9}):

        # Initialize variables
        delta = 10e-6
        if len(param) == 0:
            param.append(np.zeros(1))  # last delta_weights and delta_biases
            param.append(np.zeros(1))  # accumulated square gradient

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('RMSProp_Nesterov_optimizer have no "learn_rate" hyper-parameter')
            return
        try:
            decay_rate = hyper_param['decay_rate']
        except:
            print('RMSProp_Nesterov_optimizer have no "decay_rate" hyper-parameter')
            return
        try:
            momentum_rate = hyper_param['momentum_rate']
        except:
            print('RMSProp_Nesterov_optimizer have no "momentum_rate" hyper-parameter')
            return

        # Forward propagation
        extended_variables = self.extend_variables(self.weight_1, self.bias_1, self.weight_2, self.bias_2)
        extended_delta = param[0]
        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.split_weights(
            extended_variables - momentum_rate * extended_delta)
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2 = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2)
        extended_variables = self.extend_variables(self.weight_1, self.bias_1, self.weight_2, self.bias_2)

        # Updata variables
        accumulated_gradient = param[1]
        accumulated_gradient = decay_rate * accumulated_gradient + (
                1 - decay_rate) * extended_gradient * extended_gradient
        param[1] = accumulated_gradient
        extended_delta = learn_rate / np.sqrt(accumulated_gradient + delta) * extended_gradient
        extended_variables = extended_variables - extended_delta

        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.split_weights(extended_variables)

        return loss_

    # Adam
    def Adam_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay1_rate': 0.9, 'decay2_rate': 0.999}):

        # Initialize variables
        delta = 10e-8
        if len(param) == 0:
            param.append(np.zeros(1))  # accumulated gradient
            param.append(np.zeros(1))  # accumulated square gradient
            param.append(0)  # train steps

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('Adam_optimizer have no "learn_rate" hyper-parameter')
            return
        try:
            decay1_rate = hyper_param['decay1_rate']
        except:
            print('Adam_optimizer have no "decay1_rate" hyper-parameter')
            return
        try:
            decay2_rate = hyper_param['decay2_rate']
        except:
            print('Adam_optimizer have no "decay2_rate" hyper-parameter')
            return

        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2 = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_weight_1, delta_bias_1, delta_weight_2, delta_bias_2)
        extended_variables = self.extend_variables(self.weight_1, self.bias_1, self.weight_2, self.bias_2)

        # Updata variables
        accumulated_gradient = param[0]
        accumulated_square_gradient = param[1]
        step = param[2] + 1
        accumulated_gradient = decay1_rate * accumulated_gradient + (1 - decay1_rate) * extended_gradient
        accumulated_square_gradient = decay2_rate * accumulated_square_gradient + (
                1 - decay2_rate) * extended_gradient * extended_gradient
        param[0] = accumulated_gradient
        param[1] = accumulated_square_gradient
        param[2] = step + 1
        extended_moment1 = accumulated_gradient / (1 - np.power(decay1_rate, step))
        extended_moment2 = accumulated_square_gradient / (1 - np.power(decay2_rate, step))
        extended_delta = learn_rate * extended_moment1 / (np.sqrt(extended_moment2) + delta)
        extended_variables = extended_variables - extended_delta

        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = self.split_weights(extended_variables)

        return loss_

    ### Activation Functions ###
    # sigmoid
    def sigmoid_activation(self, input_):

        output_ = 1 / (1 + np.exp(-input_))

        return output_

    def sigmoid_gradient(self, input_):

        output_ = input_ * (1 - input_)

        return output_

    # tanh (Bipolar sigmoid)
    def tanh_activation(self, input_):

        output_ = (1 - np.exp(-input_)) / (1 + np.exp(-input_))

        return output_

    def tanh_gradient(self, input_):

        output_ = 0.5 * (1 - input_ * input_)

        return output_

    # ReLU
    def ReLU_activation(self, input_):

        output_ = np.where(input_ < 0, 0, input_)

        return output_

    def ReLU_gradient(self, input_):

        output_ = np.where(input_ > 0, 1, 0)

        return output_

    # Softmax
    def softmax_activation(self, input_):

        output_ = np.exp(input_ - input_.max(axis=0)) / np.sum(np.exp(input_ - input_.max(axis=0)), axis=0)

        return output_

    def softmax_gradient(self, input_):

        output_ = input_ * (self.output - input_)

        return output_

    # None Activation
    def none_activation(self, input_):

        output_ = input_

        return output_

    def none_gradient(self, input_):

        output_ = 1

        return output_

    ### Loss Functions ###
    # output_: Network predict output   output: dataset output
    # L2_loss (RMSE)
    def L2_loss(self, output_, output):

        loss = np.sum(0.5 * np.power(output_ - output, 2))

        return loss

    def L2_loss_gradient(self, output_, output):

        loss_gradient = output_ - output

        return loss_gradient

    # cross_entropy loss

    def cross_entropy_loss(self, output_, output):

        loss = -np.sum(np.log(output_) * output)

        return loss

    def cross_entropy_gradient(self, output_, output):

        loss_gradient = -output / output_

        return loss_gradient

    # sigmoid_cross_entropy loss

    # softmax cross_entropy loss

    ### utilities ###
    def extend_variables(self, weight_1_, bias_1_, weight_2_, bias_2_):

        variables = np.concatenate(
            (weight_1_.reshape(-1), bias_1_.reshape(-1), weight_2_.reshape(-1), bias_2_.reshape(-1)))

        return variables

    def split_weights(self, variables):

        weight_1_mark = self.input_units_number * self.hidden_units_number
        bias_1_mark = weight_1_mark + self.hidden_units_number
        weight_2_mark = bias_1_mark + self.hidden_units_number * self.output_units_number
        weight_1 = variables[: weight_1_mark].copy()
        weight_1 = weight_1.reshape(self.hidden_units_number, self.input_units_number)
        bias_1 = variables[weight_1_mark: bias_1_mark].copy()
        bias_1 = bias_1.reshape(self.hidden_units_number, 1)
        weight_2 = variables[bias_1_mark: weight_2_mark].copy()
        weight_2 = weight_2.reshape(self.output_units_number, self.hidden_units_number)
        bias_2 = variables[weight_2_mark:].copy()
        bias_2 = bias_2.reshape(self.output_units_number, 1)

        return weight_1, bias_1, weight_2, bias_2

    def save_model(self, path='model'):
        np.save(path, [self.weight_1, self.bias_1, self.weight_2, self.bias_2])

    def load_model(self, path):
        parameters = np.load(path,allow_pickle=True)

        self.weight_1 = parameters[0]
        self.bias_1 = parameters[1]
        self.weight_2 = parameters[2]
        self.bias_2 = parameters[3]

        self.input_units_number = self.weight_1.shape[1]
        self.hidden_units_number = self.weight_1.shape[0]
        self.output_units_number = self.weight_2.shape[0]
