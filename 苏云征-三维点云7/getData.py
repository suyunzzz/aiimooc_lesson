'''
@Description:  生成待分类的数据
@Author: Su Yunzheng
@Date: 2020-06-17 17:02:11
@LastEditTime: 2020-06-17 23:09:54
@LastEditors: Su Yunzheng
'''


# generate dataset to classify
import numpy as np

def gen_classify_data(numbers):
    sample_input = (np.random.rand(2, numbers) - 0.5) * 4  # 生成[-2,2)的200对点
    sample_output = np.array([[], [], []])

    for i in range(numbers):
        sample = sample_input[:, i]
        x = sample[0]
        y = sample[1]

        if ((x > -1) & (x < 1)) == 1:
            if ((y > x / 2 + 1 / 2) & (y < 1)) == 1:
                sample_output = np.append(sample_output, np.array([[0], [1], [0]]), axis=1)  # 三角形  1
            elif ((y < -0.5) & (y > -1.5)) == 1:
                sample_output = np.append(sample_output, np.array([[0], [0], [1]]), axis=1)   # 矩形  2
            else:
                sample_output = np.append(sample_output, np.array([[1], [0], [0]]), axis=1)   # else   0
        else:
            sample_output = np.append(sample_output, np.array([[1], [0], [0]]), axis=1)   # else
    print("sample_input:{},sample_output:{}".format(sample_input.shape,sample_output.shape))
    # print("sample_output:\n{},sample_input\n:{}".format(sample_output,sample_input))
    return sample_input, sample_output


if __name__ == "__main__":
    X,y=gen_classify_data(200)
    y=np.argmax(y,axis=0)
    print("y :\n{}".format(y))
    print("y shape :\n{}".format(y.shape))