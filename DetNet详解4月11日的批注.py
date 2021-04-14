#!/usr/bin/env python

#这里是测试一下有没有被修改11111，github第一次使用
#这里是测试一下有没有被修改11111，github第二次使用
import datetime

import tensorflow as tf
import numpy as np
import time as tm
import math
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import os
# start here
"""
Parameters
K - size of x
N - size of y
snrdb_low - the lower bound of noise db used during training
snr_high - the higher bound of noise db used during training
L - number of layers in DetNet
v_size = size of auxiliary variable at each layer     每层辅助变量的大小
hl_size - size of hidden layer at each DetNet layer (the dimention the layers input are increased to）每个detnet网络层的隐藏层大小（图层输入的尺寸增加到）
startingLearningRate - the initial step size of the gradient descent algorithm 梯度下降算法的初始步长
decay_factor & decay_step_size - each decay_step_size steps the learning rate decay by decay_factor   每一个decay_step_size步进学习速率衰减的decay_factor
train_iter - number of train iterations    训练迭代次数
train_batch_size - batch size during training phase  训练批次大小
test_iter - number of test iterations    测试迭代，迭代的数目
test_batch_size  - batch size during testing phase    测试批次大小，测试期间批次的大小
LOG_LOSS - equal 1 if loss of each layer should be sumed in proportion to the layer depth, otherwise all losses have the same weight   如果每一层的损失应按层深的比例计算，则等于1，否则所有损失的权重相同
res_alpha- the proportion of the previuos layer output to be added to the current layers output (view ResNet article) 前边的层加到现在层的输出比例 （看ResNet文章）
snrdb_low_test & snrdb_high_test & num_snr - when testing, num_snr different SNR values will be tested, uniformly spread between snrdb_low_test and snrdb_high_test   当测试的时候num_snr不同于SNR值的时候将被测试，在信噪比分贝最低测试和信噪比分贝最高测试之间均匀扩散
"""
sess = tf.InteractiveSession()  # 能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的
tic = tm.time()
global_tic = tm.time()
# parameters
K = 40    #K是x的大小，这里设为40
N = 80    #N是y的大小，这里设为80
L = 6  # 层数
snrdb_low = 7.0  # 最低信噪比，这里设为了7.0
snrdb_high = 14.0
snr_low = 10.0 ** (snrdb_low / 10.0)  # 最小的信噪比化为十进制
snr_high = 10.0 ** (snrdb_high / 10.0)   #
v_size = 2 * K  # 每层辅助变量的大小 ,在最前边的定义里v_size已经被定义为每层辅助变量的大小。
hl_size = 8 * K  # 隐藏层的大小
startingLearningRate = 0.0001  # 初始学习率
decay_factor = 0.97   #衰减因子
decay_step_size = 1000   #衰减步进学习率的大小
train_iter = 5000  # 训练迭代次数
train_batch_size = 5000  # 批量大小
test_iter = 40     #测试迭代
test_batch_size = 1000   #测试批次大小
LOG_LOSS = 1  # 残差网络-每一层的损失应按层深的比例计算
res_alpha = 0.9  # 要添加到当前层输出的之前层输出的 比例，或者说成前边的层加到现在层的输出比例 
num_snr = 6  
snrdb_low_test = 8.0    #信噪比分贝最低测试
snrdb_high_test = 13.0    #信噪比分贝最高测试

"""Data generation for train and test phases
In this example, both functions are the same.
This duplication is in order to easily allow testing cases where the test is over different distributions of data than in the training phase.
e.g. training over gaussian i.i.d. channels and testing over a specific constant channel.
currently both test and train are over i.i.d gaussian channel.
"""
nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
print("K=", K, "N=", N, "L=", L, "train_iter=", train_iter, "now=", nowTime)


def generate_data_iid_test(B, K, N, snr_low, snr_high):  # B为train_batch_size批次数
    H_ = np.random.randn(B, N, K)  # 从标准正态分布中随机取值 #B组N*K维随机值
    W_ = np.zeros([B, K, K])
    x_ = np.sign(np.random.rand(B, K) - 0.5)  # -0.5到0.5区间的随机值
   # The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
    y_ = np.zeros([B, N])
    w = np.random.randn(B, N)
    Hy_ = x_ * 0  # 零矩阵
    HH_ = np.zeros([B, K, K])
    SNR_ = np.zeros([B])  # B*1组
    for i in range(B):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = H_[i, :, :]
        tmp_snr = (H.T.dot(H)).trace() / K  # H转置乘以H  求对角元素的和再除以K
        H = H / np.sqrt(tmp_snr) * np.sqrt(SNR)  # 压缩了一下
        H_[i, :, :] = H  # 信道矩阵
        y_[i, :] = (H.dot(x_[i, :]) + w[i, :])  # y_=Hx_+w
        Hy_[i, :] = H.T.dot(y_[i, :])  # Hy_=H转置*y_
        HH_[i, :, :] = H.T.dot(H_[i, :, :])  # HH_=H转置*H_
        SNR_[i] = SNR
    return y_, H_, Hy_, HH_, x_, SNR_


def generate_data_train(B, K, N, snr_low, snr_high):  # 和上一段完全一样，可以合并
    H_ = np.random.randn(B, N, K)
    x_ = np.sign(np.random.rand(B, K) - 0.5)
    y_ = np.zeros([B, N])
    w = np.random.randn(B, N)
    Hy_ = x_ * 0
    HH_ = np.zeros([B, K, K])
    SNR_ = np.zeros([B])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = H_[i, :, :]
        tmp_snr = (H.T.dot(H)).trace() / K
        H = H / np.sqrt(tmp_snr) * np.sqrt(SNR)
        H_[i, :, :] = H
        y_[i, :] = (H.dot(x_[i, :]) + w[i, :])
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_, H_, Hy_, HH_, x_, SNR_


def piecewise_linear_soft_sign(x):
    t = tf.Variable(0.1)
    y = -1 + tf.nn.relu(x + t) / (tf.abs(t) + 0.00001) - \
        tf.nn.relu(x - t) / (tf.abs(t) + 0.00001)
    return y  # max(0,)relu函数是小于零都为零，大于零的数不变;abs为绝对值


def affine_layer(x, input_size, output_size, Layer_num):
    # 随机取input*output维的标准差为0.01的矩阵
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    y = tf.matmul(x, W) + w
    return y


def relu_layer(x, input_size, output_size, Layer_num):
    y = tf.nn.relu(affine_layer(
        x, input_size, output_size, Layer_num))  # relu只保留正值
    return y


def sign_layer(x, input_size, output_size, Layer_num):
    y = piecewise_linear_soft_sign(affine_layer(
        x, input_size, output_size, Layer_num))
    return y


# tensorflow placeholders, 为训练和测试网络而给模型的输入
# tf.placeholder（数值类型，一行K列，名字）#H转置*y
HY = tf.placeholder(tf.float32, shape=[None, K])
X = tf.placeholder(tf.float32, shape=[None, K])
HH = tf.placeholder(tf.float32, shape=[None, K, K])

batch_size = tf.shape(HY)[0]  # shape（）括号里的形状用矩阵表示

# MMSE算法？？？？？？
# tf.expand_dim:维度增加一维，可以使用tf.expand_dims(input, dim, name=None)函数
X_LS = tf.matmul(tf.expand_dims(HY, 1), tf.matrix_inverse(HH))  # X=HH的逆*y？
X_LS = tf.squeeze(X_LS, 1)  # 删除所有大小是1的维度
loss_LS = tf.reduce_mean(tf.square(X - X_LS))  # 损失函数军方误差
ber_LS = tf.reduce_mean(tf.cast(tf.not_equal(
    X, tf.sign(X_LS)), tf.float32))  # MMSE算法
# tf.reduce_mean函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。

S = []  # 对输入x的评估
S.append(tf.zeros([batch_size, K]))  # append函数  S的末尾加上（）的内容#1*K=40
V = []  # 辅助变量
V.append(tf.zeros([batch_size, v_size]))  # 1*K
LOSS = []  # 每一层的代价函数
LOSS.append(tf.zeros([]))
BER = []
BER.append(tf.zeros([]))

# The architecture of DetNet
for i in range(1, L):  # L为神经网络层数
    # matmul矩阵相乘，multiply是对应元素相乘
    temp1 = tf.matmul(tf.expand_dims(S[-1], 1), HH)
    temp1 = tf.squeeze(temp1, 1)  # temp1中是1的全删除
    # concat把五个（用户数40）40*1矩阵拼在一起,200*1的矩阵
    Z = tf.concat([HY, S[-1], temp1, V[-1]], 1)
    # 320=200+3*40.#就是论文里面的Zk #max(0,)relu函数是小于零都为零，大于零的数不变;abs为绝对值
    ZZ = relu_layer(Z, 3 * K + v_size, hl_size, 'relu' + str(i))
    S.append(sign_layer(ZZ, hl_size, K, 'sign' + str(i)))  # 120行定义的
    S[i] = (1 - res_alpha) * S[i] + res_alpha * \
        S[i - 1]  # 0.1*S【i】+0.9*S【i-1】#S是各层对输入的评估
    V.append(affine_layer(ZZ, hl_size, v_size, 'aff' + str(i)))
    V[i] = (1 - res_alpha) * V[i] + res_alpha * V[i - 1]
    if LOG_LOSS == 1:  # 残差网络-每一层的损失应按层深的比例计算
        LOSS.append(np.log(i) * tf.reduce_mean(
            tf.reduce_mean(tf.square(X - S[-1]), 1)))
        # tf.reduce_mean(tf.square(X - S[-1]), 1) / tf.reduce_mean(tf.square(X - X_LS), 1)))#为什么有个除法？
    else:
        LOSS.append(tf.reduce_mean(tf.reduce_mean(
            tf.square(X - S[-1]), 1) / tf.reduce_mean(tf.square(X - X_LS), 1)))
    BER.append(tf.reduce_mean(
        tf.cast(tf.not_equal(X, tf.sign(S[-1])), tf.float32)))
# The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
TOTAL_LOSS = tf.add_n(LOSS)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor,
                                           staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(TOTAL_LOSS)

# 变量运行前必须做初始化操作
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Training DetNet
for i in range(train_iter):  # num of train iter
    batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = generate_data_train(
        train_batch_size, K, N, snr_low, snr_high)
    train_step.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X})
    if i % 100 == 0:  # i能否被100整除
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = generate_data_iid_test(
            train_batch_size, K, N, snr_low, snr_high)
        results = sess.run([loss_LS, LOSS[L - 1], ber_LS, BER[L - 1]],
                           {HY: batch_HY, HH: batch_HH, X: batch_X})
        print_string = [i] + results
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              "|", ' '.join('%s' % x for x in print_string))
       # print(ber_LS)

# Testing the trained model
snrdb_list = np.linspace(snrdb_low_test, snrdb_high_test, num_snr)
snr_list = 10.0 ** (snrdb_list / 10.0)
bers = np.zeros((1, num_snr))
berss = np.zeros((1, num_snr))
times = np.zeros((1, num_snr))
tmp_bers = np.zeros((1, test_iter))
tmp_berss = np.zeros((1, test_iter))
tmp_times = np.zeros((1, test_iter))
for j in range(num_snr):
    print(datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'), "|", 'snr=', snrdb_list[j])
    for jj in range(test_iter):
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = generate_data_iid_test(test_batch_size, K, N, snr_list[j],
                                                                                     snr_list[j])
        tic = tm.time()
        tmp_bers[:, jj] = np.array(
            sess.run(BER[L - 1], {HY: batch_HY, HH: batch_HH, X: batch_X}))
        tmp_berss[:, jj] = np.array(
            sess.run(ber_LS, {HY: batch_HY, HH: batch_HH, X: batch_X}))
        toc = tm.time()
        tmp_times[0][jj] = toc - tic

    bers[0][j] = np.mean(tmp_bers, 1)
    berss[0][j] = np.mean(tmp_berss, 1)
    times[0][j] = np.mean(tmp_times[0]) / test_batch_size

print('snrdb_list')
print(snrdb_list)

print('bers')
print(bers)
print('berss')
print(berss)
print('times')
print(times)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.txt")

global_toc = tm.time()

with open(path, 'a+') as f:

    for i in range(len(times)):
        f.write(f"{snr_list[i]}     {bers[i]}    {berss[i]}    {times[i]} \n",)


print("total coat time: ", global_toc-global_tic)
