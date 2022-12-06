import random
import numpy as np
from tqdm import tqdm 


class RBM_hid_linear(object):
    '''
    定义一个RBM_hid_linear网络类（与RBM的唯一区别是前向输出不使用激活函数）
    '''
    def __init__(self, n_visible=500, n_hidden=200,
                 momentum=0.5, learning_rate=0.1, max_epoch=50, batch_size=256, penalty=0,
                 weight_v=None, bias_v=None, bias_h=None):
        '''
        RBM网络初始化
        使用动量的随机梯度下降法训练网络
        args:
            n_visible:可见层节点个数
            n_hidden：隐藏层节点个数
            momentum:动量参数 一般取值0.5,0.9,0.99  当取值0.9时，对应着最大速度1/(1-0.9)倍于梯度下降算法
            learning_rate：学习率
            max_epoch：最大训练轮数
            batch_size：小批量大小
            penalty：规范化 权重衰减系数  一般设置为1e-4  默认不使用
            weight_v：可见层与隐藏层权重初始化参数，默认[n_hidden x n_visible]
            bias_v:可见层偏置初始化 默认[n_visible]
            bias_h:隐藏层偏置初始化 默认[n_hidden]
        '''
        # 私有变量初始化
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.penalty = penalty
        self.momentum = momentum
        self.learning_rate = learning_rate

        if weight_v is None:
            self.weight_v = np.random.random((self.n_hidden, self.n_visible)) * 0.1  
        else:
            self.weight_v = weight_v
        if bias_v is None:
            self.bias_v = np.zeros(self.n_visible) 
        else:
            self.bias_v = bias_v
        if bias_h is None:
            self.bias_h = np.zeros(self.n_hidden) 
        else:
            self.bias_h = bias_h
        

    def relu(self, z):
        '''
        定义激活函数
        args:
            z：传入元素or list 、nparray
        '''

        return (z + np.abs(z)) / 2.0

    def forward(self, inpt_v):
        '''
        正向传播
        args:
            inpt : 输入数据v(可见层) 大小为batch_size * n_visible(可见层节点数)
        '''
        z = np.dot(inpt_v, self.weight_v.T) + self.bias_h  # 计算加权和 p(h|v)
        return z

    def backward(self, inpt_h):
        '''
        反向重构
        args:
            inpt : 输入数据(隐藏层) 大小为batch_size x n_hidden 
        '''
        z = np.dot(inpt_h, self.weight_v) + self.bias_v  # 计算加权和
        return self.relu(z)

    def batch(self):
        '''
        把数据集打乱，按照batch_size分组
        '''
        m, n = self.input_v.shape
        per = list(range(m))
        random.shuffle(per)  
        per = [per[k:k + self.batch_size] for k in range(0, m, self.batch_size)] 
        batch_data_v = []
        batch_data_c = []
        batch_data_r = []
        for group in per:
            batch_data_v.append(self.input_v[group])
        return batch_data_v

    def fit(self, input_v):
        '''
        开始训练网络
        args:
            input_v:输入数据集
        '''
        self.input_v = input_v
        WincV = np.zeros_like(self.weight_v)
        binc = np.zeros_like(self.bias_v)
        cinc = np.zeros_like(self.bias_h)
    
        pbar=tqdm(range(self.max_epoch))
        for epoch in pbar:
            batch_data_v = self.batch()
            num_batchs = len(batch_data_v)
            err_sum = 0.0
            m, _ = self.input_v.shape
            self.penalty = (1 - 0.9 * epoch / self.max_epoch) * self.penalty  # 随着迭代次数增加 penalty减小
            for v0 in batch_data_v:
                # 前向传播  计算h0
                h0 = self.forward(v0)
                h0_states = np.zeros_like(h0)
                h0_states[h0 > np.random.random(h0.shape)] = 1
                # 反向重构  计算v1
                v1 = self.backward(h0_states)
                # 前向传播 计算h1
                h1 = self.forward(v1)
                h1_states = np.zeros_like(h1)
                h1_states[h1 > np.random.random(h1.shape)] = 1
                
                '''梯度下降'''
                # 计算batch_size个样本的梯度估计值
                dWV = np.dot(h0_states.T, v0) - np.dot(h1_states.T, v1)  # ΔW  CD-k 算法中近似的梯度函数
                db = np.mean(v0 - v1, axis=0).T    
                dc = np.mean(h0 - h1, axis=0).T   
                
                # 计算速度更新
                WincV = self.momentum * WincV + self.learning_rate * (dWV - self.penalty * self.weight_v) / self.batch_size
                binc = self.momentum * binc + self.learning_rate * db / self.batch_size  # 可见层偏置
                cinc = self.momentum * cinc + self.learning_rate * dc / self.batch_size  # 隐藏层偏置

                # update weight
                self.weight_v = self.weight_v + WincV
                self.bias_v = self.bias_v + binc
                self.bias_h = self.bias_h + cinc
                
                # 计算误差
                err_sum = err_sum + np.mean(np.sum((v0 - v1) ** 2, axis=1))
            err_sum = err_sum / num_batchs
            pbar.set_postfix({'loss': err_sum})
            
    def predict(self, input_v):
        h = self.forward(input_v)
        v = self.backward(h)
        return v
    