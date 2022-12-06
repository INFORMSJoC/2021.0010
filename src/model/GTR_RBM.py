import random
import numpy as np
from tqdm import tqdm 
from tqdm.contrib import tzip

class GTR_RBM(object):
    '''
    定义一个GTR_RBM网络类
    '''
    def __init__(self, n_visible=500, n_hidden=200, n_condition=1,
                 momentum=0.5, learning_rate=0.1, max_epoch=50, batch_size=256, penalty=0, re_coef=0.01,
                 weight_v=None, weight_c=None,
                 bias_v=None, bias_h=None):
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
            re_coef: 社交正则系数
            weight_v：可见层与隐藏层权重初始化参数，默认[n_hidden x n_visible]
            weight_c：条件层权重初始化参数
            bias_v:可见层偏置初始化 默认[n_visible]
            bias_h:隐藏层偏置初始化 默认[n_hidden]
        '''
        # 私有变量初始化
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_condition = n_condition
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.penalty = penalty
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.re_coef = re_coef

        if weight_v is None:
            self.weight_v = np.random.random((self.n_hidden, self.n_visible)) * 0.1  
        else:
            self.weight_v = weight_v
        
        if weight_c is None:
            # weight_c = [weight_vc, weight_hc] * n_condition
            self.weight_c = [[np.random.random((self.n_visible, self.n_visible)) * 0.1, 
                              np.random.random((self.n_hidden, self.n_visible)) * 0.1]] * self.n_condition
        else:
            self.weight_c = weight_c
            
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

    def forward(self, inpt_v, inpt_c):
        '''
        正向传播
        args:
            inpt : 输入数据v(可见层) 大小为batch_size * n_visible(可见层节点数)
        '''
        z = np.dot(inpt_v, self.weight_v.T) +  np.sum([np.dot(inpt_c[n], self.weight_c[n][1].T) for n in range(self.n_condition)], axis=0) + self.bias_h 
        # z = np.dot(inpt_v, self.weight_v.T) + self.bias_h  # 计算加权和 p(h|v)
        return self.relu(z)

    def backward(self, inpt_h, inpt_v, inpt_c, weight_g):
        '''
        反向重构
        args:
            inpt : 输入数据(隐藏层) 大小为batch_size x n_hidden 
        '''
        z = np.dot(inpt_h, self.weight_v) + np.sum([np.dot(inpt_c[n], self.weight_c[n][0].T) for n in range(self.n_condition)], axis=0) + np.dot(inpt_v, weight_g) + self.bias_v
        # z = np.dot(inpt_h, self.weight_v) + self.bias_v  # 计算加权和
        return z 

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
            batch_data_c.append([self.input_c[n][group] for n in range(self.n_condition)])
            batch_data_r.append(self.input_r[group])
            
        return batch_data_v, batch_data_c, batch_data_r

    def fit(self, input_v, input_c, input_r, weight_g):
        '''
        开始训练网络
        args:
            input_v:输入数据集
        '''
        self.input_v = input_v
        self.input_c = input_c
        self.input_r = input_r
        self.weight_g = weight_g  # 地理邻近性矩阵 不用更新
        
        WincV = np.zeros_like(self.weight_v)
        WincC = [[np.zeros_like(self.weight_c[n][0]),
                  np.zeros_like(self.weight_c[n][1])] for n in range(self.n_condition)]
        binc = np.zeros_like(self.bias_v)
        cinc = np.zeros_like(self.bias_h)
    
        pbar=tqdm(range(self.max_epoch))
        for epoch in pbar:
            batch_data_v, batch_data_c, batch_data_r = self.batch()
            num_batchs = len(batch_data_v)
            err_sum = 0.0
            m, _ = self.input_v.shape
            self.penalty = (1 - 0.9 * epoch / self.max_epoch) * self.penalty  # 随着迭代次数增加 penalty减小
            for v0, c0, r in tzip(batch_data_v, batch_data_c, batch_data_r, leave=False):
                ## 网络计算过程
                v0_mask = np.zeros_like(v0)  
                v0_mask[v0 != 0] = 1
                # 前向传播  计算h0
                h0 = self.forward(v0, c0)
                h0_states = np.zeros_like(h0)
                h0_states[h0 > np.random.random(h0.shape)] = 1
                # 反向重构  计算v1
                v1 = self.backward(h0_states, v0, c0, self.weight_g)
                v1 = np.multiply(v0_mask, v1)  # 忽略缺失值
                # 前向传播 计算h1
                h1 = self.forward(v1, c0)
                h1_states = np.zeros_like(h1)
                h1_states[h1 > np.random.random(h1.shape)] = 1
                
                '''梯度下降'''
                # 计算batch_size个样本的梯度估计值
                dWV = np.dot(h0_states.T, v0) - np.dot(h1_states.T, v1)  # ΔW  CD-k 算法中近似的梯度函数
                dWC = [[np.dot(v0.T, c0[n]) - np.dot(v1.T, c0[n]),
                        np.dot(h0_states.T, c0[n]) - np.dot(h1_states.T, c0[n])] for n in range(self.n_condition)]
                db = np.mean(v0 - v1, axis=0).T    
                dc = np.mean(h0 - h1, axis=0).T   

                # # # 社交正则
                dwf = np.dot((h0_states.T - h1_states.T), r)
                # dwf = np.dot(h0_states.T, r) - np.dot(h1_states.T, v1)
                dWV = dWV + self.re_coef * dwf
                # 计算速度更新
                WincV = self.momentum * WincV + self.learning_rate * (dWV - self.penalty * self.weight_v) / self.batch_size
                WincC = [[self.momentum * WincC[n][0] + self.learning_rate * (dWC[n][0] - self.penalty * self.weight_c[n][0]) / self.batch_size,
                         self.momentum * WincC[n][1] + self.learning_rate * (dWC[n][1] - self.penalty * self.weight_c[n][1]) / self.batch_size]
                         for n in range(self.n_condition)]

                binc = self.momentum * binc + self.learning_rate * db / self.batch_size  # 可见层偏置
                cinc = self.momentum * cinc + self.learning_rate * dc / self.batch_size  # 隐藏层偏置

                # update weight
                self.weight_v = self.weight_v + WincV
                self.weight_c = [[self.weight_c[n][0] + WincC[n][0], self.weight_c[n][1] + WincC[n][1]]
                                for n in range(self.n_condition)]
                self.bias_v = self.bias_v + binc
                self.bias_h = self.bias_h + cinc
                
                # 计算误差
                err_sum = err_sum + np.mean(np.sum((v0 - v1) ** 2, axis=1))
            err_sum = err_sum / num_batchs
            pbar.set_postfix({'loss': err_sum})
            
    def predict(self, input_v, input_c):
        h = self.forward(input_v, input_c)
        v = self.backward(h, input_v, input_c, self.weight_g)
        return v
    