import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from model import *
from processing.data_processing import *
from processing.input_processing import *


"""1. Data processing"""
# homeTown and currentCity data
data_com_city = pd.read_csv(r'../data/data_com_city.csv')
data_com_city["localTime"] = pd.to_datetime(data_com_city["localTime"], format='%Y-%m-%d %H:%M:%S')

data_currentCity = data_com_city[data_com_city['location'] == 'currentCity'].copy()
data_currentCity.reset_index(drop=True, inplace=True)

data_homeTown = data_com_city[data_com_city['location'] == 'homeTown'].copy()
data_homeTown.reset_index(drop=True, inplace=True)

## 兴趣迁移模块
edge_df = get_poi_category_edge(data_com_city)           # 构造POI-venueCategory 边集
embeddings = get_edge_embeddings()                       # embedding字典
data_transform = get_venue_transform(data_currentCity, data_homeTown, data_com_city, embeddings)  #兴趣点转换

## 时间聚类模块
data_time, time_point = get_time_cluster(data_transform, n_clus=12)  # 时间聚类

# 划分训练集，并特征编码
all_data, train_data, test_data = get_split_dataset(data_time, split_size=0.8)


"""2. Input processing"""
# 地理邻近性矩阵
geo_matrix = get_GeoProx_weight(train_data)
# # 标准化
# zscore_scaler = StandardScaler()
# weight_g[weight_g != 0] = zscore_scaler.fit_transform(weight_g[weight_g != 0].reshape(-1, 1)).reshape(1, -1)[0]

# 生成可见层输入
DAE_input_v, train_data_preference = get_visi_input(train_data)
# train_data_preference.to_csv(r'./data/temp/Tokyo_train_data_preference.csv', index=False)   # 保存文件

# 时间条件层输入
DAE_input_c = get_condition_input(DAE_input_v)

# 生成社交正则项输入
path_friendship = r'../data/friendship.txt'
friendship_ind = get_friendship(train_data_preference, path_friendship)  # 处理社交关系数据

get_LBSN2vec_input(train_data_preference, friendship_ind)    # 生成LBSN2vec输入数据

# matlab run LBSN2vec 挖掘潜在好友关系
# 1. Compile ./matlab/learn_LBSN2Vec_embedding.c using mex: 
#    mex CFLAGS='$CFLAGS -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result' learn_LBSN2Vec_embedding.c
# 2. Run ./matlab/experiment_LBSN2Vec.m
# 3. Run ./matlab/friendship_Prediction.m

friendship_ind_exp = get_friendship_expand(friendship_ind)  # 补充挖掘的社交关系数据
DAE_input_r = get_regularization_input(train_data_preference, friendship_ind_exp, DAE_input_v)  # 生成社交正则项输入


"""3. Pre-training"""
target_time = 1  # 目标时间段ID

input_v = DAE_input_v[target_time].toarray()
input_c = [mat.toarray() for mat in DAE_input_c[target_time]]
input_r = DAE_input_r[target_time].toarray()
weight_g = geo_matrix  

num_venue = input_v.shape[1]
n_condition = len(DAE_input_c[target_time])

# 预训练模型超参数
batch_size = 256  # 全局
epoch_list = [50, 100, 100]  # 每个基模型的训练批次
learning_rate_list = [0.1, 0.005, 0.005]  # 每个基模型的学习率
num_layer_unit_list = [num_venue, 3000, 1000, 500]  # DAE每层单元数量
num_RBM = len(epoch_list) - 2

hid_vis = {}
# 预训练第一个基模型GTR_RBM
gtr_rbm = GTR_RBM(n_visible=num_layer_unit_list[0],
                  n_hidden=num_layer_unit_list[1],
                  n_condition=n_condition,
                  max_epoch=epoch_list[0],
                  batch_size=batch_size, 
                  learning_rate=learning_rate_list[0])

gtr_rbm.fit(input_v, input_c, input_r, weight_g)
hid_vis['gtr_rbm'] = gtr_rbm.forward(input_v, input_c)  # 缓存隐层激活概率

# 依次预训练基模型RBM
rbm_list = {}
for n in range(num_RBM):
    rbm_list[n] = RBM(n_visible=num_layer_unit_list[n+1],
                      n_hidden=num_layer_unit_list[n+2],
                      max_epoch=epoch_list[n+1],
                      batch_size=batch_size, 
                      learning_rate=learning_rate_list[n+1])
    if n == 0:
        rbm_list[n].fit(hid_vis['gtr_rbm'])
        hid_vis[f'rbm{n}'] = rbm_list[n].forward(hid_vis['gtr_rbm'])
    else:
        rbm_list[n].fit(hid_vis[f'rbm{n-1}'])
        hid_vis[f'rbm{n}'] = rbm_list[n].forward(hid_vis[f'rbm{n-1}'])
        
# 预训练最后一个基模型RBM_hid_linear
rbm_hid_linear = RBM_hid_linear(n_visible=num_layer_unit_list[-2],
                                n_hidden=num_layer_unit_list[-1],
                                max_epoch=epoch_list[-1],
                                batch_size=batch_size, 
                                learning_rate=learning_rate_list[-1])

rbm_hid_linear.fit(hid_vis[f'rbm{n}'])


"""4. Fine-tuning"""
base_model = [gtr_rbm, rbm_list, rbm_hid_linear]
DAE_input = [input_v, input_c, input_r, weight_g]

dae = DAE(base_model, DAE_input, learning_rate=0.05, max_epoch=200, batch_size=256,)

ouput = dae.train_predict()