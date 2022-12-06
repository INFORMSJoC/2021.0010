import h5py
import os.path
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as scio
from scipy import sparse
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from math import radians, cos, sin, asin, sqrt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler



def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2
    '''由经纬度计算距离'''
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为km
    return c * r   # 单位:km

def func(d, a, b):
    return a * pow(d, b)  # 幂函数表达式

def count_distance(train_data):
    """
    统计签到距离
    """
    distance = [] 
    train_data.sort_values(by=['userId_ind', 'localTime'], inplace=True)
    user_lat_lon = train_data[['userId_ind', 'latitude', 'longitude']]

    for ind, group in tqdm(user_lat_lon.groupby('userId_ind')):
        group = group.values
        for i in range(group.shape[0] - 1):
            h = haversine(group[i, 2], group[i, 1], group[i+1, 2], group[i+1, 1])  # 距离
            distance.append(h)
    distance = np.array(distance)
    return distance


def fit_distance(distance):
    """
    拟合距离的幂函数
    """
    distance_bins = np.linspace(distance.min(), distance.max(), 3000)  # 距离 x
    bins_res = pd.Series(distance).value_counts(bins = distance_bins, sort = False)  # 频率 y
    popt, pcov = curve_fit(func, distance_bins[1:], bins_res)  # 拟合幂函数
    # # plot结果
    # y = [func(i, popt[0], popt[1]) for i in distance_bins[1:]]
    # plt.plot(distance_bins[1:], bins_res, 'b-', label='real')
    # plt.plot(distance_bins[1:], y, 'r--', label='fit')
    # plt.legend()
    # plt.show()
    return popt[0], popt[1]


def calculate_GeoProx(row, venue_info, PD):
    """
    计算当前POI与其他POI的地理邻近性
    """
    geo_vec = np.zeros((venue_info.shape[0]))
    num_venue = geo_vec.shape[0]
    for i in range(row+1, num_venue):
        h = haversine(venue_info[row, 1], venue_info[row, 0],
                      venue_info[i, 1], venue_info[i, 0])
        if h != 0:
            geo = PD[0] * pow(h, PD[1])  # 地理邻近性
            geo += np.random.normal(0., 0.05) # 加入高斯白 2021-09-19
            geo_vec[i] = geo 
    return geo_vec


def get_GeoProx_weight(train_data):
    """
    基于幂律分布计算地理邻近性矩阵
         
    :params train_data  数据处理后的训练集
    :return geo_matrix  地理邻近性矩阵
    """
    data = train_data.copy()
    # 1. 统计签到距离
    distance = count_distance(data)
    
    # 2. 拟合距离的幂函数
    PD_a, PD_b = fit_distance(distance)    
    
    # 3. 计算地理邻近性
    venue_info = data[['venueId','latitude', 'longitude']].drop_duplicates()
    venue_info.sort_values(by='venueId', inplace=True)
    venue_info = venue_info[['latitude', 'longitude']].values
    
    tasks = [delayed(calculate_GeoProx)(row, venue_info, [PD_a, PD_b]) for row in range(venue_info.shape[0])]
    geo_matrix = Parallel(n_jobs=-1)(tasks)   # 多个时间段数据并行处理
    geo_matrix = np.array(geo_matrix)
    geo_matrix = geo_matrix.T + geo_matrix  # 转换为对角阵

    return geo_matrix


def calculate_preference_TFIDF(group_time):
    """
    基于TF-IDF的POI偏好计算
    """
    N_user = group_time['userId_ind'].drop_duplicates().shape[0]  # user数量
    group_time_aggCount = (group_time
                           .groupby(['userId_ind', 'venueCategoryID', 'venueId'])['latitude']
                           .count()
                           .reset_index()
                           .rename(columns={'latitude':'counts'}))  # 用户-地点的频数Df
    P_ij = []  # 存放用户i对地点j的偏好
    for ind_UV, group_UV in tqdm(group_time_aggCount.groupby(['userId_ind', 'venueCategoryID', 'venueId'])):
        user = ind_UV[0]
        category = ind_UV[1]
        venue = ind_UV[2]
        N_ic = group_time_aggCount[(group_time_aggCount['userId_ind'] == user) & 
                                   (group_time_aggCount['venueCategoryID'] == category)].shape[0]  # 用户i签到的category不同POI数量
        N_c = group_time_aggCount[(group_time_aggCount['venueCategoryID'] == category)].drop_duplicates('userId_ind').shape[0]  # 签到过category的总user数量
        sum_N_ik = group_time_aggCount[(group_time_aggCount['userId_ind'] == user)].shape[0]  # 用户i签到的所有类别的不同POI数量
        x_ij = group_UV['counts'].values[0]  # 用户i对当前POI签到频数
        num_check_i = group_time_aggCount[(group_time_aggCount['userId_ind'] == user)]['counts'].sum()  # 用户i所有签到频数

        b_ic = N_ic / sum_N_ik * np.log((N_user + 1) / N_c)  # 用户i第j行地点的类别c对用户i的重要程度
        a_ij = x_ij / num_check_i  # 代表用户i所有签到频数中，地点j所占的比重
        p_ij = a_ij * b_ic * np.log(x_ij + 1)  # 用户i对地点j的偏好
        
        P_ij.append([user, category, venue, p_ij])
    Preference = pd.DataFrame(P_ij, columns=['userId_ind', 'venueCategoryID', 'venueId', 'Preference'])
    group_time = group_time.merge(Preference, on=['userId_ind', 'venueCategoryID', 'venueId'], how='left')
    return group_time


def get_visi_input(train_data):
    """
    基于TF-IDF生成用户偏好模型输入矩阵
             
    :params train_data  数据处理后的训练集
    :return DAE_input_v  {'timeBins': visible input}
    """
    data = train_data.copy()     
    num_venue =data.drop_duplicates(['venueId']).shape[0]  # 地点数量
    num_user = data.drop_duplicates(['userId']).shape[0]   # 用户数量
    
    # 1. 基于TF-IDF的POI偏好计算
    tasks = [delayed(calculate_preference_TFIDF)(group_time) for _, group_time in data.groupby('time_bins')]
    data_timeBins_input = Parallel(n_jobs=-1)(tasks)   # 多个时间段数据并行处理
    data_timeBins_input = pd.concat(data_timeBins_input, axis=0)
    
    # 2. Preference 标准化
    zscore_scaler = StandardScaler()
    Preference = data_timeBins_input['Preference'].values.reshape(-1, 1)
    data_timeBins_input['Preference'] = zscore_scaler.fit_transform(Preference)
    # data_timeBins_input.to_csv(r'./data/Tokyo_train_data_preference.csv', index=0)  
    
    # 3. 生成DAE输入矩阵
    DAE_input_v = {}
    for ind_bins, group_time in tqdm(data_timeBins_input.groupby('time_bins')):   

        group_time = group_time.drop_duplicates(['userId_ind', 'venueId'])
        input_v = np.zeros((num_user, num_venue))  # 初始化输入数据，用户总数行，地点总数列;
        for ind_user, group_user in group_time.groupby('userId_ind'): 
            i_venueId = np.array(group_user['venueId'])  
            i_preference = np.array(group_user['Preference']) 
            input_v[ind_user-1, i_venueId-1] = i_preference
        input_v = sparse.coo_matrix(input_v)
        DAE_input_v[ind_bins] = input_v
        # path_save = f'./data/temp/Tokyo_input_v_TBins{ind_bins}.npz'
        # sparse.save_npz(path_save, input_v, True)  # 保存稀疏矩阵
    return DAE_input_v, data_timeBins_input


def calculate_condition_input(data_dic, timeBins, n_timebins=1):
    """
    基于余弦相似度计算得到n_timebins个时间条件层输入
    :params data_dic    DAE的可见层输入, coo_matrix
    :params timeBins    时间段ID
    :params n_timebins  时间条件层数量
    :return  DAE的时间条件层输入
    """
    input_c = np.zeros([n_timebins]+list(data_dic[1].shape))
    for userId in tqdm(range(data_dic[1].shape[0])):
        key_dic = {}
        for key in data_dic.keys():
            if key == timeBins:  
                continue
            cos = cosine_similarity([data_dic[timeBins][userId]], [data_dic[key][userId]])[0,0]
            key_dic[key] = cos
        sort_val = sorted(key_dic.items(), key=lambda x:x[1],reverse=True)  # 相似度排序

        for n in range(n_timebins):
            if sort_val[n][1] > 0:
                input_c[n][userId] = data_dic[sort_val[0][0]][userId]
            else: # 相似度等于小于零 使用目标时间段相近时间段的数据
                input_c[n][userId] = data_dic[timeBins-n-1 if timeBins-n-1 > 0 else max(data_dic)][userId]
                
    input_c = (timeBins, [sparse.coo_matrix(f) for f in input_c] )  # 转为稀疏矩阵存储
    return input_c


def get_condition_input(DAE_input_v):
    """
    基于余弦相似度计算得到n_timebins个时间条件层输入
    :params DAE_input_v    DAE的可见层输入, coo_matrix
    :return DAE_input_c {'timeBins': condition input}
    """
    # coo_matrix to array
    for key in DAE_input_v.keys():
        if not isinstance(DAE_input_v[key], np.ndarray):
            DAE_input_v[key] = DAE_input_v[key].toarray()

    # 基于余弦相似度计算得到时间条件层输入
    DAE_input_c_temp = []
    tasks = [delayed(calculate_condition_input)(DAE_input_v, time) for time in DAE_input_v.keys()]
    DAE_input_c_temp = Parallel(n_jobs=4)(tasks) 
    DAE_input_c = dict(DAE_input_c_temp)
    
    # array to coo_matrix
    for key in DAE_input_v.keys():
        if isinstance(DAE_input_v[key], np.ndarray):
            DAE_input_v[key] = sparse.coo_matrix(DAE_input_v[key])
    
    # # 保存条件层输入稀疏矩阵
    # for key, vals in DAE_input_c.items():
    #     for ind, val in enumerate(vals):
    #         path_save = f'./data/temp/Tokyo_input_c_TBins{key}_{ind}.npz'
    #         sparse.save_npz(path_save, val, True)  

    return DAE_input_c


def get_friendship(data, path_friendship):
    """
    处理社交关系数据
    
    :params data   get_visi_input()生成的偏好数据集
    :return  friendship_ind 社交关系
    """

    friend_dataset = pd.read_csv(path_friendship, header=None, sep='\t', names=['userId', 'friendId'])

    train_user = data.drop_duplicates('userId')[['userId', 'userId_ind']].reset_index(drop=True)
    friendship_tem = friend_dataset.merge(train_user, on='userId', how='right')
    friendship_tem = friendship_tem.merge(train_user, left_on='friendId', right_on='userId', how='right')
    friendship_tem = friendship_tem.dropna().astype('int')

    # 训练集社交关系 userId
    friendship = friendship_tem.iloc[:, :2]
    friendship.columns = ['userId', 'friendId']

    # 训练集社交关系 userId_ind
    friendship_ind = friendship_tem.iloc[:, [2,4]]
    friendship_ind.columns = ['userId_ind', 'friendId_ind']
    # friendship_ind.to_csv(r'./data/Tokyo_friendship_ind.csv', index=0)   # 保存文件
    return friendship_ind


def get_LBSN2vec_input(data, friendship_ind, path_save=r'./matlab/LBSN2vec_input.mat'):
    """
    生成LBSN2vec输入数据并保存
    
    :params data   get_visi_input()生成的偏好数据集
    :params  friendship_ind  社交关系
    :params  path_save  保存地址
    """
    
    data.sort_values(by=['venueId'], inplace=True) 
    venueId = data.drop_duplicates(['venueId'])['venueId'].values.reshape((-1, 1))  # venueId
    data.sort_values(by=['userId'], inplace=True)  
    userId = data.drop_duplicates(['userId'])['userId'].values.reshape((-1, 1))  # userId
    data.sort_values(by=['localTime'], inplace=True)  # 按时间排序
    selected_checkins = data[['userId_ind', 'time_bins', 'venueId', 'venueCategoryID']].values

    
    scio.savemat(path_save, {'selected_checkins': selected_checkins, 'selected_users_IDs': userId,
                            'selected_venue_IDs': venueId, 'friendship_ind': friendship_ind.values})  # 保存 .mat 文 件

    
def get_friendship_expand(friendship_ind):
    """
    扩充社交关系数据
    
    :params  friendship_ind  社交关系
    :params  path_save  保存地址
    """
    path = './matlab/predict_userss_rank.mat'
    if os.path.isfile(path):
        # 读取潜在好友数据 行为user index，列为推荐列表长度 K，值为潜在好友Id
        mat = h5py.File('./matlab/predict_users_rank.mat')
        friendship_predict = np.transpose(mat['predict_users_rank'])[:, 1:3]
        friendship_predict = pd.DataFrame(friendship_predict)
        friendship_predict.columns = ['userId_ind', 'friendId_ind']  # 更改列名
        friendship_ind = pd.concat([friendship_ind, friendship_predict])

        # 双边社交关系
        friendship_ind_exp = pd.DataFrame(np.r_[friendship_ind.values, friendship_ind.values[:, [1, 0]]],
                                          columns= ['userId_ind', 'friendId_ind'], dtype='int').dropna()
    else:
        friendship_ind_exp = friendship_ind
    return friendship_ind_exp


def get_regularization_input(data, friendship_ind_exp, DAE_input_v):
    """
    生成社交正则项输入
    
    :params  data  get_visi_input()生成的偏好数据集
    :params  friendship_ind_exp  社交关系
    :params  DAE_input_v  DAE的可见层输入, coo_matrix
    :return DAE_input_r  {'timeBins': regularization input}
    """
    friendship_ind_exp = friendship_ind_exp.astype(int)
    num_venue = data.drop_duplicates(['venueId']).shape[0]  # 地点数量
    user_all =  data.drop_duplicates(['userId_ind'])['userId_ind'].sort_values()
    num_user = user_all.shape[0]   # 用户数量

    # 游客的目标城市签到记录
    visitors_loc = data[(data['venue_trans'] == 0) & (data['user_label'] == 'visitors')]  

    # 游客在目标城市的平均偏好
    input_r_vis = np.zeros(num_venue)
    for j, group in visitors_loc.groupby('userId'):
        j_r = np.zeros(num_venue)  # 该游客的偏好
        j_venue_index = np.array(group['venueId'])  
        j_preference = np.array(group['Preference'])

        j_r[j_venue_index-1] = j_preference  # -1因为venue索引从1开始
        input_r_vis += j_r
    input_r_vis = input_r_vis / visitors_loc['userId'].unique().shape[0] 


    # coo_matrix to array
    for key in DAE_input_v.keys():
        if not isinstance(DAE_input_v[key], np.ndarray):
            DAE_input_v[key] = DAE_input_v[key].toarray()
            
    # 正则项输入矩阵计算        
    DAE_input_r = {}        
    user_label = data[['userId_ind', 'user_label']].drop_duplicates().set_index('userId_ind')  # 本地人和游客对应表
    for key in tqdm(DAE_input_v.keys()):
        input_v_TBins = DAE_input_v[key]
        input_r = np.zeros((num_user, num_venue))
        
        for user_i in user_all:
            user_i_label = user_label.loc[user_i, 'user_label'] 
            friendship_i = friendship_ind_exp[friendship_ind_exp['userId_ind'] == user_i]  # 社交关系
            if friendship_i.shape[0] > 0:
                input_r[user_i-1] = input_v_TBins[friendship_i['friendId_ind'].values-1].mean(axis=0)
            if user_i_label == 'visitors':
                input_r[user_i-1] = (input_r[user_i-1] + input_r_vis) / 2

        DAE_input_r[key] = sparse.coo_matrix(input_r)
        
    # array to coo_matrix
    for key in DAE_input_v.keys():
        if isinstance(DAE_input_v[key], np.ndarray):
            DAE_input_v[key] = sparse.coo_matrix(DAE_input_v[key])
        
    return DAE_input_r