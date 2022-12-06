import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
from embedding import Node2Vec
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


def get_poi_category_edge(data):
    """
    构造POI-category 边集
    """
    data_com = data.copy()
    edge_df = data_com[['venueId', 'venueCategory']].drop_duplicates()

    # POIvenueCategory文本分词
    edge_df['venueCategory_split'] = edge_df['venueCategory'].apply(lambda x:re.findall('[a-zA-Z]+',x))
    edge_df.reset_index(drop=True, inplace=True)
    venueCategory_split = pd.DataFrame(edge_df['venueCategory_split'].tolist())   # 文本分词
    edge_df = edge_df.drop('venueCategory_split', axis=1).join(venueCategory_split
                                                               .stack()
                                                               .reset_index(level=1, drop=True)
                                                               .rename('venueCategory_split'))
    # 上下文单词编码
    le = LabelEncoder()
    le.fit(edge_df['venueCategory_split'])
    edge_df['venueCategory_split_id'] = le.transform(edge_df['venueCategory_split'])
    edge_df['venueCategory_split_id'] = edge_df['venueCategory_split_id'] + data_com['venueId'].max() + 1  
    edge_df = edge_df[['venueId', 'venueCategory_split_id']]
    # 保存临时文件
    tem_path = r'./data/temp/'
    if os.path.exists(tem_path):
            pass
    else:
        os.makedirs(tem_path)
    edge_df.to_csv('./data/temp/POI_edgelist.txt', index=False, header=False, sep=' ')
    return edge_df


def get_edge_embeddings():
    """
    embedding
    """ 
    G = nx.read_edgelist('./data/temp/POI_edgelist.txt',
                         create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])
    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=0.25, q=4, workers=1, use_rejection_sampling=0)
    model.train(embed_size=16, window_size = 5, iter = 10)
    embeddings = model.get_embeddings()
    return embeddings


def get_venue_transform(data_city, data_uncity, data_com_city, embeddings):
    """
    兴趣点转换
    """
    data_city = data_city.copy()
    data_uncity_venue = data_uncity['venueId'].unique()
    data_city_venue = data_city['venueId'].unique()

    # 计算embedding 余弦相似度
    venue_ori = [int(ind) for ind in data_uncity['venueId'].unique()]
    cos = cosine_similarity([embeddings[str(ind)] for ind in data_uncity_venue],
                            [embeddings[str(ind)] for ind in data_city_venue])
    venue_trans = []
    for ind in tqdm(range(cos.shape[0])): 
        venue_trans.append([int(ind) for ind in data_city_venue][np.argmax(cos[ind, :])])

    venue_ori_trans = pd.DataFrame()
    venue_ori_trans['venueId'] = venue_ori
    venue_ori_trans['venue_trans'] = venue_trans
    # venue_信息表
    venue_cate = data_com_city.drop_duplicates(['venueId', 'venueCategory']).copy()  
    venue_cate.drop(['userId', 'localTime', 'user_label'], axis=1, inplace=True)
    venue_cate.rename(columns={'venueId':'venue_trans'}, inplace = True)
    # 兴趣点转换
    data_uncity_trans = data_uncity.merge(venue_ori_trans, on='venueId', how='left')
    data_uncity_trans = (data_uncity_trans[['userId', 'localTime', 'venueId', 
                                'venue_trans', 'user_label']].merge(venue_cate, on='venue_trans',how='left'))
    data_uncity_trans.drop(['venueId'], axis=1, inplace=True)
    data_uncity_trans.rename(columns={'venue_trans':'venueId'}, inplace = True)
    data_uncity_trans.reset_index(drop=True, inplace=True)
    data_uncity_trans['venue_trans'] = 1
    
    # 合并数据
    data_city['venue_trans'] = 0
    data_city = data_city[data_uncity_trans.columns]  
    data_transform = pd.concat([data_uncity_trans, data_city], axis=0)
    data_transform.sort_values(['userId', 'localTime'], inplace=True)
    data_transform.reset_index(drop=True, inplace=True)
    return data_transform


def get_time_cluster(data, n_clus=12):
    """
    时间聚类
     
    :params data  兴趣点迁移处理后的数据集
    :params n_clus  时间段划分数量
    """
    # 编码
    data_time = data.copy()
    time_encodding = pd.Series(pd.date_range(start='20170101',end='20170102',freq='T')).dt.time
    time_encodding = time_encodding.iloc[:-1]

    le = LabelEncoder() 
    le = le.fit(time_encodding)
    times = pd.to_datetime(data_time["localTime"].dt.hour.astype(str) +
                          ':' + data_time["localTime"].dt.minute.astype(str), format='%H:%M').dt.time
    time_encoding = le.transform(times)
    
    # 聚类
    time_encoding = np.array(time_encoding)
    time_encoding = time_encoding.reshape(-1,1)

    km = KMeans(n_clusters=n_clus)
    km.fit(time_encoding)
    # 聚类质心
    time_point = pd.Series(le.inverse_transform([int(x) for x in km.cluster_centers_])).sort_values()
    data_time["time_bins"] = km.labels_ + 1
    # data_time.to_csv('./data/temp/Tokyo_time_data.csv', index=False)
    return data_time, time_point


def get_split_dataset(data, split_size=0.8):
    """
    划分训练集，对地点、类别特征进行标签编码
     
    :params data  时间聚类划分处理后的数据集
    :params split_size  训练集划分比例
    """
    data_time = data.copy()
    data_time.sort_values(by=['localTime'], inplace=True)  
    
    size_train = int(data_time.shape[0] * split_size)
    train_data = data_time.iloc[:size_train].copy()
    test_data = data_time.iloc[size_train:].copy()

    # 去除测试集中在训练集没有出现过的用户和POI
    userId_train = train_data.drop_duplicates(['userId'])['userId']  # 不重复user集合
    venueId_train = train_data.drop_duplicates(['venueId'])['venueId']  # 不重复venue集合

    test_data = test_data.loc[(test_data['userId'].isin(userId_train)) &
                              test_data['venueId'].isin(venueId_train)]
    print(f'dataset shape: all{data_time.shape}; train{train_data.shape}; test{test_data.shape}')

    # 编码
    columns = ['venueId', 'venueCategoryID']
    for col in columns:
        scale = LabelEncoder()
        train_data[col] = scale.fit_transform(train_data[col]) + 1
        test_data[col] = scale.transform(test_data[col]) + 1

    scale = LabelEncoder()
    train_data['userId_ind'] = scale.fit_transform(train_data['userId']) + 1
    test_data['userId_ind'] = scale.transform(test_data['userId']) + 1

    all_data = pd.concat([train_data, test_data])  # 合并数据集

    # 保存文件
    # train_data.to_csv(r'./data/temp/Tokyo_train_data.csv', index=0)   
    # test_data.to_csv(r'./data/temp/Tokyo_test_data.csv', index=0) 
    # all_data.to_csv(r'./data/temp/Tokyo_data.csv', index=0) 

    return all_data, train_data, test_data
