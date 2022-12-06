import numpy as np
import pandas as pd
from tqdm import tqdm
import utm


# 读取原始数据集
def get_load_country_dataset(country='JP'):
    """
    读取数据集
    : params country: 国家简称
    : return : 数据集中包含该国家所有的签到记录
    """
    # 签到数据集
    df_checkins = pd.read_csv(r".\data\dataset\dataset_WWW_Checkins_anonymized.txt",
                              header=None,
                              sep='\t',
                              names=['userId', 'venueId', 'utcTimestamp', 'offset']
                             )
#     df_checkins.columns = ['userId', 'venueId', 'utcTimestamp', 'offset']
    # POI数据集
    df_pois = pd.read_csv(r".\data\dataset\raw_POIs.txt",
                          header=None,
                          sep='\t',
                          names=['venueId',  'latitude', 'longitude', 'venueCategory', 'country'] 
                           )
    # 根据共有的venueId字段取交集合并两个数据集
    df_dataset = pd.merge(df_checkins, df_pois, how='inner', on='venueId') 
    # 国家数据集
    df_dataset = df_dataset.loc[df_dataset['country'] == country]
    df_dataset.reset_index(drop=True, inplace=True)
    return df_dataset


def get_utcTime_to_localTime(data):
    # UTC时间转本地时间
    da = data.copy()
    da["utcTimestamp"] = pd.to_datetime(da["utcTimestamp"], 
                                          format='%a %b %d %H:%M:%S +0000 %Y')

    da["utcTimestamp"] = da.apply(lambda x:x["utcTimestamp"] -
                                   pd.Timedelta(minutes=x['offset']),
                                   axis=1)
    da.rename(columns={'utcTimestamp': 'localTime'}, inplace=True)
    da.drop(columns='offset', inplace=True)
    return da


def get_venue_category_encoder(data):
    # venueId, venueCategory编码
    da = data.copy()
    scale = LabelEncoder()
    da['venueId'] = scale.fit_transform(da['venueId']) 
    da['venueCategoryID'] = (scale.fit_transform(da['venueCategory']) +
                             da['venueId'].max() + 1)  # 不和venueId编码重复 
    return da


def get_utm_coordinate(data):
    
    # utm坐标转换
    da = data.copy()
    ite = zip(da['latitude'], da['longitude'])
    list_utm = []
    for lat, lon in tqdm(ite):
        list_utm.append(utm.from_latlon(lat,lon))

    df_utm = pd.DataFrame(list_utm,
                          columns=['Eeasting', 'Northing', 'Zone_number', 'Zone_letter'])

    da_utm = pd.concat([da, df_utm], axis=1)
    da_utm.drop(columns = 'country', inplace=True)
    # 划分25*25km的地域块
    da_utm['block_label'] =  ((da_utm['Eeasting'] / 25000).round().astype(str) + 
                                (da_utm['Northing'] / 25000).round().astype(str) +
                                da_utm['Zone_number'].astype(str) + 
                                da_utm['Zone_letter'])
    return da_utm


def get_define_home(data):
    # 定义用户home
    da = data.copy()
    home_set = []
    for ind, groups in tqdm(da.groupby('userId')):
        # 用户访问最多的地区定义为家所在地区
        home_block_label = groups['block_label'].value_counts().index[0]  
        df_home_block = groups[groups['block_label'] == home_block_label].copy()
        # 家所在地区内POI的utm坐标取mean定义为home
        home_coor = df_home_block[['Eeasting', 'Northing']].mean().values  
        lat, lon = utm.to_latlon(home_coor[0], 
                                 home_coor[1],
                                 int(home_block_label[-3:-1]),
                                 home_block_label[-1])  # 转换为经纬度
        home_set.append((ind, f'{ind}home', np.nan, lat, lon, 'home', 1000, home_coor[0], 
                         home_coor[1], int(home_block_label[-3:-1]), home_block_label[-1], home_block_label))

    home_df = pd.DataFrame(home_set, columns=da.columns)
    return home_df


def get_city_dataset(data, coordinate, radius):
    """
    筛选城市
    : params coordinate: 城市经纬度，[latitude, longitude]
    : params radius: 城市半径，单位km
    """
    da = data.copy()
    bool_city = []                  # 存放布尔值
    for ind, row in tqdm(da.iterrows()):
        h = haversine(row['longitude'], row['latitude'], coordinate[1], coordinate[0])
        bool_city.append(h <= radius)  # 筛选出目标城市数据集

    print('data_city shape:', da.iloc[bool_city].shape)
    data_city = da.iloc[bool_city].copy().reset_index(drop=True)
    data_uncity = da.iloc[[not b for b in bool_city]].copy()
    return data_city, data_uncity


def get_native_visitors(data, home_df, coordinate, radius):
    """
    筛选游客和本地人
    """
    da = data.copy()
    
    bool_unnative, bool_native = [], []  # 存放布尔值
    for ind, row in tqdm(home_df.iterrows()):
        h = haversine(row['longitude'], row['latitude'], coordinate[1], coordinate[0])
        bool_unnative.append(h > 100)  
        bool_native.append(h <= radius)  
    # 本地人ID
    native_ID = home_df.iloc[bool_native]['userId'].values
    # 非本地人ID (家在目标城市100km外的用户)
    unnative_ID = home_df.iloc[bool_unnative]['userId'].unique()
    # 游客ID（家在目标城市100km外，且在目标城市内有签到记录的用户）
    visitors_ID = da.loc[da['userId'].isin(unnative_ID), 'userId'].unique()
    return native_ID, unnative_ID, visitors_ID


def get_city_dataset_process(data_city, data_uncity, native_ID, visitors_ID, unnative_ID):
    """
    得到最终数据集(目标城市and家乡城市)
    """
    # 在目标城市 本地居民和>100km的外地游客数据
    temp = list(native_ID)
    temp.extend(list(visitors_ID))
    data_city = data_city[data_city['userId'].isin(temp)]

    # 过滤被访问少于20次的POI 
    counts = data_city['venueId'].value_counts()
    POI_id_mult = list(counts[counts > 20].index)
    data_city = data_city[data_city['venueId'].isin(POI_id_mult)].copy()
    data_city.reset_index(drop=True, inplace=True)

    data_city.loc[data_city['userId'].isin(native_ID), 'user_label'] = 'native'
    data_city.loc[data_city['userId'].isin(visitors_ID), 'user_label'] = 'visitors'
    data_city['location'] = 'currentCity'

    # # 数据统计显示
    # data_city_native = data_city.loc[data_city['user_label'] == 'native']  
    # print('目标城市本地人数量:', data_city_native['userId'].unique().shape[0], 
    #      'POI:', data_city_native['venueId'].unique().shape[0],
    #      'checkins:', data_city_native.shape[0])
    # data_city_native = data_city.loc[data_city['user_label'] == 'visitors']  
    # print('游客数量:', data_city_native['userId'].unique().shape[0], 
    #      'POI:', data_city_native['venueId'].unique().shape[0],
    #      'checkins:', data_city_native.shape[0])
    
    # 在家乡城市 >100km的外地游客数据
    data_uncity = data_uncity[data_uncity['userId'].isin(unnative_ID)].copy()
    counts = data_uncity['venueId'].value_counts()
    POI_id_mult = list(counts[counts > 20].index)  # 过滤签到频率低的POI
    data_uncity = data_uncity[data_uncity['venueId'].isin(POI_id_mult)].copy()
    data_uncity.reset_index(drop=True, inplace=True)
    data_uncity['user_label'] = 'visitors'
    data_uncity['location'] = 'homeTown'
    # 数据统计显示
    data_city_native = data_uncity.loc[data_uncity['user_label'] == 'visitors']  
    # print('家乡城市游客数量:', data_city_native['userId'].unique().shape[0], 
    #      'POI:', data_city_native['venueId'].unique().shape[0],
    #      'checkins:', data_city_native.shape[0])
    
    # 数据合并保存
    data_city = data_city[data_uncity.columns]  # 列对齐
    data_com_city = pd.concat([data_uncity, data_city])
    data_com_city.drop(['Eeasting', 'Northing', 'Zone_number',
                        'Zone_letter', 'block_label'], axis=1, inplace=True)
    data_com_city.reset_index(drop=True, inplace=True)
    return data_city, data_uncity, data_com_city