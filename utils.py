"""
Created on May 25, 2020

create amazon electronic dataset

@author: Ziyao Geng
"""
import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def create_amazon_electronic_dataset(file, embed_dim=8, maxlen=40):
    """
    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    with open('raw_data/remap.pkl', 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']

    train_data, val_data, test_data = [], [], []

    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        pos_list = hist['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []
        for i in range(1, len(pos_list)):
            hist.append([pos_list[i - 1], cate_list[pos_list[i-1]]])
            hist_i = hist.copy()
            if i == len(pos_list) - 1:
                test_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                # test_data.append([hist_i, [pos_list[i]], 1])
                # test_data.append([hist_i, [neg_list[i]], 0])
            elif i == len(pos_list) - 2:
                val_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                # val_data.append([hist_i, [pos_list[i]], 1])
                # val_data.append([hist_i, [neg_list[i]], 0])
            else:
                train_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                # train_data.append([hist_i, [pos_list[i]], 1])
                # train_data.append([hist_i, [neg_list[i]], 0])

    # feature columns
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim)]]
                        #sparseFeature('cate_id', cate_count, embed_dim)]]
    # feature_columns = [[],
    #                    [sparseFeature('item_id', item_count, embed_dim),
    #                     ],[sparseFeature('cate_id', cate_count, embed_dim)]]
    # behavior
    #behavior_list = ['item_id']  # , 'cate_id'
    behavior_list = ['item_id']#, 'cate_id']
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
               pad_sequences(train['hist'], maxlen=maxlen),
               np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
             pad_sequences(val['hist'], maxlen=maxlen),
             np.array(val['target_item'].tolist())]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
              pad_sequences(test['hist'], maxlen=maxlen),
              np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)

# create_amazon_electronic_dataset('raw_data/remap.pkl')



def create_movielens20M_dataset():
    def preprocess_target(target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target
    rating_dataset_path = './movielens_prerocess/ml-20m/ratings.csv'

    sep = ','
    engine = 'c'
    header = 'infer'
    rating_df = pd.read_csv(rating_dataset_path, sep=sep, engine=engine, header=header)


    # In[121]:


    category_dataset_path = './movielens_prerocess/ml-20m/movies.csv'

    category_df = pd.read_csv(category_dataset_path, sep=sep, engine=engine, header=header)

    # keey only movieId and genres(categories of each movie)
    category_df = category_df[['movieId','genres']]
    # 类别只保留最后一个
    category_df['genres'] = category_df['genres'].str.split('|')
    category_df['genres'] = category_df['genres'].map(lambda x: x[-1])


    # In[122]:


    def build_map(df, col_name):
        """
        制作一个映射，键为列名，值为序列数字
        :param df: reviews_df / meta_df
        :param col_name: 列名
        :return: 字典，键
        """
        key = sorted(df[col_name].unique().tolist())
        m = dict(zip(key, range(len(key))))
        df[col_name] = df[col_name].map(lambda x: m[x])
        return m, key


    # In[123]:


    # 物品ID映射
    asin_map, asin_key = build_map(category_df, 'movieId')
    # 物品种类映射
    cate_map, cate_key = build_map(category_df, 'genres')
    # 用户ID映射
    revi_map, revi_key = build_map(rating_df, 'userId')


    # In[124]:


    user_count, item_count, cate_count = len(revi_map), len(asin_map), len(cate_map)


    # In[125]:


    cate_list = np.array(category_df['genres'], dtype='int32')


    # In[126]:


    category_df.head()


    # In[127]:


    # 按物品id排序，并重置索引
    category_df = category_df.sort_values('movieId')
    category_df = category_df.reset_index(drop=True)


    # In[128]:


    # reviews_df文件物品id进行映射，并按照用户id、浏览时间进行排序，重置索引
    rating_df['movieId'] = rating_df['movieId'].map(lambda x: asin_map[x])
    rating_df = rating_df.sort_values(['userId', 'timestamp'])
    rating_df = rating_df.reset_index(drop=True)
    rating_df = rating_df[['userId', 'movieId','rating','timestamp']]


    # In[133]:


    embed_dim=8
    maxlen=40


    train_data, val_data, test_data = [], [], []

    rating_user_group = rating_df.groupby('userId')
    total_group = len(rating_user_group)
    
    ##userId_list = []
    #for u, h in rating_user_group:
    #    userId_list.append(u)

    for each_group in tqdm(range(0,138492+1)):
        user_id = each_group
        hist = rating_user_group.get_group(each_group)
        pos_list = hist['movieId'].tolist()
        raw_label_list = hist['rating'].tolist()
        label_list = preprocess_target(np.array(raw_label_list,dtype='int32')).tolist()
        #def gen_neg():
        #    neg = pos_list[0]
        #    while neg in pos_list:
        #        neg = random.randint(0, item_count - 1)
        #    return neg

        #neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []
        for i in range(1, len(pos_list)):
            hist.append([pos_list[i - 1], cate_list[pos_list[i-1]]])
            hist_i = hist.copy()
            if i == len(pos_list) - 1:
                test_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], label_list[i]])
                #test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], int(1-label_list[i])])
            elif i == len(pos_list) - 2:
                val_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], label_list[i]])
                #val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], int(1-label_list[i])])
            else:
                train_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], label_list[i]])
                #train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], int(1-label_list[i])])
        hist.clear()
    # feature columns
    feature_columns = [[],
                    [sparseFeature('item_id', item_count, embed_dim)]]
                        #sparseFeature('cate_id', cate_count, embed_dim)]]

    # behavior
    behavior_list = ['item_id']#, 'cate_id']

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    #print("============Saving dataframe==============")
    #train.to_pickle("./movielens_prerocess/train.pkl")
    #val.to_pickle("./movielens_prerocess/val.pkl")
    #test.to_pickle("./movielens_prerocess/test.pkl")

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
            pad_sequences(train['hist'], maxlen=maxlen),
            np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
            pad_sequences(val['hist'], maxlen=maxlen),
            np.array(val['target_item'].tolist())]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
            pad_sequences(test['hist'], maxlen=maxlen),
            np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)


def create_ml_1m_dataset(file, trans_score=2, embed_dim=8, test_neg_num=100):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param test_neg_num: A scalar. The number of test negative samples
    :return: user_num, item_num, train_df, test_df
    """

    def preprocess_target(target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target

    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(file, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
    # filtering
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    data_df = data_df[data_df.item_count >= 5]
    # trans score
    #data_df = data_df[data_df.label >= trans_score]
    # sort
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])

    # read category file
    category_df = pd.read_csv('./movielens_prerocess/ml-1m/movies.dat', sep="::", engine='python',names=['movieId', 'title', 'genres'])
    # keey only movieId and genres(categories of each movie)
    category_df = category_df[['movieId','genres']]
    # 类别只保留最后一个
    category_df['genres'] = category_df['genres'].str.split('|')
    category_df['genres'] = category_df['genres'].map(lambda x: x[-1])



    def build_map(df, col_name):
        """
        制作一个映射，键为列名，值为序列数字
        :param df: reviews_df / meta_df
        :param col_name: 列名
        :return: 字典，键
        """
        key = sorted(df[col_name].unique().tolist())
        m = dict(zip(key, range(len(key))))
        df[col_name] = df[col_name].map(lambda x: m[x])
        return m, key
    # 物品种类映射
    cate_map, cate_key = build_map(category_df, 'genres')
    cate_count = len(cate_map)

    all_movie_list = list()
    movie2genres = {}
    for index, element in category_df.iterrows():
        all_movie_list.append(int(element["movieId"]))
        movie2genres[int(element["movieId"])] = int(element["genres"])

    #cate_list = np.array(category_df['genres'], dtype='int32')
    # 按物品id排序，并重置索引
    #category_df = category_df.sort_values('movieId')
    #category_df = category_df.reset_index(drop=True)


    # split dataset and negative sampling
    print('============Negative Sampling===============')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = data_df['item_id'].max()

    train_data, val_data, test_data = [], [], []
    for user_id, df in tqdm(data_df[['user_id', 'item_id', 'label']].groupby('user_id')):

        pos_list = df['item_id'].tolist()

        raw_label_list = df['label'].tolist()
        label_list = preprocess_target(np.array(raw_label_list,dtype='int32')).tolist()

        #def gen_neg():
        #    neg = pos_list[0]
        #    while neg in set(pos_list):
        #        while True:
        #            neg = random.randint(1, item_id_max)
        #            if neg in set(all_movie_list):
        #                break
        #    return neg

        #neg_list = [gen_neg() for i in range(len(pos_list))]# + test_neg_num)]

        hist = []
        for i in range(1, len(pos_list)):
            hist.append([pos_list[i - 1]])
            hist_i = hist.copy()
            if i == len(pos_list) - 1:
                test_data.append([hist_i, [pos_list[i], movie2genres[pos_list[i]]], label_list[i]])
                #test_data.append([hist_i, [neg_list[i], movie2genres[neg_list[i]]], 0])
                #test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], int(1-label_list[i])])
            elif i == len(pos_list) - 2:
                val_data.append([hist_i, [pos_list[i], movie2genres[pos_list[i]]], label_list[i]])
                #val_data.append([hist_i, [neg_list[i], movie2genres[neg_list[i]]], 0])
                #val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], int(1-label_list[i])])
            else:
                train_data.append([hist_i, [pos_list[i], movie2genres[pos_list[i]]], label_list[i]])
                #train_data.append([hist_i, [neg_list[i], movie2genres[neg_list[i]]], 0])
                #train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], int(1-label_list[i])])
                
    # feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    

    # feature columns
    feature_columns = [[],
                    [sparseFeature('item_id', item_num, embed_dim),sparseFeature('cate_id', cate_count, embed_dim)]]

    maxlen=40


    # behavior
    behavior_list = ['item_id', 'cate_id']

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    #print("============Saving dataframe==============")
    #train.to_pickle("./movielens_prerocess/train.pkl")
    #val.to_pickle("./movielens_prerocess/val.pkl")
    #test.to_pickle("./movielens_prerocess/test.pkl")

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
            pad_sequences(train['hist'], maxlen=maxlen),
            np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
            pad_sequences(val['hist'], maxlen=maxlen),
            np.array(val['target_item'].tolist())]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
            pad_sequences(test['hist'], maxlen=maxlen),
            np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    print('============Data Preprocess End=============')

    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)