# -*- coding:utf-8 -*-

import os
import pickle
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calc_corr(a, b):
    s1 = Series(a)
    s2 = Series(b)
    return s1.corr(s2)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def create_graph(num_nodes, data):
    edge_index = [[], []]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            x, y = data[:, i], data[:, j]
            corr = calc_corr(x, y)
            if abs(corr) >= 0.2:   
                edge_index[0].append(i)
                edge_index[1].append(j)

    edge_index = torch.LongTensor(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index


def adj2coo(adj):
    # adj numpy
    edge_index_temp = sp.coo_matrix(adj)
    values = edge_index_temp.data
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    edge_index = torch.LongTensor(indices)

    return edge_index


def load_data():
    #path = r'D:\Users\三木\Desktop\shanghai.csv'
    path = r'D:\Users\三木\Desktop\qingdao.csv'
    df = pd.read_csv(path)
    # 这里需要注意，如果除了时间列(idx == 0)之外还有其他列是非数字列，需要将这些
    # 列先处理成数字或者删除这些列
    for idx, c in enumerate(df.columns):
        if idx == 0:
            continue   # 跳过时间列
        df[c].fillna(df[c].mean(), inplace=True)  #
    return df


def nn_seq(args):
    num_nodes, seq_len = args.input_size, args.seq_len,
    batch_size, output_size = args.batch_size, args.output_size
    pre_len=args.pre_len

    data = load_data()
    #data = data[:1000]

    # 去掉时间列
    data.drop([data.columns[0]], axis=1, inplace=True)
    # split
    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    # 归一化
    # 这里也需要保证去除了时间列之后，其他列都是数字类型，否则无法归一化
    scaler = MinMaxScaler()
    scaler.fit_transform(data[:int(len(data) * 0.8)].values)  #
    train = scaler.transform(train.values)
    val = scaler.transform(val.values)
    test = scaler.transform(test.values)
    edge_index = create_graph(num_nodes, data[:int(len(data) * 0.8)].values)
    def process(dataset, step_size, shuffle):
        dataset = dataset.tolist()##转换为列表
        seq = []
        for i in tqdm(range(0, len(dataset) - seq_len - output_size + 1, step_size)):###tqdm显示进度条进度
            train_seq = []
            for j in range(i, i + seq_len):
                x = []
                for c in range(len(dataset[0])):  # 前24个时刻的所有变量
                    x.append(dataset[j][c])
                train_seq.append(x)
            # 下几个时刻的所有变量
            train_labels = []
            for j in range(len(dataset[0])):
                train_label = []
                for k in range(i + seq_len, i + seq_len + pre_len):
                    train_label.append(dataset[k][j])
                train_labels.append(train_label)
            # tensor
            train_seq = torch.FloatTensor(train_seq)
            # print(train_seq.shape)   # 24 13
            train_labels = torch.FloatTensor(train_labels)
            # edge_index = create_graph(num_nodes, train_seq.numpy())
            # graph = Data(x=train_seq.T, edge_index=edge_index, y=train_labels)
            seq.append((train_seq, train_labels))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

        return seq

    Dtr = process(train, step_size=1, shuffle=True)
    Val = process(val, step_size=1, shuffle=True)
    Dte = process(test, step_size=3, shuffle=False)
    return Dtr, Val, Dte, scaler, edge_index


def save_pickle(dataset, file_name):
    f = open(file_name, "wb")
    pickle.dump(dataset, f)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset
