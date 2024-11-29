#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(np.arange(num_shards), 50, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def nlp_iid(dataset, num_users):
    """
    按照 IID（独立同分布）方式将 NLP 数据划分为多个客户端。
    :param dataset: NLP 数据集
    :param num_users: 客户端数量
    :return: 客户端索引字典，每个键代表一个客户端，值是数据集的样本索引
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


# def quantity_based_label_skew(dataset, num_users, alpha = 6):
#     """
#     Sample non-I.I.D client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 100, 40
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     # labels = dataset.train_labels.numpy()
#     labels = np.array(dataset.TrainLabels)
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(np.arange(num_shards), alpha, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users
def quantity_based_label_skew(dataset, num_users, alpha=6):
    """
    使用数量驱动的标签偏斜来划分 NLP 数据集，将数据集划分为多个客户端。
    :param dataset: 数据集
    :param num_users: 客户端数量
    :param alpha: 每个客户端将获得的类别数量
    :return: 客户端索引字典，每个键代表一个客户端，值是数据集的样本索引
    """
    labels = np.array(dataset['label'])
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    dict_users = {i: [] for i in range(num_users)}
    label_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    for i in range(num_users):
        selected_labels = random.sample(list(unique_labels), alpha)
        for label in selected_labels:
            dict_users[i].extend(label_indices[label])
            label_indices[label] = [idx for idx in label_indices[label] if idx not in dict_users[i]]

    return dict_users


def distribution_based_label_skew(dataset, num_users, beta=0.1, min_require_size=1):
    """
    使用 Dirichlet 分布来模拟非 I.I.D 数据分布，以创建客户端数据划分。
    :param dataset: Dataset 数据集
    :param num_users: 客户端数量
    :param beta: Dirichlet 分布的浓度参数
    :param min_require_size: 每个客户端数据集的最小样本数
    :return: 客户端索引字典，每个键代表一个客户端，值是数据集的样本索引
    """
    N = len(dataset)
    y_train = np.array(dataset['label'])  # 获取所有样本的标签
    net_dataidx_map = {}
    min_size = 0

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in np.unique(y_train):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_split = np.split(idx_k, proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_split)]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
