#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import collections
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from sampling import *
from iCIFAR100 import iCIFAR100
import random
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from update import DatasetSplit

def get_trainable_param_names(model):
    return [name for name, param in model.named_parameters() if param.requires_grad]

def get_frozen_param_names(model):
    return [name for name, param in model.named_parameters() if not param.requires_grad]

def build_continual_dataset(args, class_order):

    class_mask = split_single_dataset(args, class_order)

    return class_mask


def get_trainand_test_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = '../../data'
    trans_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.24705882352941178),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trans_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
    #                                transform=trans_train)

    # test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
    #                               transform=trans_test)     

    train_dataset = iCIFAR100(data_dir, train=True, download=True,
                                transform=trans_train)

    test_dataset = iCIFAR100(data_dir, train=False, download=True,
                                test_transform=trans_test)  
    all_classes = [0, args.total_classes]
    test_dataset.getTestData(all_classes)
    train_dataset.getTrainData(all_classes)
    return train_dataset, test_dataset

def get_dataset(args, train_dataset, m, start, end, task_num):
    # sample training data amongst users
    if args.iid:
        current_class = random.sample([x for x in range(start, end)], task_num)
        train_dataset = train_dataset.filter(lambda example: example['label'] in current_class)
        user_groups = nlp_iid(train_dataset, m)
    else:
        if args.niid_type == "Q":
            current_class = random.sample([x for x in range(start, end)], task_num)
            train_dataset = train_dataset.filter(lambda example: example['label'] in current_class)
            user_groups = quantity_based_label_skew(train_dataset, m, alpha=args.alpha)
        else:
            # 从end-start这么多类中随机抽取task num个类
            current_class = random.sample([x for x in range(start, end)], task_num)
            # 只保留标签属于 current_class 中的样本。
            train_dataset = train_dataset.filter(lambda example: example['label'] in current_class)
            # 根据beta程度进行标签采样
            user_groups = distribution_based_label_skew(train_dataset, m, beta=args.beta)


    return train_dataset, user_groups

# def split_single_dataset(args, class_order):
#     nb_classes = args.total_classes
#     assert nb_classes % (args.task_num+1) == 0
#     classes_per_task = nb_classes // (args.task_num+1)
#
#     labels = [i for i in range(nb_classes)]
#
#     mask = list()
#
#     # if args.shuffle:
#     #     random.shuffle(labels)
#     class_till_now = classes_per_task
#     for _ in range(args.task_num+1):
#
#         # scope = class_order[:class_till_now]
#         # class_till_now += classes_per_task
#         scope = labels[:classes_per_task]
#         labels = labels[classes_per_task:]
#
#         mask.append(scope)
#
#     return mask

def split_single_dataset(args, class_order):
    nb_classes = args.total_classes
    assert nb_classes % (args.task_num + 1) == 0
    classes_per_task = nb_classes // (args.task_num + 1)

    labels = [i for i in range(nb_classes)]

    mask = list()

    # if args.shuffle:
    #     random.shuffle(labels)
    # class_till_now = classes_per_task
    for _ in range(args.task_num + 1):
        # scope = class_order[:class_till_now]
        # class_till_now += classes_per_task
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)

    return mask



# def compute_weight(centers_list, feature_list, epsilon=1e-6):
#     weight = []
#     for idx in range(len(centers_list)):
#         non_empty_indices = [i for i, a_item in enumerate(feature_list[idx]) if len(a_item) > 0]
#         if non_empty_indices:
#             # 获取非空的特征和中心数据
#             non_empty_a = [feature_list[idx][i] for i in non_empty_indices]
#             non_empty_b = np.array(centers_list[idx])
#
#             # 使用余弦距离来计算特征和中心之间的距离矩阵
#             distances_matrix = cdist(non_empty_b, non_empty_a, metric='cosine')
#
#             # 在距离矩阵中添加 epsilon，以避免零值
#             distances_matrix = np.clip(distances_matrix, epsilon, None)
#
#             # 计算总距离
#             total_distance = np.sum(distances_matrix, axis=1, keepdims=True)
#
#             # 避免总距离为零的情况
#             total_distance = [item if item > epsilon else epsilon for sublist in total_distance.tolist() for item in sublist]
#
#             # 计算倒数以获得权重
#             reciprocal_data = [1 / value for value in total_distance]
#
#             # 归一化数据
#             min_val = np.min(reciprocal_data)
#             max_val = np.max(reciprocal_data)
#             if max_val - min_val > epsilon:
#                 normalized_data = [(value - min_val) / (max_val - min_val) for value in reciprocal_data]
#             else:
#                 normalized_data = [1.0 / len(reciprocal_data) for _ in reciprocal_data]  # 如果最大值和最小值接近，直接均匀分配权重
#
#             # 使用 softmax 来计算权重
#             softmax_data = F.softmax(torch.tensor(normalized_data) / 0.2, dim=0)
#         else:
#             # 如果没有非空的特征，则返回均匀权重
#             softmax_data = torch.tensor([1.0 / len(centers_list)] * len(centers_list), dtype=torch.float64)
#
#         weight.append(softmax_data)
#
#     return weight
#
# def average_weights(weights_list, model, classes, niid_type, feature_list, backbone_weight, numclass):
#     centers_list = [[] for i in range(0, numclass)]
#     weight = []
#     trainable_params = get_trainable_param_names(model)
#     idx = 0
#     for _, name in enumerate(trainable_params):
#         if name.startswith('centers'):
#             for w in weights_list:
#                 centers_list[idx].append(w[name].squeeze().cpu().numpy())
#             idx += 1
#     # 求14式中的w
#     weight = compute_weight(centers_list, feature_list, numclass)
#
#     avg_weights = collections.OrderedDict()
#     weight_names = weights_list[0].keys()
#     index=0
#     for name in weight_names:
#         if name not in trainable_params:
#             if name in model.state_dict():
#                 avg_weights[name] = model.state_dict()[name]
#         else:
#             if name.startswith('centers'):
#                 aggregated_weight_tensor = torch.stack(
#                     [w[name] * weight[index][i] for i, w in enumerate(weights_list)]).sum(dim=0)
#                 avg_weights[name] = aggregated_weight_tensor
#                 index += 1
#             else:
#                 avg_weights[name] = torch.stack([w[name] * backbone_weight[i] for i, w in enumerate(weights_list)]).sum(dim=0)
#
#     return avg_weights

def compute_weight(centers_list, feature_list, epsilon=1e-6, device='cuda'):
    weight = []
    for idx in range(len(centers_list)):
        non_empty_indices = [i for i, a_item in enumerate(feature_list[idx]) if len(a_item) > 0]
        if non_empty_indices:
            # 获取非空的特征和中心数据
            non_empty_a = torch.tensor(feature_list[idx], device=device)
            non_empty_b = torch.tensor(centers_list[idx], device=device)

            # 使用余弦距离来计算特征和中心之间的距离矩阵
            distances_matrix = torch.cdist(non_empty_b, non_empty_a, p=2)

            # 在距离矩阵中添加 epsilon，以避免零值
            distances_matrix = torch.clamp(distances_matrix, min=epsilon)

            # 计算总距离
            total_distance = torch.sum(distances_matrix, dim=1, keepdim=True)

            # 避免总距离为零的情况
            total_distance[total_distance < epsilon] = epsilon

            # 计算倒数以获得权重
            reciprocal_data = 1.0 / total_distance

            # 归一化数据
            min_val = torch.min(reciprocal_data)
            max_val = torch.max(reciprocal_data)
            if max_val - min_val > epsilon:
                normalized_data = (reciprocal_data - min_val) / (max_val - min_val)
            else:
                normalized_data = torch.ones_like(reciprocal_data) / len(reciprocal_data)

            # 使用 softmax 来计算权重
            softmax_data = torch.nn.functional.softmax(normalized_data / 0.2, dim=0)
        else:
            # 如果没有非空的特征，则返回均匀权重
            softmax_data = torch.tensor([1.0 / len(centers_list)] * len(centers_list), dtype=torch.float64, device=device)

        weight.append(softmax_data.cpu())

    return weight


def average_weights(weights_list, model, classes, niid_type, backbone_weight, numclass, device='cuda'):
    # 初始化列表来保存中心向量
    # centers_list = [[] for _ in range(0, numclass)]
    trainable_params = get_trainable_param_names(model)
    # idx = 0

    # 获取 centers 的权重列表
    # for _, name in enumerate(trainable_params):
    #     if name.startswith('centers'):
    #         for w in weights_list:
    #             centers_list[idx].append(w[name].squeeze().detach())
    #         idx += 1

    # 计算权重 w
    # weight = compute_weight(centers_list, feature_list, epsilon=1e-6, device=device)

    avg_weights = collections.OrderedDict()
    weight_names = weights_list[0].keys()

    for name in weight_names:
        if name not in trainable_params:
            if name in model.state_dict():
                avg_weights[name] = model.state_dict()[name]
        else:
            # 确保所有张量在同一设备上
            aggregated_weight_tensor = torch.stack([w[name].to(device) * backbone_weight[i] for i, w in enumerate(weights_list)]).sum(dim=0)
            avg_weights[name] = aggregated_weight_tensor


    # 遍历所有权重名，计算平均权重
    # for name in weight_names:
    #     if name not in trainable_params:
    #         if name in model.state_dict():
    #             avg_weights[name] = model.state_dict()[name]
    #     else:
    #         if name.startswith('centers'):
    #             # 将所有张量转移到指定设备上（例如 GPU）
    #             aggregated_weight_tensor = torch.stack(
    #                 [w[name].to(device) * weight[index][i].to(device) for i, w in enumerate(weights_list)]
    #             ).sum(dim=0)
    #             avg_weights[name] = aggregated_weight_tensor
    #             index += 1
    #         else:
    #             avg_weights[name] = torch.stack(
    #                 [w[name].to(device) * backbone_weight[i] for i, w in enumerate(weights_list)]
    #             ).sum(dim=0).to(device)

    return avg_weights


def global_server(model, global_model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'lora' in name.lower() or 'classifier' in name:
                if name in global_model.state_dict():
                    # 检查模型和全局模型之间参数尺寸是否一致
                    if param.size() == global_model.state_dict()[name].size():
                        global_model.state_dict()[name].copy_(param.data.to(global_model.state_dict()[name].device))
                    else:
                        print(f"跳过参数 '{name}'，因为尺寸不匹配："
                              f"模型参数尺寸 {param.size()} vs 全局模型参数尺寸 {global_model.state_dict()[name].size()}")
    return global_model



# def global_server(model, global_model, Waq, Wav, Wbq, Wbv, current_task):
#     para_aq = []
#     para_bq = []
#     para_av = []
#     para_bv = []
#     centers = []
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if 'linear_a_q_{}'.format(current_task) in name:
#                 para_aq.append(param.data)
#             if 'linear_a_v_{}'.format(current_task) in name:
#                 para_av.append(param.data)
#             if 'linear_b_q_{}'.format(current_task) in name:
#                 para_bq.append(param.data)
#             if 'linear_b_v_{}'.format(current_task) in name:
#                 para_bv.append(param.data)
#             if 'centers' in name:
#                 centers.append(param.data)
#         idx = 0# 4 * current_task
#         num = 0
#         if current_task == 0:
#             for name, param in global_model.named_parameters():
#                 if 'linear_a_q_{}'.format(current_task) in name:
#                     param.data[idx:idx+4, :] = para_aq[0]
#                 if 'linear_a_v_{}'.format(current_task) in name:
#                     param.data[idx:idx+4, :] = para_av[0]
#                 if 'linear_b_q_{}'.format(current_task) in name:
#                     param.data[:, idx:idx+4] = para_bq[0]
#                 if 'linear_b_v_{}'.format(current_task) in name:
#                     param.data[:, idx:idx+4] = para_bv[0]
#                 if 'centers' in name:
#                     param.data = centers[num]
#                     num += 1
#         else:
#             for name, param in global_model.named_parameters():
#                 if 'linear_a_q_{}'.format(current_task) in name:
#                     param.data[idx:idx+4, :] = Waq[0] + para_aq[0]
#                 if 'linear_a_v_{}'.format(current_task) in name:
#                     param.data[idx:idx+4, :] = Wav[0] + para_av[0]
#                 if 'linear_b_q_{}'.format(current_task) in name:
#                     param.data[:, idx:idx+4] = Wbq[0] + para_bq[0]
#                 if 'linear_b_v_{}'.format(current_task) in name:
#                     param.data[:, idx:idx+4] = Wbv[0] + para_bv[0]
#                 if 'centers' in name:
#                     param.data = centers[num]
#                     num += 1
#     return global_model

def average_weights2(weights_list, model):
     avg_weights = collections.OrderedDict()
     weight_names = weights_list[0].keys()
     for name in weight_names:
         avg_weights[name] = torch.stack([w[name] for w in weights_list]).mean(dim=0)

     return avg_weights


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.encoders_lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Users in one epoch  : {args.client_local}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def initialize_datasets(self):
    # 对测试集和验证集进行编码
    def preprocess_function(examples):
        return self.global_model.tokenizer(
            examples['input_text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )

    # 测试集和验证集预处理
    self.test_set = self.test_set.map(preprocess_function, batched=True)
    self.valid_set = self.valid_set.map(preprocess_function, batched=True) if self.valid_set else None

    # 创建 DatasetSplit 子集
    self.test_dataset = DatasetSplit(self.test_set, list(range(len(self.test_set))))
    self.valid_dataset = DatasetSplit(self.valid_set, list(range(len(self.valid_set)))) if self.valid_set else None

    # 创建 DataLoader
    self.test_loader = DataLoader(
        self.test_dataset, batch_size=self.args.local_bs, shuffle=False, num_workers=0,
        collate_fn=self.data_collator
    )
    self.list_of_testloader.append(self.test_loader)

    if self.valid_dataset:
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=self.args.local_bs, shuffle=False, num_workers=0,
            collate_fn=self.data_collator
        )
    else:
        print("Warning: valid_set not found. Validation loader is not initialized.")


