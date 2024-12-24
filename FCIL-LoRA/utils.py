#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
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
from transformers import BartModel
from update import DatasetSplit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import pandas as pd
from datasets import concatenate_datasets, Dataset
import pickle
from torch import nn
from VLT import LLMWithLoRA, MyBart
from collections import OrderedDict

def prepare_sequence_finetune(args):
    with open(args.sequence_file.replace('_reduce', ''), 'r') as f:
        datas = f.readlines()[args.idrandom]
        data = datas.split()

    args.task_name = data

    if args.classifier_lr is None:
        args.classifier_lr = args.learning_rate

    if 'ewc' in args.baseline:
        args.lamda = 5000

    output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[args.ft_task]) + "_model/"
    ckpt = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[args.ft_task - 1]) + "_model/"


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


def get_dataset_noniid(args, train_dataset, m, start, end, task_num, idxs_users):
    # sample training data amongst users
    if args.iid:
        # current_class = random.sample([x for x in range(start, end)], task_num)
        # train_dataset = train_dataset.filter(lambda example: example['label'] in current_class)
        user_groups = nlp_iid(train_dataset, m)
    else:
        if args.niid_type == "Q":
            # current_class = random.sample([x for x in range(start, end)], task_num)
            # train_dataset = train_dataset.filter(lambda example: example['label'] in current_class)
            user_groups = quantity_based_label_skew(train_dataset, m, alpha=args.alpha)
        else:
            # 从end-start这么多类中随机抽取task num个类
            # current_class = random.sample([x for x in range(start, end)], task_num)
            # 只保留标签属于 current_class 中的样本。
            # train_dataset = train_dataset.filter(lambda example: example['label'] in current_class)
            # 根据beta程度进行标签采样
            user_groups = distribution_based_label_skew(train_dataset, m, beta=args.beta)
    # plot_user_groups_distribution(args, train_dataset, user_groups)

    # 使用映射前先对键值对进行排序，确保匹配的一致性
    sorted_user_keys = sorted(user_groups.keys())
    sorted_idxs_users = sorted(idxs_users)

    # 映射用户组，确保每个客户端得到正确的数据索引
    user_groups_mapped = {}
    for old_key, new_key in zip(sorted_user_keys, sorted_idxs_users):
        user_groups_mapped[new_key] = user_groups[old_key]

    # 更新用户组
    user_groups = user_groups_mapped

    return train_dataset, user_groups


# 假设 user_groups 是一个字典，其中键是用户ID，值是该用户对应的样本
def plot_user_groups_distribution(args, dataset, user_groups):
    # 获取所有标签
    all_labels = np.array(dataset['labels'])  # 获取完整数据集的所有标签

    # 统计每个用户组的标签分布
    user_label_counts = {}
    for user, indices in user_groups.items():
        label_counts = {}
        # 根据索引获取每个用户的数据标签
        user_labels = all_labels[indices]
        for label in np.unique(user_labels):
            label_counts[label] = np.sum(user_labels == label)
        user_label_counts[user] = label_counts

    # 转换为矩阵，行表示用户，列表示标签
    unique_labels = list(np.unique(all_labels))
    label_matrix = np.array(
        [[user_label_counts[user].get(label, 0) for label in unique_labels] for user in user_label_counts])

    # 绘制热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(label_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=unique_labels,
                yticklabels=user_label_counts.keys())
    title = 'Label Distribution Across Users when beta = {}'.format(args.beta)
    plt.title(title)
    plt.xlabel("Labels")
    plt.ylabel("Users")
    plt.show()


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


def load_json(file_name, encoding="utf-8"):
    with open(file_name, 'r', encoding=encoding) as f:
        content = json.load(f)
    return content


def dump_json(obj, file_name, encoding="utf-8", default=None):
    if default is None:
        with open(file_name, 'w', encoding=encoding) as fw:
            json.dump(obj, fw)
    else:
        with open(file_name, 'w', encoding=encoding) as fw:
            json.dump(obj, fw, default=default)


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
            softmax_data = torch.tensor([1.0 / len(centers_list)] * len(centers_list), dtype=torch.float64,
                                        device=device)

        weight.append(softmax_data.cpu())

    return weight


def average_weights(weights_list, model, classes, niid_type, backbone_weight, numclass):
    trainable_params = get_trainable_param_names(model)

    avg_weights = collections.OrderedDict()
    weight_names = weights_list[0].keys()

    for name in weight_names:
        # 检查是否有 module 前缀
        if name.startswith("module."):
            stripped_name = name[len("module."):]  # 去掉 module 前缀
        else:
            stripped_name = name

        try:
            # 确保与模型的键名对齐
            if stripped_name not in trainable_params:
                if stripped_name in model.state_dict():
                    avg_weights[stripped_name] = model.state_dict()[stripped_name]
            else:
                # 聚合权重并确保在相同设备
                aggregated_weight_tensor = torch.zeros_like(weights_list[0][name])
                for i, w in enumerate(weights_list):
                    aggregated_weight_tensor += w[name] * backbone_weight[i]
                avg_weights[stripped_name] = aggregated_weight_tensor

        except KeyError as e:
            print(f"Warning: Key {name} ({stripped_name}) not found in weights_list or model state_dict. Skipping.")

    return avg_weights


# def average_weights(weights_list, model, classes, niid_type, backbone_weight, numclass):
#     trainable_params = get_trainable_param_names(model)
#
#     avg_weights = collections.OrderedDict()
#     weight_names = weights_list[0].keys()
#
#     for name in weight_names:
#         if name not in trainable_params:
#             if name in model.state_dict():
#                 avg_weights[name] = model.state_dict()[name]
#         else:
#             # 确保所有张量在同一设备上
#             aggregated_weight_tensor = torch.stack(
#                 [w[name] * backbone_weight[i] for i, w in enumerate(weights_list)]).sum(dim=0)
#             avg_weights[name] = aggregated_weight_tensor
#
#     return avg_weights


# def average_weights(weights_list, model, classes, niid_type, backbone_weight, numclass, device='cuda'):
#     # 获取所有可训练的参数名
#     trainable_params = get_trainable_param_names(model)
#
#     avg_weights = collections.OrderedDict()
#     weight_names = weights_list[0].keys()
#
#     for name in weight_names:
#         if name not in trainable_params:
#             if name in model.state_dict():
#                 avg_weights[name] = model.state_dict()[name]
#         else:
#             # 对于可训练的权重，按权重加权聚合
#             aggregated_weight_tensor = torch.zeros_like(weights_list[0][name], device=device)
#
#             # 通过加权平均来聚合权重
#             for i, w in enumerate(weights_list):
#                 aggregated_weight_tensor += w[name].to(device) * backbone_weight[i]
#
#             avg_weights[name] = aggregated_weight_tensor  # 权重加权平均
#
#     return avg_weights

# def global_server(model, global_model, args):
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if args.is_peft:
#                 if 'lora' in name.lower():
#                     if name in global_model.state_dict():
#                         if param.size() == global_model.state_dict()[name].size():
#                             # 确保数据被拷贝到正确的设备
#                             global_model.state_dict()[name].copy_(param.data.to(global_model.state_dict()[name].device))
#                         else:
#                             print(f"Skipping parameter '{name}' due to size mismatch: "
#                                   f"Model size {param.size()} vs Global model size {global_model.state_dict()[name].size()}")
#                     else:
#                         print(f"Skipping parameter '{name}' due to size mismatch: ")
#             else:
#                 if name in global_model.state_dict():
#                     if param.size() == global_model.state_dict()[name].size():
#                         # 确保数据被拷贝到正确的设备
#                         global_model.state_dict()[name].copy_(param.data.to(global_model.state_dict()[name].device))
#                     else:
#                         print(f"Skipping parameter '{name}' due to size mismatch: "
#                               f"Model size {param.size()} vs Global model size {global_model.state_dict()[name].size()}")
#
#     return global_model


def global_server(model, global_model, args):
    # 提取全局模型的 state_dict
    global_model_state_dict = global_model.state_dict()

    for name, param in model.named_parameters():
        if args.is_peft:
            if 'lora' in name.lower():
                if name in global_model_state_dict:
                    if param.size() == global_model_state_dict[name].size():
                        # 避免重复的设备传输
                        if param.device != global_model_state_dict[name].device:
                            global_model_state_dict[name].copy_(param.data.to(global_model_state_dict[name].device))
                        else:
                            global_model_state_dict[name].copy_(param.data)
                    else:
                        print(f"Skipping parameter '{name}' due to size mismatch.")
                else:
                    print(f"Skipping parameter '{name}' due to size mismatch.")
        else:
            if name in global_model_state_dict:
                if param.size() == global_model_state_dict[name].size():
                    # 避免重复的设备传输
                    if param.device != global_model_state_dict[name].device:
                        global_model_state_dict[name].copy_(param.data.to(global_model_state_dict[name].device))
                    else:
                        global_model_state_dict[name].copy_(param.data)
                else:
                    print(f"Skipping parameter '{name}' due to size mismatch.")

    return global_model


# def global_server(model, global_model, args):
#     """
#     更新全局模型的权重，依据 args.is_peft 来判断更新 LoRA 层还是全量更新
#     """
#     with torch.no_grad():
#         print(f"global_model.named_parameters(): {list(global_model.named_parameters())}")  # 查看结构
#         for global_param, (name, param) in zip(global_model.named_parameters(), model.named_parameters()):
#             # 确保global_param是parameter类型
#             if isinstance(global_param, tuple):
#                 global_param = global_param[1]  # 解包tuple，只取参数部分
#             print(f"param type: {type(param)}, global_param type: {type(global_param)}")
#
#             # 判断是否只更新 LoRA 层还是全量更新
#             if args.is_peft:
#                 # 只更新 LoRA 层
#                 if 'lora' in name.lower() or 'classifier' in name.lower():
#                     if param.size() == global_param.size():
#                         global_param.data.copy_(param.data)
#                     else:
#                         print(f"Skipping parameter '{name}' due to size mismatch: "
#                               f"Model size {param.size()} vs Global model size {global_param.size()}")
#             else:
#                 # 全量更新
#                 if param.size() == global_param.size():
#                     global_param.data.copy_(param.data)
#                 else:
#                     print(f"Skipping parameter '{name}' due to size mismatch: "
#                           f"Model size {param.size()} vs Global model size {global_param.size()}")
#
#     return global_model


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
    if args.mode == 'federated':
        print('    Federated parameters:')
        if args.iid:
            print('    IID')
        else:
            print('    Non-IID')
        print(f'    Users in one epoch  : {args.client_local}')
        print(f'    Local Batch size   : {args.local_bs}')
        print(f'    Local Epochs       : {args.local_ep}\n')
        print(f'    Beta               : {args.beta}')
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


def compare_model_params(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(param1.data, param2.data, atol=1e-5):
            return False
    return True


def compare_model_and_weights(model, weights_dict, threshold=1e-6):
    """
    比较模型的权重和一个 OrderedDict 格式的权重字典。

    参数:
    - model (torch.nn.Module): 要比较的 PyTorch 模型。
    - weights_dict (OrderedDict): 以 OrderedDict 格式存储的权重。
    - threshold (float): 判断权重是否相等的阈值。
    """
    model_weights = model.state_dict()

    # 确保模型权重和给定的权重字典都在同一设备上（CPU 或 GPU）
    device = next(iter(model_weights.values())).device
    if not all(weight.device == device for weight in weights_dict.values()):
        weights_dict = {k: v.to(device) for k, v in weights_dict.items()}

    all_equal = True
    for key in model_weights.keys():
        if key not in weights_dict:
            print(f"Key {key} missing in weights_dict")
            all_equal = False
        else:
            model_weight = model_weights[key]
            given_weight = weights_dict[key]

            # 由于浮点数精度问题，直接比较可能不准确，因此计算差异并检查是否小于阈值
            diff = torch.abs(model_weight - given_weight).max().item()
            if diff > threshold:
                print(f"Difference in key {key}: {diff}")
                all_equal = False

    if all_equal:
        print("All weights are equal within the given threshold.")
    else:
        print("Some weights are different.")


def compute_forgetting_rate(task_accuracies, previous_task_accuracies):
    """
    计算每个任务的遗忘度（基于已学任务的准确率衰退）。
    """
    forgetting_rates = []

    # 遍历任务，计算每个任务的遗忘度
    for task_idx in range(1, len(task_accuracies)):
        total_fgt_task = 0  # 当前任务的总遗忘度
        total_categories = 0  # 当前任务的类别数

        # 遍历当前任务与所有前任务之间的准确率变化
        for subtask_idx in range(task_idx):
            current_accuracies = task_accuracies[task_idx]
            previous_accuracies = previous_task_accuracies[subtask_idx]

            # 对于每一个任务的每一类别，计算准确率差异
            for i in range(len(previous_accuracies)):
                total_fgt_task += (previous_accuracies[i] - current_accuracies[i])
                total_categories += 1

        # 当前任务的遗忘度是对已学类别准确率差异的平均值
        task_fgt = total_fgt_task / total_categories  # 每个任务的遗忘度
        forgetting_rates.append(task_fgt)

    # 计算所有任务的平均遗忘度
    total_fgt = np.mean(forgetting_rates)
    return total_fgt


def compute_final_acc(args, centralized_trainer):
    acc = 0
    total_weight = 0
    # 计算 ACC 和 FGT
    task_num = len(centralized_trainer.task_accuracies)
    # 加权计算所有任务的准确率
    for i in range(task_num):
        # 计算每个任务的类别数
        if i == 0:
            task_weight = args.fg_nc
        else:
            task_weight = centralized_trainer.task_size

        task_acc = sum(centralized_trainer.task_accuracies[i]) / len(
            centralized_trainer.task_accuracies[i])  # 当前任务的准确率
        acc += task_acc * task_weight
        total_weight += task_weight

    # 最终加权准确率
    acc /= total_weight
    return acc


def _load_clinc150_data(clinc150_data_path):
    """加载并格式化 clinc150 数据"""
    # 读取 JSON 文件
    with open(clinc150_data_path, 'r') as f:
        clinc150_data = json.load(f)

    # 提取 train 和 test 数据
    clinc150_train = _convert_clinc150_to_dataframe(clinc150_data.get('train', []))
    clinc150_test = _convert_clinc150_to_dataframe(clinc150_data.get('test', []))
    return clinc150_train, clinc150_test


def _convert_clinc150_to_dataframe(data):
    """将 clinc150 数据转换为 DataFrame 格式，保留字符串标签"""
    if not data:
        return pd.DataFrame(columns=['input_text', 'label'])
    texts, labels = zip(*data)  # 解压数据为文本和标签
    return pd.DataFrame({'input_text': texts, 'label': labels})  # 直接保留标签字符串


def _merge_datasets(dataset, clinc_df):
    """合并 datasets.Dataset 和 clinc150 DataFrame"""
    # 转换 clinc_df 为 datasets.Dataset
    clinc_dataset = Dataset.from_pandas(clinc_df)

    # 检查两者的列名是否一致，如果不一致需要统一
    for column in clinc_dataset.column_names:
        if column not in dataset.column_names:
            raise ValueError(f"列 {column} 不在主数据集列中，请检查列名一致性！")
    return concatenate_datasets([dataset, clinc_dataset])


def _load_fewrel_data(fewrel_data_path):
    """加载并格式化 FewRel 数据"""
    with open(fewrel_data_path, 'rb') as f:
        datas = pickle.load(f)

        # 获取训练集、验证集和测试集
        train_dataset, val_dataset, test_dataset = datas

        # 处理训练集数据
        train_texts = []
        train_labels = []
        for group_id, group in enumerate(train_dataset):
            for sample in group:  # 每个group是一个包含420个样本的列表
                train_texts.append(sample['text'])
                train_labels.append(sample['semantic_label'])

        # 处理测试集数据
        test_texts = []
        test_labels = []
        for group_id, group in enumerate(test_dataset):
            for sample in group:  # 每个group是一个包含420个样本的列表
                test_texts.append(sample['text'])
                test_labels.append(sample['semantic_label'])

        # 创建训练集和测试集的字典
        train_data = {
            'text': train_texts,
            'labels': train_labels
        }
        test_data = {
            'text': test_texts,
            'labels': test_labels
        }

        # 将字典转换为 Dataset 对象
        train_data = Dataset.from_dict(train_data)
        test_data = Dataset.from_dict(test_data)

        return train_data, test_data


def _load_trace_data(trace_data_path):
    """加载并格式化 FewRel 数据"""
    with open(trace_data_path, 'rb') as f:
        datas = pickle.load(f)

        # 获取训练集、验证集和测试集
        train_dataset, _, test_dataset = datas

        # 处理训练集数据
        train_texts = []
        train_labels = []
        for group_id, group in enumerate(train_dataset):
            for sample in group:  # 每个group是一个包含420个样本的列表
                train_texts.append(sample['text'])
                train_labels.append(sample['semantic_label'])

        # 处理测试集数据
        test_texts = []
        test_labels = []
        for group_id, group in enumerate(test_dataset):
            for sample in group:  # 每个group是一个包含420个样本的列表
                test_texts.append(sample['text'])
                test_labels.append(sample['semantic_label'])

        # 创建训练集和测试集的字典
        train_data = {
            'text': train_texts,
            'labels': train_labels
        }
        test_data = {
            'text': test_texts,
            'labels': test_labels
        }

        # 将字典转换为 Dataset 对象
        train_data = Dataset.from_dict(train_data)
        test_data = Dataset.from_dict(test_data)

        return train_data, test_data


def prepare_sequence_finetune(args):
    """Prepare a sequence of tasks for class-incremental learning."""
    with open(args.sequence_file.replace('_reduce', ''), 'r') as f:
        datas = f.readlines()[args.idrandom]
        data = datas.split()

    args.task_name = data

    if 'banking77' in args.dataset_name:
        args.ntasks = 7
        args.dataset_name = args.sequence_file.split('/')[-1]
    elif 'clinc150' in args.sequence_file:
        args.ntasks = 15
        args.dataset_name = args.sequence_file.split('/')[-1]
    elif '20news' in args.sequence_file:
        args.ntasks = 10
        args.dataset_name = args.sequence_file.split('/')[-1]
    elif 'fewrel' in args.sequence_file:
        args.ntasks = 8
        args.dataset_name = args.sequence_file.split('/')[-1]
    elif 'tacred' in args.sequence_file:
        args.ntasks = 8
        args.dataset_name = args.sequence_file.split('/')[-1]
    else:
        raise NotImplementedError('The current dataset is not supported!')

    if args.classifier_lr is None:
        args.classifier_lr = args.learning_rate

    if 'ewc' in args.baseline:
        args.lamb = 5000  # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000 for ewc

    output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[args.ft_task]) + "_model/"
    # 前一个任务的ckpt
    ckpt = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[args.ft_task - 1]) + "_model/"

    if args.ft_task > 0 and 'mtl' not in args.baseline:
        args.prev_output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
            args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[args.ft_task - 1]) + "_model/"
    else:
        args.prev_output = ''
    args.task = args.ft_task

    args.output_dir = output

    args.saved_output_dir = [args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[t]) + "_model/" for t in range(args.ft_task + 1)]

    if args.task == 0:  # Load the pre-trained model.
        if 'bart-base' in args.baseline:
            args.model_name_or_path = '/home/qiuwenqi/LLM/models/bart-base'  # Use the local backup.
        elif 'bart-large' in args.baseline:
            args.model_name_or_path = '/home/qiuwenqi/LLM/models/bart-large'
        else:
            raise NotImplementedError('Currently, we only support BART as the backbone model.')

    else:
        args.model_name_or_path = ckpt

    print('saved_output_dir: ', args.saved_output_dir)
    print('output_dir: ', args.output_dir)
    print('prev_output: ', args.prev_output)
    print('args.dataset_name: ', args.dataset_name)
    print('args.model_name_or_path: ', args.model_name_or_path)

    return args


def configure_logging(args):
    if args.mode == 'centralized':
        keyname = '/output/logs-Centralized-1216' + '/{}'.format(args.dataset)
        if args.is_peft:
            nam = "lora"
            args.store_name = '_'.join(
                [args.dataset, args.model, args.mode, nam, 'epoch-' + str(args.epochs), 'lr-' + str(args.encoders_lr),
                 'r-' + str(args.r), args.baseline])
        else:
            nam = "full-finetune"
            args.store_name = '_'.join(
                [args.dataset, args.model, args.mode, nam, 'epoch-' + str(args.epochs), 'lr-' + str(args.encoders_lr)
                 ])
    elif args.mode == "federated":
        if args.type == 'iid':
            keyname = '/output/logs-Federated' + "/iid" + '/{}'.format(args.dataset)
        else:
            keyname = '/output/logs-Federated-1216' + "/non-iid" + '/{}'.format(args.dataset)
        if args.is_peft:
            nam = "FCL-lora"
            if args.type == 'iid':
                args.store_name = '_'.join(
                    [args.dataset, args.model, args.mode, args.type, nam,
                     'epoch-' + str(args.epochs), 'lr-' + str(args.encoders_lr),
                     'r-' + str(args.r), "iid"])
            else:
                args.store_name = '_'.join(
                    [args.dataset, args.model, args.mode, args.type, nam,
                     'epoch-' + str(args.epochs), 'lr-' + str(args.encoders_lr),
                     'r-' + str(args.r), "beta-" + str(args.beta)])
        else:
            nam = "FCL-full"
            args.store_name = '_'.join([args.dataset, args.model, args.mode,
                                        args.type, nam, 'epoch-' + str(args.epochs), 'lr-' + str(args.encoders_lr),
                                        "beta-" + str(args.beta)])
    return keyname


def update_args(args):
    current_task = args.task
    output = args.base_dir + f"/seq_{str(args.idrandom)}_seed{str(args.seed)}" + "/" + str(
        args.baseline) + '/' + str(args.dataset) + '/' + 'task_' + str(current_task) + "_model/"
    # 前一个任务的ckpt路径
    ckpt = args.base_dir + f"/seq_{str(args.idrandom)}_seed{str(args.seed)}" + "/" + str(
        args.baseline) + '/' + str(args.dataset) + '/' + 'task_' + str(current_task - 1) + "_model/"
    # 前一个任务的输出
    if current_task > 0 and 'mtl' not in args.baseline:
        args.prev_output = args.base_dir + f"/seq_{str(args.idrandom)}_seed{str(args.seed)}" + "/" + str(
            args.baseline) + '/' + str(args.dataset) + '/' + 'task_' + str(current_task - 1) + "_model/"
    else:
        args.prev_output = ''

    args.output_dir = output

    args.saved_output_dir = [args.base_dir + f"/seq_{str(args.idrandom)}_seed{str(args.seed)}" + "/" + str(
        args.baseline) + '/' + str(args.dataset) + '/' + 'task_' + str(t) + "_model/" for t in range(current_task + 1)]

    if args.task == 0:  # Load the pre-trained model.
        if 'bart-base' in args.baseline:
            args.model_name_or_path = '/home/qiuwenqi/LLM/models/bart-base'  # Use the local backup.
        elif 'bart-large' in args.baseline:
            args.model_name_or_path = '/home/qiuwenqi/LLM/models/bart-large'
        else:
            raise NotImplementedError('Currently, we only support BART as the backbone model.')

    else:
        args.model_name_or_path = ckpt


def initialize_model(args):
    # 创建模型
    if args.task == 0:
        model = LLMWithLoRA(
            modelname=args.model_name_or_path,
            is_peft=args.is_peft,
            num_classes=args.total_classes,
            r=args.r,
            # lora_layer=["query", "value"]
        )
        # model = model.to(args.device)
        data_collator = model.data_collator
        tokenizer = model.tokenizer
        if 'bart' in args.baseline:
            if 'distill' in args.baseline or 'ewc' in args.baseline:
                teacher = LLMWithLoRA(
                    modelname=args.model_name_or_path,
                    is_peft=args.is_peft,
                    num_classes=args.total_classes,
                    r=args.r,
                    # lora_layer=["query", "value"]
                )
                for param in teacher.parameters():
                    param.requires_grad = False
                model = MyBart(model, teacher=teacher, args=args)
            elif 'classification' in args.baseline:
                model = MyBart(model, args=args)

    else:
        ckpt_path = os.path.join(args.model_name_or_path, "mybart_checkpoint.pt")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")  # 加载保存的 checkpoint
            if 'bart' in args.baseline:
                if 'distill' in args.baseline or 'ewc' in args.baseline:
                    base_model = LLMWithLoRA(
                        modelname="/home/qiuwenqi/LLM/models/bart-base",  # 基础模型路径
                        is_peft=args.is_peft,
                        num_classes=args.total_classes,
                        r=args.r,
                    )
                    data_collator = base_model.data_collator
                    tokenizer = base_model.tokenizer
                    teacher = LLMWithLoRA(
                        modelname="/home/qiuwenqi/LLM/models/bart-base",
                        is_peft=args.is_peft,
                        num_classes=args.total_classes,
                        r=args.r,
                        # lora_layer=["query", "value"]
                    )
                    for param in teacher.parameters():
                        param.requires_grad = False
                    model = MyBart(base_model, teacher=teacher, args=args)
                    model.load_state_dict(checkpoint["state_dict"], strict=False)  # 加载权重
                    print(f"Loaded MyBart model from {ckpt_path}")
                elif 'classification' in args.baseline:
                    model = LLMWithLoRA(
                        modelname="/home/qiuwenqi/LLM/models/bart-base",
                        is_peft=args.is_peft,
                        num_classes=args.total_classes,
                        r=args.r,
                        # lora_layer=["query", "value"]
                    )
                    data_collator = model.data_collator
                    tokenizer = model.tokenizer
                    model = MyBart(model, args=args)
                    model.load_state_dict(checkpoint["state_dict"], strict=False)  # 加载权重
                    print(f"Loaded MyBart model from {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    if args.is_peft:
        # 固定所有参数，只更新lora参数
        for name, param in model.model.named_parameters():
            param.requires_grad = False
        for name, param in model.model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
    else:
        # full fine tune
        for param in model.model.parameters():
            param.requires_grad = True


    return model, data_collator, tokenizer



def compare_all_model_parameters(model_before, model_after):
    """
    比较两个模型的权重，检查所有参数层是否一致。

    Args:
        model_before (torch.nn.Module): 保存前的模型。
        model_after (torch.nn.Module): 加载后的模型。

    Returns:
        bool: 如果所有参数一致返回 True，否则返回 False。
    """
    # 获取两个模型的 state_dict
    state_dict_before = model_before.state_dict()
    state_dict_after = model_after.state_dict()

    # 遍历所有参数层
    print("开始检查所有参数层是否一致：")
    for layer in state_dict_before.keys():
        param_before = state_dict_before[layer]
        param_after = state_dict_after[layer]

        # 判断参数是否相等
        if not torch.equal(param_before, param_after):
            print(f"❌ 参数 {layer} 不一致")
            return False  # 发现不一致，立即返回

    print("✅ 所有参数层一致，加载正确")
    return True


def before_train_utils(current_task, classes_cache, total_classes):
    classes = classes_cache.get(current_task)

    # Update class range
    if classes[1] > total_classes:
        classes[1] = total_classes

    if classes[0] >= total_classes:
        print("All tasks completed. Stopping training.")
        return classes, True  # Return classes and a flag indicating all tasks are completed

    return classes, False  # Return classes and False if tasks are not completed


def strip_module_prefix(state_dict):
    """
    移除所有键名中的 'module.' 前缀（如果存在），并保持为 OrderedDict。
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def add_module_prefix(state_dict):
    """
    为所有键名添加 'module.' 前缀（如果尚未添加），并保持为 OrderedDict。
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            new_key = 'module.' + k
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict