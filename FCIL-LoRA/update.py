#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from CPN import *
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DataCollatorWithPadding
from transformers import Trainer



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        # dataset 是一个包含多字段（如 'input_ids', 'attention_mask', 'label'）的 DatasetDict 对象
        self.dataset = dataset.select(idxs)
        # 直接存储所有标签以便后续使用
        self.labels = [self.dataset[i]['label'] for i in range(len(self.dataset))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 获取当前索引处的数据，并确保它是一个字典类型，包含所有字段
        example = self.dataset[index]

        return {
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'label': example['label']
        }

    def get_all_labels(self):
        return self.labels


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, tokenizer):
        self.args = args

        # 使用 tokenizer 对数据集进行编码，然后再将数据集传给 DatasetSplit
        dataset = dataset.map(
            lambda example: tokenizer(
                example['input_text'],
                truncation=True,
                padding='max_length',
                max_length=128
            ),
            batched=True
        )

        # 然后创建 DatasetSplit 子集
        self.client_dataset = DatasetSplit(dataset, idxs)
        # print(f"Length of DatasetSplit: {len(self.dataset)}")
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # 使用 DataLoader 加载数据集
        self.trainloader = DataLoader(self.client_dataset, batch_size=self.args.local_bs, shuffle=True, num_workers=0,
                                      collate_fn=self.data_collator)

    def update_weights(self, model, old_model, lr_c, lr_e, Waq, Wav, unique_labels):
        # print(model)
        # pg = [p for p in model.parameters() if p.requires_grad]
        # train_name = [name for name, param in model.named_parameters() if param.requires_grad]

        model.train()

        # 在 update_weights 中重新构建网络参数列表和优化器
        network_params = []
        for name, param in model.named_parameters():
            # 判断哪些层需要更新，只更新 LoRA 部分和确保 requires_grad 为 True
            if 'lora' in name.lower() and param.requires_grad:
                lr = lr_e
                param_group = {'params': [param], 'lr': lr, 'weight_decay': 0.00001}
                network_params.append(param_group)

        # 使用这些参数构建优化器
        self.optimizer = torch.optim.Adam(network_params)

        # Local epoch
        for iter in range(self.args.local_ep):
            for batch_idx, batch in enumerate(self.trainloader):
                input_ids = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                labels = batch['labels'].to(self.args.device)
                # decoder_input_ids = labels
                self.optimizer.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

                # Test
                # logits = logits[:, self.classes[0]:self.classes[1]]

                pred = torch.argmax(logits, dim=1)

                # 计算分类交叉熵损失（假设这是 NLP 分类任务）
                loss_fct = torch.nn.CrossEntropyLoss()
                loss_dce = loss_fct(logits, labels)

                # 计算 LoRA 的 L1 正则化损失
                # l1_loss = torch.tensor(0., dtype=torch.float32, device=self.args.device)
                # for name, param in model.named_parameters():
                #     if 'lora' in name.lower() and param.requires_grad:
                #         l1_loss += torch.norm(param, p=1)

                # 总损失
                # if old_model is None:
                #     loss = loss_dce
                # else:
                #     # 正交损失部分仍然适用，略作调整
                #     # ort_loss = torch.tensor(0., dtype=torch.float32, device=self.args.device)
                #     # for name, param in model.named_parameters():
                #     #     if 'lora_A' in name.lower() and param.requires_grad:
                #     #         for pre_param in Waq + Wav:
                #     #             ort_loss += torch.abs(torch.mm(pre_param, param.T)).sum()
                #     pass
                #
                #     # loss = loss_dce + 0.01 * l1_loss + 0.5 * ort_loss
                #     loss = loss_dce

                loss_dce.backward()
                self.optimizer.step()

            # 打印更新后的 lora_A 参数
            #     for name, param in model.named_parameters():
            #         if 'lora_B' in name:
            #             print(f"Value of {name} after optimizer step: {param[0][0].item()}")
            #             break  # 打印一个 lora_A 层后跳出循环
            #     for name, param in model.named_parameters():
            #         if 'lora_A' in name:
            #             print(f"Value of {name} after optimizer step: {param[0][0].item()}")
            #             break  # 打印一个 lora_A 层后跳出循环
                # pass
                # 记录特征，用于计算每个类别的平均特征
                # features = hidden_states[:, 0, :]  # 假设取 encoder 最后一层的第一个 token 的隐藏状态作为特征
                # # 第i个样本的特征，对应的是第lbl类，把他存进去
                # for i, lbl in enumerate(labels):
                #     lbl = lbl.item()
                #     label_feature_dict[lbl].append(features[i].detach().cpu().numpy())

        # 计算每个类别的平均特征
        # averages = {}
        # # 对于第cls类的特征值value
        # for cls, values in label_feature_dict.items():
        #     # 如果这一类一个样本都没抽到，就没有特征值
        #     if not values:
        #         averages[cls] = []
        #     else:
        #         average_value = sum(values) / len(values)
        #         averages[cls] = average_value

        return model.state_dict(), None
