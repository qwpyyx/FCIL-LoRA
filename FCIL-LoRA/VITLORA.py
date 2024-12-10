import shutil
import os
from VLT import *
from utils import *
from update import *
from tqdm import tqdm
from CPN import *
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import json
import pandas as pd
from replay import ExperienceReplay
import math
import matplotlib.pyplot as plt
import pickle
from data import get_dataset
from utils import _load_clinc150_data, _load_fewrel_data

class vitlora:
    def __init__(self, args, file_name, model, task_size, device):
        self.all_tasks_completed = None
        self.data_dir = '/home/qiuwenqi/LLM/Datasets/banking77'  # 假设数据集文件已经上传到此路径
        self.file_name = file_name
        self.args = args
        self.epochs = args.local_ep
        self.model = model
        self.global_model = copy.deepcopy(model)
        # self.global_model = LLMWithLoRA(
        #                                 modelname=self.args.model_path,
        #                                 is_peft=self.args.is_peft,
        #                                 num_classes=self.args.total_classes,
        #                                 r=self.args.r,
        #                                 lora_layer=["query", "value"]
        #                                 )
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = self.args.device

        self.task_accuracies = []  # 保存每个任务结束后的最佳准确率
        self.previous_task_accuracies = []  # 保存每次任务开始前对所有已学任务的准确率
        self.list_of_individual_testloader = []  # 保存每个任务的独立测试集加载器

        self._load_datasets()

        # Replay
        self.experience_replay = ExperienceReplay(max_size=50, old_data_ratio=self.args.old_data_replay_ratio)

        # 数据转换
        self.data_collator = DataCollatorWithPadding(tokenizer=self.global_model.tokenizer)
        self.classes = None
        self.old_model = None
        self.list_of_testloader = list()
        self.W_aq = []
        self.W_av = []
        self.W_bq = []
        self.W_bv = []

    def _load_datasets(self):
        if "banking" in self.args.dataset:
            print("Using data from banking 77 dataset")
            dataset = load_dataset('csv',
                                   data_files={'train': f"{self.data_dir}/train.csv",
                                               'test': f"{self.data_dir}/test.csv"},
                                   delimiter=',')  # 确保读取CSV格式，指定分隔符
            # 重命名列
            dataset = dataset.rename_column(original_column_name='text', new_column_name='input_text')
            dataset = dataset.rename_column(original_column_name='category', new_column_name='label')

            #dataset = get_dataset("fewrel", tokenizer=None, args=self.args)

            # 分割训练集和验证集
            self.train_set = dataset['train']
            self.test_set = dataset['test']

        elif "clinc" in self.args.dataset:

            print("Using data from clinc150 dataset")
            # 加载并合并 clinc150 数据集
            clinc150_train, clinc150_test = _load_clinc150_data(
                clinc150_data_path='/home/qiuwenqi/LLM/Datasets/clinc150/data_full.json'
            )

            self.train_set = clinc150_train
            self.test_set = clinc150_test

        elif "fewrel" in self.args.dataset:
            print("Using data from FewRel dataset")

            # dataset = get_dataset("fewrel", tokenizer=None, args=self.args)

            # 加载 fewrel 数据集
            fewrel_train, fewrel_test = _load_fewrel_data(
                fewrel_data_path='/home/qiuwenqi/LLM/Datasets/FewRel/FewRel-2021.pkl'
            )
            # dataset_train = fewrel_train.rename_column(original_column_name='text', new_column_name='input_text')
            # dataset_test = fewrel_test.rename_column(original_column_name='labels', new_column_name='label')

            fewrel_train = fewrel_train.rename_column('text', 'input_text')
            fewrel_train = fewrel_train.rename_column('labels', 'label')

            fewrel_test = fewrel_test.rename_column('text', 'input_text')
            fewrel_test = fewrel_test.rename_column('labels', 'label')

            self.train_set = fewrel_train
            self.test_set = fewrel_test

        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

    def beforeTrain_raw(self, current_task, logger_file=None):

        # 如果没有提前计算并缓存过类别范围，则进行计算
        if not hasattr(self, 'classes_cache'):
            self.classes_cache = {}
            for task in range(self.args.total_num):  # 假设任务总数为8
                if task == 0:
                    start_class = 0  # 第一个任务的类别从0开始
                    end_class = self.args.fg_nc  # 第一个任务的类别范围为 [0, fg_nc]
                else:
                    start_class = self.args.fg_nc + (task - 1) * self.task_size  # 后续任务的起始类别
                    end_class = min(self.args.fg_nc + task * self.task_size, self.total_classes)  # 后续任务的结束类别
                self.classes_cache[task] = [start_class, end_class]

        # 从缓存中获取当前任务的类别范围
        self.classes = self.classes_cache.get(current_task)

        if self.classes[1] > self.total_classes:
            self.classes[1] = self.total_classes

        if self.classes[0] >= self.total_classes:
            print("All tasks completed. Stopping training.")
            self.all_tasks_completed = True
            return

        # 预先筛选训练集和测试集
        if not hasattr(self, 'task_train_sets'):  # 如果没有预先缓存训练集
            print("Preprocessing all task's train and test sets...")
            self.task_train_sets = {}
            self.task_test_sets = {}
            self.current_test_set = {}

            for task in range(self.args.total_num):  # 假设任务总数为8
                # 设置当前任务的类别范围
                if task == 0:
                    start_class = 0  # 第一个任务的类别从0开始
                    end_class = self.args.fg_nc  # 第一个任务的类别范围为 [0, fg_nc]
                else:
                    start_class = self.args.fg_nc + (task - 1) * self.task_size  # 后续任务的起始类别
                    end_class = min(self.args.fg_nc + task * self.task_size, self.total_classes)

                self.task_train_sets[task] = self.preprocessed_train_set.filter(
                    lambda example: start_class <= example['label'] < end_class
                )

                self.current_test_set[task] = self.preprocessed_test_set.filter(
                    lambda example: start_class <= example['label'] < end_class
                )

                self.task_test_sets[task] = self.preprocessed_test_set.filter(
                    lambda example: 0 <= example['label'] < end_class
                )

        print(f"Now is training task {current_task}")
        print(f"train_class is {self.classes[0]} to {self.classes[1]}")
        print(f"test_class is 0 to {self.classes[1]}")

        # 获取当前任务的训练集和测试集
        self.train_set = self.task_train_sets[current_task]
        self.test_set = self.task_test_sets[current_task]
        self.current_test = self.current_test_set[current_task]

        self.train_dataset = DatasetSplit(self.train_set, list(range(len(self.train_set))))
        self.test_dataset = DatasetSplit(self.test_set, list(range(len(self.test_set))))
        self.current_dataset = DatasetSplit(self.current_test, list(range(len(self.current_test))))

        # 创建数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.local_bs, shuffle=True, num_workers=0,
                                       collate_fn=self.data_collator)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.local_bs, shuffle=True, num_workers=0,
                                      collate_fn=self.data_collator)

        individual_test_loader = DataLoader(
            self.current_dataset, batch_size=self.args.local_bs, shuffle=True,
            num_workers=0, collate_fn=self.data_collator
        )
        self.list_of_individual_testloader.append(individual_test_loader)

        # 计算前任务的准确率
        if current_task > 0:
            print("Computing previous task accuracies...")
            previous_acc = []
            for i, test_loader in enumerate(self.list_of_individual_testloader):
                acc, _ = self.inference(self.model, test_loader)
                previous_acc.append(acc)
            print('Task {} previous accuracies: {}'.format(current_task - 1, previous_acc))
            self.previous_task_accuracies.append(previous_acc)

            if logger_file:
                logger_file.write(f"Task {current_task - 1}'s previous accuracies: {previous_acc}\n")
                logger_file.flush()

        # 将模型和设备准备好
        self.model.train()
        self.model.to(self.device)

    # def beforeTrain_raw(self, current_task, logger_file=None):
    #     # 获取训练前的基线准确率
    #     print("Computing previous task accuracies...")
    #
    #     # 设置当前任务的类别范围
    #     if current_task == 0:
    #         self.classes = [0, self.args.fg_nc]  # 第一个任务
    #     else:
    #         self.classes = [self.args.fg_nc + (current_task - 1) * self.task_size,
    #                         min(self.args.fg_nc + current_task * self.task_size, self.total_classes)]  # 后续任务
    #
    #     print(f"Now is training task {current_task}")
    #
    #     if self.classes[1] > self.total_classes:
    #         self.classes[1] = self.total_classes
    #
    #     if self.classes[0] >= self.total_classes:
    #         print("All tasks completed. Stopping training.")
    #         self.all_tasks_completed = True
    #         return
    #
    #     print(f"train_class is {self.classes[0]} to {self.classes[1]}")
    #     print(f"test_class is 0 to {self.classes[1]}")
    #
    #     # 筛选当前任务相关的训练集样本，只包含当前任务的新类别
    #     self.train_set = self.preprocessed_train_set.filter(
    #         lambda example: self.classes[0] <= example['label'] < self.classes[1])
    #     self.train_dataset = DatasetSplit(self.train_set, list(range(len(self.train_set))))
    #     self.train_loader = DataLoader(
    #         self.train_dataset, batch_size=self.args.local_bs, shuffle=True, num_workers=0,
    #         collate_fn=self.data_collator
    #     )
    #
    #     # 筛选测试集样本，包含所有已学类别（从任务0到当前任务的所有类别）
    #     self.test_set = self.preprocessed_test_set.filter(lambda example: 0 <= example['label'] < self.classes[1])
    #     self.test_dataset = DatasetSplit(self.test_set, list(range(len(self.test_set))))
    #     self.test_loader = DataLoader(
    #         self.test_dataset, batch_size=self.args.local_bs, shuffle=True, num_workers=0,
    #         collate_fn=self.data_collator
    #     )
    #
    #     # start_class, end_class = self.classes
    #
    #     current_test_set = self.preprocessed_test_set.filter(
    #         lambda example: self.classes[0] <= example['label'] < self.classes[1])
    #     test_dataset = DatasetSplit(current_test_set, list(range(len(current_test_set))))
    #
    #     individual_test_loader = DataLoader(
    #         test_dataset, batch_size=self.args.local_bs, shuffle=True,
    #         num_workers=0, collate_fn=self.data_collator
    #     )
    #     self.list_of_individual_testloader.append(individual_test_loader)
    #
    #     if current_task > 0:
    #         previous_acc = []
    #         for i, test_loader in enumerate(self.list_of_individual_testloader):
    #             acc, _ = self.inference(self.model, test_loader)
    #             previous_acc.append(acc)
    #         self.previous_task_accuracies.append(previous_acc)
    #
    #         if logger_file:
    #             logger_file.write(f"Task {current_task - 1}'s previous accuracies: {previous_acc}\n")
    #             logger_file.flush()
    #
    #     # 将模型和设备准备好
    #     self.model.train()
    #     self.model.to(self.device)


    def raw_train(self, current_task, old_class=0, tf_writer=None, logger_file=None):
        """集中式增量式训练的实现"""
        bst_acc = -1
        description = "Centralized Training Task={}, Epoch={}, acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"

        # TODO 注意encoders_lr非常影响效果
        network_params = []
        if self.args.is_peft:
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    network_params.append({'params': param, 'lr': self.args.encoders_lr})
        else:
            for param in self.model.parameters():
                network_params.append({'params': param, 'lr': self.args.encoders_lr})

        optimizer = torch.optim.Adam(network_params)
        loss_fct = torch.nn.CrossEntropyLoss()

        print(100 * '#')
        print("Begin Training!")
        # Train
        for epoch in range(self.args.epochs):

            total_loss = 0
            total_correct = 0
            num_samples = 0

            # 从经验回放中采样旧任务数据
            # 采样旧任务的数据
            if self.args.is_replay:
                if self.experience_replay.size() > 0:
                    old_data = self.experience_replay.sample(500)  # 采样旧任务的数据

                    # 从采样的 old_data 中提取数据并转换为 tensor
                    input_ids = []
                    attention_mask = []
                    labels = []

                    for data in old_data:
                        # data 是一个字典，直接提取对应的值
                        input_ids.append(torch.tensor(data['input_ids']).to(self.device))  # 转为 tensor
                        attention_mask.append(torch.tensor(data['attention_mask']).to(self.device))  # 转为 tensor
                        labels.append(torch.tensor(data['label']).to(self.device))  # 转为 tensor

                    # 将所有的 input_ids, attention_mask, labels 拼接成一个批次
                    input_ids = torch.stack(input_ids)
                    attention_mask = torch.stack(attention_mask)
                    labels = torch.tensor(labels).to(self.device)  # 确保标签也在同一设备

                    # 将这些数据封装成一个批次字典
                    old_batch = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': labels
                    }

                    # 直接使用上述的批次数据进行训练
                    optimizer.zero_grad()
                    logits = self.model(**old_batch)
                    loss = loss_fct(logits, old_batch['labels'])
                    loss.backward()
                    optimizer.step()

            # 使用 train_loader 进行训练
            for batch_idx, batch in enumerate(self.train_loader):
                inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'labels': batch['labels'].to(self.device)
                }

                optimizer.zero_grad()
                logits = self.model(**inputs)
                loss = loss_fct(logits, inputs['labels'])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(inputs['labels'])
                _, preds = torch.max(logits, dim=1)
                total_correct += (preds == inputs['labels']).sum().item()
                num_samples += len(inputs['labels'])

            avg_loss = total_loss / num_samples
            avg_acc = 100.0 * total_correct / num_samples
            bst_acc = max(bst_acc, avg_acc)

            print(description.format(current_task, epoch + 1, avg_acc, avg_loss, bst_acc))

            # 保存当前模型
            # is_best = avg_acc > bst_acc
            # self.save_checkpoint(self.model.state_dict(), is_best)

            # 将训练过程中的 ACC 和损失写入日志
            if tf_writer:
                tf_writer.add_scalar(f'Task_{current_task}/train_acc', avg_acc, epoch)
                tf_writer.add_scalar(f'Task_{current_task}/train_loss', avg_loss, epoch)

            if logger_file:
                output_log = 'After {} epochs, Train acc: {:.4f}, Train loss: {:.4f}'.format(epoch + 1, avg_acc,
                                                                                             avg_loss)
                logger_file.write(output_log + '\n')
                logger_file.flush()

        # 在测试集上评估
        test_acc, test_loss = self.inference(self.model, self.test_loader)
        print(f"Test Accuracy for Task {current_task}: {test_acc:.4f}%")

        # 将测试集的 ACC 和损失写入日志
        if tf_writer:
            tf_writer.add_scalar(f'Task_{current_task}/test_acc', test_acc)
            tf_writer.add_scalar(f'Task_{current_task}/test_loss', test_loss)

        if logger_file:
            output_log = f'Task {current_task} - Test acc: {test_acc:.4f}, Test loss: {test_loss:.4f}'
            logger_file.write(output_log + '\n')
            logger_file.flush()

        # self.task_accuracies.append([test_acc])

        return test_acc

    def preprocess_test_set_FL(self):
        # 定义预处理函数
        def preprocess_function(examples):
            return self.global_model.tokenizer(
                examples['input_text'],
                padding='max_length',
                truncation=True,
                max_length=128
            )

        # 对整个测试集进行一次性预处理
        self.test_set = self.test_set.map(preprocess_function, batched=True)
        # self.train_set = self.train_set.map(preprocess_function, batched=True)
        # 备份预处理后的完整测试集
        self.preprocessed_test_set = self.test_set
        # self.preprocessed_train_set = self.train_set

    def preprocess_test_set(self):
        # 定义预处理函数
        def preprocess_function(examples):
            return self.global_model.tokenizer(
                examples['input_text'],
                padding='max_length',
                truncation=True,
                max_length=128
            )

        # 对整个测试集进行一次性预处理
        self.test_set = self.test_set.map(preprocess_function, batched=True)
        self.train_set = self.train_set.map(preprocess_function, batched=True)
        # 备份预处理后的完整测试集
        self.preprocessed_test_set = self.test_set
        self.preprocessed_train_set = self.train_set

    def inference(self, model, test_loader):
        model.eval()
        test_loss = 0.0
        correct = 0.0
        # extracted_features = []
        # extracted_label = []

        loss_fct = torch.nn.CrossEntropyLoss()  # 分类损失函数

        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'labels': batch['labels'].to(self.device)
                }

                # 模型推理
                logits = model(**inputs)

                # 检查 logits 的形状是否正确
                # print(f"Logits shape: {logits.shape}, Expected number of classes: {self.numclass}")

                # 计算分类交叉熵损失
                loss = loss_fct(logits, inputs['labels'])
                test_loss += loss.item()

                # 计算预测结果
                pred = torch.argmax(logits, dim=1)
                # print(f"Prediction shape: {pred.shape}, Labels shape: {labels.shape}")

                correct += pred.eq(inputs['labels'].view_as(pred)).sum().item()

        test_loss /= len(test_loader)  # 计算平均损失
        acc = 100. * correct / len(test_loader.dataset)

        return acc, test_loss

    def save_checkpoint(self, state, is_best):
        # 拼接路径
        checkpoint_dir = os.path.join(
            os.path.abspath(os.path.dirname(os.getcwd())) + 'PILoRA-cifar' + '/checkpoints_epoch',
            self.args.store_name
        )

        # 确保目录存在
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 保存模型状态
        filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
        torch.save(state, filename)

        # 如果是最佳模型，则复制到新的文件
        if is_best:
            best_filename = filename.replace('pth.tar', 'best.pth.tar')
            shutil.copyfile(filename, best_filename)

    def map_new_class_index(self, labels, class_order):
        # 创建一个标签到新类索引的映射字典
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(class_order)}

        # 使用映射字典对标签进行映射
        mapped_labels = [label_mapping[label] for label in labels]

        return mapped_labels

    def setup_data(self, shuffle):
        # 获取训练集和测试集的类别标签
        train_targets = self.train_set['label']
        test_targets = self.test_set['label']

        # 获取训练集和测试集中的所有唯一标签，确保包含所有类别
        unique_classes = sorted(set(train_targets + test_targets))

        self.total_classes = len(unique_classes)

        # 创建类别的顺序 (直接使用唯一标签，而不是整数)
        if shuffle:
            class_order = np.random.permutation(unique_classes).tolist()  # 打乱后的标签顺序
        else:
            class_order = unique_classes  # 保持默认的排序顺序

        self.class_order = class_order

        # 创建一个标签到新类索引的映射字典
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(self.class_order)}

        # 使用 map() 更新数据集标签
        self.train_set = self.train_set.map(lambda example: {'label': label_mapping[example['label']]})
        self.test_set = self.test_set.map(lambda example: {'label': label_mapping[example['label']]})

    def beforeTrain(self, current_task, logger_file):

        if current_task == 0:
            self.classes = [0, self.args.fg_nc]  # 第一个任务
        else:
            self.classes = [self.args.fg_nc + (current_task - 1) * self.task_size,
                            min(self.args.fg_nc + current_task * self.task_size, self.total_classes)]  # 后续任务

        print(f"Now is training task {current_task}")

        if self.classes[1] > self.total_classes:
            self.classes[1] = self.total_classes

        if self.classes[0] >= self.total_classes:
            print("All tasks completed. Stopping training.")
            self.all_tasks_completed = True
            return

        # print(f"train_class is {self.classes[0]} to {self.classes[1]}")
        print(f"test_class is 0 to {self.classes[1]}")

        # 筛选测试集样本，包含所有已学类别（从任务0到当前任务的所有类别）
        self.test_set = self.preprocessed_test_set.filter(lambda example: 0 <= example['label'] < self.classes[1])
        self.test_dataset = DatasetSplit(self.test_set, list(range(len(self.test_set))))
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.args.local_bs, shuffle=True, num_workers=0,
            collate_fn=self.data_collator
        )

        current_test_set = self.preprocessed_test_set.filter(
            lambda example: self.classes[0] <= example['label'] < self.classes[1])
        test_dataset = DatasetSplit(current_test_set, list(range(len(current_test_set))))

        individual_test_loader = DataLoader(
            test_dataset, batch_size=self.args.local_bs, shuffle=True,
            num_workers=0, collate_fn=self.data_collator
        )

        self.list_of_individual_testloader.append(individual_test_loader)

        # 训好模型后，对之前的所有任务单独测试当前全局模型对之前任务的性能
        if current_task > 0:
            previous_acc = []
            for i, test_loader in enumerate(self.list_of_individual_testloader):
                acc, _ = self.inference(self.global_model, test_loader)
                previous_acc.append(acc)
            self.previous_task_accuracies.append(previous_acc)

            if logger_file:
                logger_file.write(f"Task {current_task - 1}'s previous accuracies: {previous_acc}\n")
                logger_file.flush()

        self.model.train()
        self.model.to(self.device)
        self.global_model.to(self.device)

    def train(self, current_task, logger_file=None):
        bst_acc = -1
        description = "Global inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
        # local_weights = []
        # center_lr = self.args.centers_lr
        encoder_lr = self.args.encoders_lr

        # test_en = []
        # best_acc = 0.0
        # early_stopping_patience = 10
        # current_patience = 0

        for epoch in tqdm(range(self.args.epochs)):
            local_weights = []
            sample_num = []
            m = self.args.client_local
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)

            # load dataset and user groups
            train_dataset, user_groups = get_dataset_noniid(self.args, train_dataset=self.train_set, m=m,
                                                     start=self.classes[0], end=self.classes[1],
                                                     task_num=self.task_size)

            # 使用映射前先对键值对进行排序，确保匹配的一致性
            sorted_user_keys = sorted(user_groups.keys())
            sorted_idxs_users = sorted(idxs_users)

            # 映射用户组，确保每个客户端得到正确的数据索引
            user_groups_mapped = {}
            for old_key, new_key in zip(sorted_user_keys, sorted_idxs_users):
                user_groups_mapped[new_key] = user_groups[old_key]

            # 更新用户组
            user_groups = user_groups_mapped

            for idx in idxs_users:
                # 每个客户端使用其对应的索引
                local_data_indices = user_groups[idx]
                # 每个节点样本数
                sample_num.append(len(user_groups[idx]))
                # 加载每个节点的模型
                local_model = LocalUpdate(args=self.args, dataset=train_dataset,
                                          idxs=local_data_indices, tokenizer=self.global_model.tokenizer
                                          )

                # Local Epoch
                w, _ = local_model.update_weights(
                    model=copy.deepcopy(self.model), lr=encoder_lr)

                # local model
                local_weights.append(copy.deepcopy(w))

            average_weight = [i / sum(sample_num) for i in sample_num]

            # # 使用此函数检查self.global_model是否更新
            # if compare_model_params(self.model, self.global_model):
            #     print("global_model and model are identical!")
            # else:
            #     print("global_model and model have different parameters.")


            agg_weights = average_weights(local_weights, self.model, self.classes,
                                                       self.args.niid_type, average_weight, self.numclass)

            # compare_model_and_weights(self.model, agg_weights)


            self.model.load_state_dict(agg_weights)

            # if compare_model_params(self.model, self.global_model):
            #     print("global_model and model are identical!")
            # else:
            #     print("global_model and model have different parameters.")
            #
            #
            #
            #



            self.global_model = global_server(self.model, self.global_model, self.args)

            # if compare_model_params(self.model, self.global_model):
            #     print("global_model and model are identical!")
            # else:
            #     print("global_model and model have different parameters.")



            test_acc, test_loss = self.inference(self.global_model, self.test_loader)

            # Use decay learning rate
            encoder_lr = self.args.encoders_lr * (1 + math.cos(epoch * math.pi / self.args.epochs)) / 2

            output_log = 'After {} global rounds, Test acc: {}, inference loss: {}'.format(
                epoch + 1, test_acc, test_loss)
            logger_file.write(output_log + '\n')
            logger_file.flush()

            is_best = test_acc > bst_acc
            bst_acc = max(bst_acc, test_acc)

        print(description.format(test_acc, test_loss, bst_acc))

        if logger_file:
            output_log = f'Task {current_task} - Test acc: {test_acc:.4f}, Test loss: {test_loss:.4f}'
            logger_file.write(output_log + '\n')
            logger_file.flush()

    def afterTrain(self, current_task, logger_file):
        # 更新类别数量
        # self.numclass = min((current_task + 1) * self.task_size, self.total_classes)

        # if self.args.is_replay:
        #     print(f"Adding task {current_task}'s data to experience replay...")
        #     # 将当前任务数据保存到经验回放缓存中
        #     self.experience_replay.add(self.train_dataset)

        # 在每个任务结束后保存每个任务的独立测试集上的准确率
        current_acc = []
        for i, test_loader in enumerate(self.list_of_individual_testloader):
            acc, _ = self.inference(self.global_model, test_loader)
            current_acc.append(acc)

        if logger_file:
            logger_file.write(f"Task {current_task}'s current accuracies: {current_acc}\n")
            logger_file.flush()

        self.task_accuracies.append(current_acc)
        # # 创建保存模型的目录
        # path = os.path.join(self.args.save_path, self.file_name)
        # if not os.path.isdir(path):
        #     os.makedirs(path)

        if self.numclass >= self.total_classes:
            print("Reached the maximum number of classes. Training complete.")
            return

        # 保存当前的全局模型
        # filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        # torch.save(self.global_model.state_dict(), filename)

        # 更新旧模型为最新的训练后模型
        # self.old_model = copy.deepcopy(self.global_model)
        # self.old_model.load_state_dict(torch.load(filename, map_location=self.device))
        # self.old_model.to(self.device)
        # self.old_model.eval()
        print(100 * '#')
        print(" had done {} tasks".format(current_task))

    def afterTrain_raw(self, current_task, logger_file=None):

        if self.args.is_replay:
            print(f"Adding task {current_task}'s data to experience replay...")
            # 将当前任务数据保存到经验回放缓存中
            self.experience_replay.add(self.train_dataset)

        start_class, end_class = self.classes

        # 在每个任务结束后保存每个任务的独立测试集上的准确率
        current_acc = []

        print("Begin Training Current Task...")
        for i, test_loader in enumerate(self.list_of_individual_testloader):
            acc, _ = self.inference(self.model, test_loader)
            current_acc.append(acc)
        print('Task {} current accuracies: {}'.format(current_task, current_acc))

        if logger_file:
            logger_file.write(f"Task {current_task}'s current accuracies: {current_acc}\n")
            logger_file.flush()

        self.task_accuracies.append(current_acc)

        # 创建保存模型的目录
        # path = os.path.join(self.args.save_path, self.file_name)
        # if not os.path.isdir(path):
        #     os.makedirs(path)

        # 检查是否完成所有任务
        if end_class >= self.total_classes:
            print("Reached the maximum number of classes. Training complete.")
            return

        # 保存当前的模型
        # filename = os.path.join(path, f'{end_class - self.task_size}_model.pkl')
        # torch.save(self.model.state_dict(), filename)

        print(100 * '#')
        print(f"Completed training task {current_task}")
