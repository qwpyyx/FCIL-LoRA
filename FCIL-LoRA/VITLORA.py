import shutil
import os
from VLT import *
from utils import *
from update import *
from tqdm import tqdm
from CPN import *
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from datasets import concatenate_datasets, Dataset
import json
import pandas as pd
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
class vitlora:
    def __init__(self, args, file_name, model, task_size, device):
        # 数据集路径设置为新的banking77数据集路径
        self.all_tasks_completed = None
        self.data_dir = '/home/qiuwenqi/LLM/Datasets/banking77'  # 假设数据集文件已经上传到此路径
        self.file_name = file_name
        self.args = args
        self.epochs = args.local_ep
        self.model = model
        self.global_model = model
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device

        self.task_accuracies = []  # 保存每个任务结束后的最佳准确率
        self.previous_task_accuracies = []  # 保存每次任务开始前对所有已学任务的准确率
        self.list_of_individual_testloader = []  # 保存每个任务的独立测试集加载器

        self._load_datasets()

        # 数据转换
        self.data_collator = DataCollatorWithPadding(tokenizer=self.global_model.tokenizer)
        self.classes = None
        self.old_model = None
        self.list_of_testloader = list()
        self.W_aq = []
        self.W_av = []
        self.W_bq = []
        self.W_bv = []

        # # 定义 Trainer 参数
        # self.training_args = TrainingArguments(
        #     output_dir=f'./results',  # 输出目录
        #     num_train_epochs=self.args.epochs,  # 训练 epoch 数
        #     per_device_train_batch_size=self.args.local_bs,  # 每设备训练 batch 大小
        #     per_device_eval_batch_size=self.args.local_bs,  # 每设备评估 batch 大小
        #     warmup_ratio=0.1,  # 预热比例
        #     weight_decay=0.01,  # 权重衰减
        #     logging_dir='./logs',  # 日志目录
        #     logging_steps=10,  # 日志打印频率
        #     save_steps=100,  # 模型保存频率
        #     evaluation_strategy="epoch",  # 每个 epoch 后进行评估
        #     save_strategy="epoch",  # 每个 epoch 后保存模型（与评估策略一致）
        #     load_best_model_at_end=True,  # 训练结束时加载最佳模型
        #     metric_for_best_model="accuracy",  # 选择最佳模型的标准
        # )
        #
        # # 构建 Trainer（初始化时构建一次）
        # self.trainer = Trainer(
        #     model=self.model,  # 使用当前模型
        #     args=self.training_args,  # 训练参数
        #     train_dataset=None,  # 训练集（将在后续更新）
        #     eval_dataset=None,  # 验证集（将在后续更新）
        #     data_collator=self.data_collator,  # 数据处理器
        #     compute_metrics=self.compute_metrics,  # 计算评估指标
        # )

    def _load_datasets(self):
        """加载数据集并进行预处理"""
        dataset = load_dataset('csv',
                               data_files={'train': f"{self.data_dir}/train.csv",
                                           'test': f"{self.data_dir}/test.csv"},
                               delimiter=',')  # 确保读取CSV格式，指定分隔符

        # 重命名列
        dataset = dataset.rename_column(original_column_name='text', new_column_name='input_text')
        dataset = dataset.rename_column(original_column_name='category', new_column_name='label')

        # 分割训练集和验证集
        self.train_set = dataset['train']
        self.test_set = dataset['test']

        if self.args.combine:
            # 加载并合并 clinc150 数据集
            clinc150_train, clinc150_test = self._load_clinc150_data(
                clinc150_data_path='/home/qiuwenqi/LLM/Datasets/clinc150/data_full.json'
            )

            # 合并数据集
            self.train_set = self._merge_datasets(self.train_set, clinc150_train)
            self.test_set = self._merge_datasets(self.test_set, clinc150_test)

    def _load_clinc150_data(self, clinc150_data_path):
        """加载并格式化 clinc150 数据"""
        # 读取 JSON 文件
        with open(clinc150_data_path, 'r') as f:
            clinc150_data = json.load(f)

        # 提取 train 和 test 数据
        clinc150_train = self._convert_clinc150_to_dataframe(clinc150_data.get('train', []))
        clinc150_test = self._convert_clinc150_to_dataframe(clinc150_data.get('test', []))
        return clinc150_train, clinc150_test

    def _convert_clinc150_to_dataframe(self, data):
        """将 clinc150 数据转换为 DataFrame 格式，保留字符串标签"""
        if not data:
            return pd.DataFrame(columns=['input_text', 'label'])
        texts, labels = zip(*data)  # 解压数据为文本和标签
        return pd.DataFrame({'input_text': texts, 'label': labels})  # 直接保留标签字符串

    def _merge_datasets(self, dataset, clinc_df):
        """合并 datasets.Dataset 和 clinc150 DataFrame"""
        # 转换 clinc_df 为 datasets.Dataset
        clinc_dataset = Dataset.from_pandas(clinc_df)

        # 检查两者的列名是否一致，如果不一致需要统一
        for column in clinc_dataset.column_names:
            if column not in dataset.column_names:
                raise ValueError(f"列 {column} 不在主数据集列中，请检查列名一致性！")

        # 合并数据集
        return concatenate_datasets([dataset, clinc_dataset])

    def beforeTrain_raw(self, current_task):
        # 获取训练前的基线准确率
        print("Computing previous task accuracies...")

        # 设置当前任务的类别范围
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

        print(f"train_class is {self.classes[0]} to {self.classes[1]}")
        print(f"test_class is 0 to {self.classes[1]}")

        # 筛选当前任务相关的训练集样本，只包含当前任务的新类别
        self.train_set = self.preprocessed_train_set.filter(
            lambda example: self.classes[0] <= example['label'] < self.classes[1])
        self.train_dataset = DatasetSplit(self.train_set, list(range(len(self.train_set))))
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.args.local_bs, shuffle=True, num_workers=0,
            collate_fn=self.data_collator
        )

        # self.train_set.rename_column('label', 'labels')

        # 筛选测试集样本，包含所有已学类别（从任务0到当前任务的所有类别）
        self.test_set = self.preprocessed_test_set.filter(lambda example: 0 <= example['label'] < self.classes[1])
        self.test_dataset = DatasetSplit(self.test_set, list(range(len(self.test_set))))
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.args.local_bs, shuffle=False, num_workers=0,
            collate_fn=self.data_collator
        )
        # self.test_set.rename_column('label', 'labels')
        # self.trainer.train_dataset = self.train_set
        # self.trainer.eval_dataset = self.test_set

        print(f"Eval dataset sample: {next(iter(self.test_dataset))}")

        # 确保字段完整性
        print(f"Sample from train_set: {self.train_dataset[0]}")
        # print(f"Sample from test_set: {self.test_dataset[0]}")

        # # 更新 Trainer 的数据集
        # if hasattr(self, "trainer"):
        #     self.trainer.train_dataset = self.train_set
        #     self.trainer.eval_dataset = self.test_set


        # 处理当前任务的测试集
        # current_test_set = self.test_set.filter(
        #     lambda example: self.classes[0] <= example['label'] < self.classes[1])
        # self.list_of_individual_testloader.append(current_test_set)
        #
        # # 计算之前任务的准确率
        # if current_task > 0:
        #     previous_acc = []
        #     for test_loader in self.list_of_individual_testloader:
        #         eval_results = self.trainer.evaluate(test_loader)
        #         acc = eval_results['eval_accuracy']
        #         previous_acc.append(acc)
        #     self.previous_task_accuracies.append(previous_acc)

        # 将模型和设备准备好
        self.model.train()
        self.model.to(self.device)

    def raw_train(self, current_task, old_class=0, tf_writer=None, logger_file=None):
        """集中式增量式训练的实现"""
        description = "Centralized Training Task={}, Epoch={}, acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=-1) if predictions.ndim > 1 else predictions
            if len(predictions) != len(labels):
                min_len = min(len(predictions), len(labels))
                predictions = predictions[:min_len]
                labels = labels[:min_len]
            return {"accuracy": accuracy_score(labels, predictions)}

        training_args = TrainingArguments(
            output_dir=f"./results/task_{current_task}",
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.local_bs,
            per_device_eval_batch_size=self.args.local_bs,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.test_set,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )
        # TODO 这个有点问题，要想弄好，应该要自己定义train，特别是改trainer类中的prediction_step
        # 现在的方法是通过compute_metric做了一个权衡
        trainer.label_names = ['labels']

        print(f"Length of eval_dataset: {len(self.test_dataset)}")
        print(next(iter(self.test_dataset)))

        # 使用trainer进行训练
        trainer.train()

        # 评估
        eval_results = trainer.evaluate()
        test_acc = eval_results["eval_accuracy"]  # 获取评估的准确率
        test_loss = eval_results["eval_loss"]  # 获取评估的损失

        # 打印结果
        print(f"Test Accuracy for Task {current_task}: {test_acc:.4f}%")

        # 将测试集的 ACC 和损失写入日志
        if tf_writer:
            tf_writer.add_scalar(f'Task_{current_task}/test_acc', test_acc)
            tf_writer.add_scalar(f'Task_{current_task}/test_loss', test_loss)

        if logger_file:
            output_log = f'Task {current_task} - Test acc: {test_acc:.4f}, Test loss: {test_loss:.4f}'
            logger_file.write(output_log + '\n')
            logger_file.flush()

        return test_acc

        # # TODO 注意encoders_lr非常影响效果
        # network_params = []
        #
        # # for name, param in self.model.named_parameters():
        # #     if param.requires_grad == True:
        # #         print(name)
        #
        # if self.args.is_peft:
        #     for name, param in self.model.named_parameters():
        #         if 'lora' in name.lower() and param.requires_grad:
        #             network_params.append({'params': param, 'lr': self.args.encoders_lr, 'weight_decay': 0.00001})
        # else:
        #     for param in self.model.parameters():
        #         network_params.append({'params': param, 'lr': self.args.encoders_lr, 'weight_decay': 0.00001})
        #
        # # 构建 Adam 优化器
        # optimizer = torch.optim.Adam(network_params)
        #
        # # 对当前任务进行训练
        # for epoch in range(self.args.epochs):
        #     # pre_train_params = {name: param.clone() for name, param in self.model.named_parameters()}
        #
        #     self.model.train()
        #
        #     total_loss = 0
        #     total_correct = 0
        #     num_samples = 0
        #
        #     # 使用 train_loader 进行训练
        #     for batch_idx, batch in enumerate(self.train_loader):
        #         inputs = {
        #         'input_ids': batch['input_ids'].to(self.device),
        #         'attention_mask': batch['attention_mask'].to(self.device),
        #         'labels': batch['labels'].to(self.device)
        #         }
        #
        #         # 清空梯度
        #         optimizer.zero_grad()
        #
        #         # 前向传播
        #         logits = self.model(**inputs)
        #
        #         # 计算分类交叉熵损失
        #         loss_fct = torch.nn.CrossEntropyLoss()
        #         loss = loss_fct(logits, inputs['labels'])
        #
        #         # 反向传播和优化
        #         loss.backward()
        #         optimizer.step()
        #
        #         # 统计损失和准确率
        #         total_loss += loss.item() * len(inputs['labels'])
        #         _, preds = torch.max(logits, dim=1)
        #         total_correct += (preds == inputs['labels']).sum().item()
        #         num_samples += len(inputs['labels'])
        #
        #     # 计算平均损失和准确率
        #     avg_loss = total_loss / num_samples
        #     avg_acc = 100.0 * total_correct / num_samples
        #     bst_acc = max(bst_acc, avg_acc)
        #
        #     # 打印当前的任务进度
        #     print(description.format(current_task, epoch + 1, avg_acc, avg_loss, bst_acc))
        #
        #     # 保存当前模型
        #     is_best = avg_acc > bst_acc
        #     self.save_checkpoint(self.model.state_dict(), is_best)
        #
        #     # 将训练过程中的 ACC 和损失写入日志
        #     if tf_writer:
        #         tf_writer.add_scalar(f'Task_{current_task}/train_acc', avg_acc, epoch)
        #         tf_writer.add_scalar(f'Task_{current_task}/train_loss', avg_loss, epoch)
        #
        #     if logger_file:
        #         output_log = 'After {} epochs, Train acc: {:.4f}, Train loss: {:.4f}'.format(epoch + 1, avg_acc,
        #                                                                                      avg_loss)
        #         logger_file.write(output_log + '\n')
        #         logger_file.flush()
        #
        # # 在测试集上评估
        # test_acc, test_loss = self.inference(self.model, self.test_loader)
        # print(f"Test Accuracy for Task {current_task}: {test_acc:.4f}%")
        #
        # # 将测试集的 ACC 和损失写入日志
        # if tf_writer:
        #     tf_writer.add_scalar(f'Task_{current_task}/test_acc', test_acc)
        #     tf_writer.add_scalar(f'Task_{current_task}/test_loss', test_loss)
        #
        # if logger_file:
        #     output_log = 'Task {} - Test acc: {:.4f}, Test loss: {:.4f}'.format(current_task, test_acc, test_loss)
        #     logger_file.write(output_log + '\n')
        #     logger_file.flush()
        #
        # return test_acc

    def preprocess_test_set(self):
        # 定义预处理函数
        def preprocess_function(examples):
            tokenized_inputs = self.global_model.tokenizer(
                examples["input_text"], padding="max_length", truncation=True, max_length=128
            )
            tokenized_inputs["label"] = examples["label"]  # 确保标签被保留
            return tokenized_inputs

        # 对整个测试集进行一次性预处理
        self.test_set = self.test_set.map(preprocess_function, batched=True, remove_columns=["input_text"])
        self.train_set = self.train_set.map(preprocess_function, batched=True, remove_columns=["input_text"])
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
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                # 兼容处理标签字段名
                labels = batch.get('labels') if 'labels' in batch else batch.get('label')
                labels = labels.to(self.device)

                # 模型推理
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

                # 检查 logits 的形状是否正确
                # print(f"Logits shape: {logits.shape}, Expected number of classes: {self.numclass}")

                # 计算分类交叉熵损失
                loss = loss_fct(logits, labels)
                test_loss += loss.item()

                # 计算预测结果
                pred = torch.argmax(logits, dim=1)
                # print(f"Prediction shape: {pred.shape}, Labels shape: {labels.shape}")

                correct += pred.eq(labels.view_as(pred)).sum().item()

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

    def beforeTrain(self, current_task):

        if current_task == 0:
            self.classes = [0, min(self.numclass, self.total_classes)]
        else:
            # self.classes = [0, min(self.numclass + current_task * self.task_size, self.total_classes)]
            # self.classes = [self.numclass, min(self.numclass + self.task_size, self.total_classes)]
            self.classes = [0, min((current_task + 1) * self.task_size, self.total_classes)]

        print("Now is training task {}, and self.classes is {}.".format(current_task, self.classes))

        if self.classes[1] > self.total_classes:
            self.classes[1] = self.total_classes

        if self.classes[0] >= self.total_classes:
            print("All tasks completed. Stopping training.")
            self.all_tasks_completed = True  # 标志任务已经全部完成
            return

        # 筛选当前任务相关的样本
        self.test_set = self.preprocessed_test_set.filter(
            lambda example: self.classes[0] <= example['label'] < self.classes[1])

        # 创建 DatasetSplit 和 DataLoader
        self.test_dataset = DatasetSplit(self.test_set, list(range(len(self.test_set))))
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.args.local_bs, shuffle=False, num_workers=0,
            collate_fn=self.data_collator
        )
        self.list_of_testloader.append(self.test_loader)

        # 获取训练前的基线准确率
        if current_task > 0:
            previous_acc = []
            for i, test_loader in enumerate(self.list_of_individual_testloader):
                acc, _ = self.inference(self.global_model, test_loader)
                previous_acc.append(acc)
            self.previous_task_accuracies.append(previous_acc)

        # 增加新的任务独立测试集加载器
        # 为当前任务创建独立的测试集加载器，仅包含当前任务的类别
        start_class = current_task * self.task_size
        end_class = min((current_task + 1) * self.task_size, self.total_classes)
        print(f"Creating individual test loader for task {current_task} with classes {start_class} to {end_class}.")

        current_test_set = self.preprocessed_test_set.filter(
            lambda example: start_class <= example['label'] < end_class)
        test_dataset = DatasetSplit(current_test_set, list(range(len(current_test_set))))
        individual_test_loader = DataLoader(
            test_dataset, batch_size=self.args.local_bs, shuffle=False, num_workers=0,
            collate_fn=self.data_collator
        )
        self.list_of_individual_testloader.append(individual_test_loader)

        self.model.train()
        self.model.to(self.device)
        self.global_model.to(self.device)

    def train(self, current_task, old_class=0, tf_writer=None, logger_file=None):
        bst_acc = -1
        description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
        # local_weights = []
        center_lr = self.args.centers_lr
        encoder_lr = self.args.encoders_lr

        # best_acc = 0.0
        # early_stopping_patience = 10
        # current_patience = 0

        for epoch in tqdm(range(self.args.epochs)):
            local_weights = []
            # count_num= [[] for i in range(0, self.task_size)]
            # feature_list = [[] for i in range(0, self.task_size)]
            sample_num = []
            m = self.args.client_local
            # 每一轮随机从num_users中选取m个客户端参与训练
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
            # load dataset and user groups
            train_dataset, user_groups = get_dataset(self.args, train_dataset=self.train_set, m=m,
                                                     start=self.classes[0], end=self.classes[1],
                                                     task_num=self.task_size)
            # 确保筛选后的数据集不为空
            assert len(train_dataset) > 0, "Filtered train dataset is empty. Please check the filtering criteria."

            # 检查用户组和选择的客户端数量是否一致
            if len(user_groups) != len(idxs_users):
                raise ValueError(f"Mismatch in the number of users and selected indices for remapping. "
                                 f"User groups: {len(user_groups)}, Selected clients: {len(idxs_users)}")

            # 使用映射前先对键值对进行排序，确保匹配的一致性
            sorted_user_keys = sorted(user_groups.keys())
            sorted_idxs_users = sorted(idxs_users)

            # 映射用户组，确保每个客户端得到正确的数据索引
            user_groups_mapped = {}
            for old_key, new_key in zip(sorted_user_keys, sorted_idxs_users):
                user_groups_mapped[new_key] = user_groups[old_key]

            # 打印映射结果以便调试和验证
            # print(f"User groups mapping: {user_groups_mapped}")

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

                unique_labels = set(train_dataset['label'])

                # Local Epoch
                w, _ = local_model.update_weights(
                    model=copy.deepcopy(self.model), old_model=copy.deepcopy(self.old_model), lr_c=center_lr,
                    lr_e=encoder_lr,
                    Waq=self.W_aq, Wav=self.W_av, unique_labels=unique_labels)

                # local model
                local_weights.append(copy.deepcopy(w))
                # 对于第key类的values
                # for cls, values in feature_average.items():
                #     # 确保 key 是在类别范围内
                #     if 0 <= cls < self.numclass:
                #         # feature_list里面的第一层0表示第0类。第二层的0表示第0个节点的第0类的平均特征
                #         feature_list[cls].append(values)
                #     else:
                #         print(f"Warning: The Class {cls} is out of range for numclass {self.numclass}")

            average_weight = [i / sum(sample_num) for i in sample_num]

            # # update global weights
            # self.model.load_state_dict(self.model)
            self.model.load_state_dict(average_weights(local_weights, self.model, self.classes,
                                                       self.args.niid_type, average_weight, self.numclass))
            self.global_model = global_server(self.model, self.global_model)
            # self.model_ave.load_state_dict(average_weights2(local_weights, self.model))

            test_acc, test_loss = self.inference(self.global_model, self.test_loader)

            # center_lr, encoder_lr = center_lr*0.95, encoder_lr*0.95
            # if epoch < self.args.epochs/2:
            #     center_lr = self.args.centers_lr
            #     encoder_lr = self.args.encoders_lr
            # else:
            center_lr = self.args.centers_lr * 0.5 * (1 + math.cos(epoch * math.pi / self.args.epochs))
            encoder_lr = self.args.encoders_lr * 0.5 * (1 + math.cos(epoch * math.pi / self.args.epochs))

            tf_writer.add_scalar('test_acc', test_acc, epoch)
            tf_writer.add_scalar('test_loss', test_loss, epoch)

            output_log = 'After {} global rounds, Test acc: {}, inference loss: {}'.format(
                epoch + 1, test_acc, test_loss)
            logger_file.write(output_log + '\n')
            logger_file.flush()

            is_best = test_acc > bst_acc
            bst_acc = max(bst_acc, test_acc)
            # print(description.format(test_acc, test_loss, bst_acc))

            self.save_checkpoint(self.model.state_dict(), is_best)

        print(description.format(test_acc, test_loss, bst_acc))

    def afterTrain(self, current_task):
        # 更新类别数量
        self.numclass = min((current_task + 1) * self.task_size, self.total_classes)

        # 在每个任务结束后保存每个任务的独立测试集上的准确率
        current_acc = []
        for i, test_loader in enumerate(self.list_of_individual_testloader):
            acc, _ = self.inference(self.global_model, test_loader)
            current_acc.append(acc)
        self.task_accuracies.append(current_acc)

        # 创建保存模型的目录
        path = os.path.join(self.args.save_path, self.file_name)
        if not os.path.isdir(path):
            os.makedirs(path)

        if self.numclass >= self.total_classes:
            print("Reached the maximum number of classes. Training complete.")
            return

        # 保存当前的全局模型
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        torch.save(self.global_model.state_dict(), filename)

        # 更新旧模型为最新的训练后模型
        self.old_model = copy.deepcopy(self.global_model)
        self.old_model.load_state_dict(torch.load(filename, map_location=self.device))
        self.old_model.to(self.device)
        self.old_model.eval()
        print(100 * '#')
        print(" had done {} tasks".format(current_task))

    def afterTrain_raw(self, current_task, logger_file=None):
        # 获取当前任务的类别范围
        start_class, end_class = self.classes  # 直接复用 self.classes

        # 在每个任务结束后保存每个任务的独立测试集上的准确率
        current_acc = []
        for i, test_loader in enumerate(self.list_of_individual_testloader):
            acc, _ = self.inference(self.model, test_loader)
            current_acc.append(acc)

            if logger_file:
                logger_file.write(f"Task {current_task}'s current Test acc: {acc:.4f}\n")
                logger_file.flush()

        self.task_accuracies.append(current_acc)

        # 创建保存模型的目录
        path = os.path.join(self.args.save_path, self.file_name)
        if not os.path.isdir(path):
            os.makedirs(path)

        # 检查是否完成所有任务
        if end_class >= self.total_classes:
            print("Reached the maximum number of classes. Training complete.")
            return

        # 保存当前的模型
        filename = os.path.join(path, f'{end_class - self.task_size}_model.pkl')
        torch.save(self.model.state_dict(), filename)

        print(100 * '#')
        print(f"Completed training task {current_task}")
