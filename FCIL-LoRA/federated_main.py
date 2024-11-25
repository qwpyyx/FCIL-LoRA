#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# Revised by Adonis

import os
import warnings

warnings.filterwarnings('ignore')
from tensorboardX import SummaryWriter
from options import args_parser
from utils import exp_details
from VLT import *
from VITLORA import vitlora
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_folders(cur_path):
    folders_util = [
        os.path.join(cur_path + '/logs-roberta-large-LoRA', args.store_name),
        os.path.join(cur_path + '/checkpoints', args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder, exist_ok=True)


if __name__ == '__main__':

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    task_size = int((args.total_classes - args.fg_nc) / args.task_num)  # number of classes in each incremental task
    args.type = 'iid' if args.iid == 1 else 'non-iid'
    if args.mode == 'centralized':
        args.store_name = '_'.join(
            [args.dataset, args.model, args.mode, 'lr-' + str(args.centers_lr)])
    else:
        args.store_name = '_'.join(
            [args.dataset, args.model, args.mode, args.type, 'lr-' + str(args.centers_lr)])

    cur_path = os.path.join(os.path.abspath(os.getcwd()), 'PILoRA-cifar')
    prepare_folders(cur_path)
    exp_details(args)
    setup_seed(args.seed)

    # BUILD MODEL
    file_name = args.store_name
    class_set = list(range(args.total_classes))

    # 这里改成LLM,而且是已经有了lora的架构
    # 应该是config 和 model， peft
    model = LLMWithLoRA(modelname=args.model_path,  # 可以选择适合的LLM，例如't5-base'或其他预训练模型
                        num_classes=args.total_classes,
                        r=args.r,
                        lora_layer=["query", "value"])  # 指定 LoRA 作用的层
    model = model.to(args.device)

    # --------------------
    if args.mode == 'federated':
        global_model = vitlora(args, file_name, model, task_size, args.device)
        global_model.setup_data(shuffle=True)

        # 固定所有参数，只更新lora参数
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

        num_params = np.sum([p.numel() for p in model.parameters() if p.requires_grad or not p.requires_grad])
        learnable_params = np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])
        print('# of learnable params: {}; total params: {}; Proportion: {:.4f}%'.format(learnable_params, num_params, (
                learnable_params / num_params) * 100))

        # 初始化测试集和验证集
        global_model.preprocess_test_set()

        # 每个任务
        for i in range(args.task_num + 1):
            # 是否是第一个任务
            old_class = 0 if i == 0 else len(class_set[:args.fg_nc + (i - 1) * task_size])

            filename = 'log_task{}.txt'.format(i)
            logger_file = open(os.path.join(cur_path + '/logs-roberta-large-LoRA', args.store_name, filename), 'w')
            tf_writer = SummaryWriter(log_dir=os.path.join(cur_path + '/logs-roberta-large-LoRA', args.store_name))

            # 执行任务相关的初始化
            global_model.beforeTrain(i)
            # 训练和后处理
            # 如果任务已完成，则跳出循环
            if global_model.all_tasks_completed:
                break

            global_model.train(i, old_class=old_class, tf_writer=tf_writer, logger_file=logger_file)
            global_model.afterTrain(i)

        # 计算 ACC 和 FGT
        task_num = len(global_model.task_accuracies)
        # 计算 ACC：对每个任务的准确率求和，再除以任务数和任务大小
        acc = sum([sum(task_acc) for task_acc in global_model.task_accuracies]) / (task_num * global_model.task_size)

        # 计算 FGT
        fgt = 0
        if task_num > 1:
            for i in range(1, task_num):
                for j in range(i):
                    fgt += global_model.previous_task_accuracies[i - 1][j] - global_model.task_accuracies[i][j]
            fgt /= (task_num - 1)

        print(f"Final Average Accuracy (ACC): {acc:.4f}%")
        print(f"Final Forgetting (FGT): {fgt:.4f}%")

        # 将 ACC 和 FGT 记录到 TensorBoard
        tf_writer.add_scalar('Final/ACC', acc)
        tf_writer.add_scalar('Final/FGT', fgt)

        # 将 ACC 和 FGT 写入日志文件
        output_log = f"Final Average Accuracy (ACC): {acc:.4f}%, Final Forgetting (FGT): {fgt:.4f}%\n"
        logger_file.write(output_log)
        logger_file.flush()

        # 关闭日志文件和 TensorBoard writer
        logger_file.close()
        tf_writer.close()

    # ------------------------------------------------------------------------------------
    elif args.mode == 'centralized':
        # 初始化模型
        centralized_model = LLMWithLoRA(modelname=args.model_path,
                                        num_classes=args.total_classes,
                                        r=args.r,
                                        lora_layer=["query", "value"])
        centralized_model = centralized_model.to(args.device)

        # 固定所有参数，只更新lora参数
        for name, param in centralized_model.named_parameters():
            param.requires_grad = False
        for name, param in centralized_model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

        # 集中式方法
        print("Starting centralized raw method training...")

        # 初始化数据
        centralized_trainer = vitlora(args, file_name, centralized_model, task_size, args.device)
        centralized_trainer.setup_data(shuffle=True)

        centralized_trainer.preprocess_test_set()

        # 每个任务的训练过程
        for i in range(args.task_num + 1):
            # 设置当前任务的数据和类别（集中式场景使用 beforeTrain_raw）
            centralized_trainer.beforeTrain_raw(i)
            if centralized_trainer.all_tasks_completed:
                break

            # 训练
            filename = 'log_task_raw{}.txt'.format(i)
            logger_file = open(os.path.join(cur_path + '/logs-roberta-large-LoRA', args.store_name, filename), 'w')
            tf_writer = SummaryWriter(log_dir=os.path.join(cur_path + '/logs-roberta-large-LoRA', args.store_name))

            centralized_trainer.raw_train(current_task=i, old_class=0, tf_writer=tf_writer, logger_file=logger_file)
            centralized_trainer.afterTrain_raw(current_task=i)

        # 计算 ACC 和 FGT
        task_num = len(centralized_trainer.task_accuracies)
        # 计算 ACC：对每个任务的准确率求和，再除以任务数和任务大小
        acc = sum([sum(task_acc) for task_acc in centralized_trainer.task_accuracies]) / (task_num * centralized_trainer.task_size)

        # 计算 FGT
        fgt = 0
        if task_num > 1:
            for i in range(1, task_num):
                for j in range(i):
                    fgt += centralized_trainer.previous_task_accuracies[i - 1][j] - centralized_trainer.task_accuracies[i][j]
            fgt /= (task_num - 1)

        print(f"Final Average Accuracy (ACC): {acc:.4f}%")
        print(f"Final Forgetting (FGT): {fgt:.4f}%")

        # 将 ACC 和 FGT 记录到 TensorBoard
        tf_writer.add_scalar('Final/ACC', acc)
        tf_writer.add_scalar('Final/FGT', fgt)

        # 将 ACC 和 FGT 写入日志文件
        output_log = f"Final Average Accuracy (ACC): {acc:.4f}%, Final Forgetting (FGT): {fgt:.4f}%\n"
        logger_file.write(output_log)
        logger_file.flush()

        # 关闭日志文件和 TensorBoard writer
        logger_file.close()
        tf_writer.close()

