#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# Revised by Adonis

import os
import warnings

os.environ["WANDB_MODE"] = "disabled"
warnings.filterwarnings('ignore')
from tensorboardX import SummaryWriter
from options import args_parser
from utils import exp_details
from VLT import *
from VITLORA import vitlora
import random
import numpy as np
import deepspeed
from utils import compute_forgetting_rate


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_folders(cur_path, keyname):
    folders_util = [
        os.path.join(cur_path + keyname, args.store_name)]
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
        keyname = '/logs-Centralized' + '/{}'.format(args.dataset)
        if args.is_peft:
            nam = "lora"
            args.store_name = '_'.join(
                [args.dataset, args.model, args.mode, nam, 'epoch-' + str(args.epochs), 'lr-' + str(args.encoders_lr)])
        else:
            nam = "full-finetune"
            args.store_name = '_'.join(
                [args.dataset, args.model, args.mode, nam, 'epoch-' + str(args.epochs), 'lr-' + str(args.encoders_lr),
                 'r-' + str(args.r)])
    elif args.mode == "federated":
        keyname = '/logs-Federated' + '/{}'.format(args.dataset)
        if args.is_peft:
            nam = "FCL-lora"
            args.store_name = '_'.join(
                [args.dataset, args.model, args.mode, args.type, nam,
                 'epoch-' + str(args.epochs), 'lr-' + str(args.encoders_lr),
                 'r-' + str(args.r), "beta-" + str(args.beta)])
        else:
            nam = "FCL-full"
            args.store_name = '_'.join([args.dataset, args.model, args.mode,
                                        args.type, nam, 'epoch-' + str(args.epochs), 'lr-' + str(args.encoders_lr),
                                        "beta-" + str(args.beta)])

    cur_path = os.path.join(os.path.abspath(os.getcwd()), 'PILoRA-cifar')
    prepare_folders(cur_path, keyname)
    exp_details(args)
    setup_seed(args.seed)

    # BUILD MODEL
    file_name = args.store_name
    class_set = list(range(args.total_classes))

    model = LLMWithLoRA(modelname=args.model_path,  # 可以选择适合的LLM，例如't5-base'或其他预训练模型
                        is_peft=args.is_peft,
                        num_classes=args.total_classes,
                        r=args.r,
                        lora_layer=["query", "value"])  # 指定 LoRA 作用的层
    model = model.to(args.device)

    # DeepSpeed初始化
    if args.deepspeed:
        model, optimizer, _, _ = deepspeed.initialize(args=args, model=model)

    # --------------------
    if args.mode == 'federated':

        print("Beginning federated learning...")
        print("Whether use peft? {}".format(args.is_peft))

        num_params = np.sum([p.numel() for p in model.parameters() if p.requires_grad or not p.requires_grad])
        learnable_params = np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])
        print('# of learnable params: {}; total params: {}; Proportion: {:.4f}%'.format(learnable_params, num_params, (
                learnable_params / num_params) * 100))

        global_model = vitlora(args, file_name, model, task_size, args.device)
        global_model.setup_data(shuffle=True)
        global_model.preprocess_test_set_FL()

        old_class = 0

        # 每个任务
        for i in range(args.task_num + 1):

            filename = 'log_task_raw{}.txt'.format(i)
            logger_file = open(os.path.join(cur_path + keyname, args.store_name, filename), 'w')
            tf_writer = SummaryWriter(log_dir=os.path.join(cur_path + keyname, args.store_name))

            global_model.beforeTrain(i, logger_file=logger_file)
            if global_model.all_tasks_completed:
                break

            global_model.train(i, old_class=old_class, tf_writer=tf_writer, logger_file=logger_file)
            global_model.afterTrain(i, logger_file=logger_file)

        # 计算 ACC 和 FGT
        acc = 0
        total_weight = 0
        # 计算 ACC 和 FGT
        task_num = len(global_model.task_accuracies)
        # 加权计算所有任务的准确率
        for i in range(task_num):
            # 计算每个任务的类别数
            if i == 0:
                task_weight = args.fg_nc
            else:
                task_weight = global_model.task_size

            task_acc = sum(global_model.task_accuracies[i]) / len(global_model.task_accuracies[i])  # 当前任务的准确率
            acc += task_acc * task_weight
            total_weight += task_weight

        # 最终加权准确率
        acc /= total_weight

        fgt = compute_forgetting_rate(global_model.task_accuracies, global_model.previous_task_accuracies)

        print(f"Final Average Accuracy (ACC): {acc:.4f}%")
        print(f"Final Forgetting (FGT): {fgt:.4f}%")
        logger_file.write('Task_accuracies is {}  \n'.format(global_model.task_accuracies))
        logger_file.write('previous_task_accuracies is {}\n'.format(global_model.previous_task_accuracies))

        # 将 ACC 和 FGT 写入日志文件
        output_log = f"Final Average Accuracy (ACC): {acc:.4f}%, Final Forgetting (FGT): {fgt:.4f}%\n"
        logger_file.write(output_log)
        logger_file.flush()
        logger_file.close()

    # ------------------------------------------------------------------------------------
    elif args.mode == 'centralized':
        # 初始化模型
        centralized_model = model
        centralized_model = centralized_model.to(args.device)

        # 集中式方法
        print("Starting centralized raw method training...")
        print("Whether use peft? {}".format(args.is_peft))

        num_params = np.sum([p.numel() for p in model.parameters() if p.requires_grad or not p.requires_grad])
        learnable_params = np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])
        print('# of learnable params: {}; total params: {}; Proportion: {:.4f}%'.format(learnable_params, num_params, (
                learnable_params / num_params) * 100))

        # 初始化数据
        centralized_trainer = vitlora(args, file_name, centralized_model, task_size, args.device)
        centralized_trainer.setup_data(shuffle=True)

        centralized_trainer.preprocess_test_set()

        # 每个任务的训练过程
        for i in range(args.task_num + 1):
            filename = 'log_task_raw{}.txt'.format(i)
            logger_file = open(os.path.join(cur_path + keyname, args.store_name, filename), 'w')

            centralized_trainer.beforeTrain_raw(i, logger_file=logger_file)
            if centralized_trainer.all_tasks_completed:
                break
            centralized_trainer.raw_train(current_task=i, old_class=0, tf_writer=None, logger_file=logger_file)
            centralized_trainer.afterTrain_raw(current_task=i, logger_file=logger_file)

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

        # 计算 FGT
        fgt = compute_forgetting_rate(centralized_trainer.task_accuracies, centralized_trainer.previous_task_accuracies)

        print(f"Final Average Accuracy (ACC): {acc:.4f}%")
        print(f"Final Forgetting (FGT): {fgt:.4f}%")

        logger_file.write('Task_accuracies is {}  \n'.format(centralized_trainer.task_accuracies))
        logger_file.write('previous_task_accuracies is {}\n'.format(centralized_trainer.previous_task_accuracies))

        # 将 ACC 和 FGT 写入日志文件
        output_log = f"Final Average Accuracy (ACC): {acc:.4f}%, Final Forgetting (FGT): {fgt:.4f}%\n"
        logger_file.write(output_log)
        logger_file.flush()

        logger_file.close()
