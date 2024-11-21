#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


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
        os.path.join(cur_path + '/logs-roberta-LoRA', args.store_name),
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
    args.store_name = '_'.join(
        [args.dataset, args.model, args.type, 'lr-' + str(args.centers_lr)])
    cur_path = os.path.abspath(os.path.dirname(os.getcwd()))
    # 在当前目录构建检查点和日志
    cur_path = cur_path
    prepare_folders(cur_path)
    # 打印一些信息
    exp_details(args)

    setup_seed(args.seed)

    # BUILD MODEL
    file_name = args.store_name
    class_set = list(range(args.total_classes))

    # 这里改成LLM,而且是已经有了lora的架构
    # 应该是config 和 model， peft
    model = LLMWithLoRA(modelname='/home/qiuwenqi/LLM/models/roberta-base',  # 可以选择适合的LLM，例如't5-base'或其他预训练模型
                        num_classes=args.total_classes,
                        r=args.r,
                        lora_layer=["query", "value"])  # 指定 LoRA 作用的层
    model = model.to(args.device)

    global_model = vitlora(args, file_name, model, task_size, args.device)
    global_model.setup_data(shuffle=True)

    # 冻结所有参数
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 解冻 LoRA 的参数
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
        logger_file = open(os.path.join(cur_path + '/logs-roberta-LoRA', args.store_name, filename), 'w')
        tf_writer = SummaryWriter(log_dir=os.path.join(cur_path + '/logs-roberta-LoRA', args.store_name))

        # 执行任务相关的初始化
        global_model.beforeTrain(i)
        # 训练和后处理
        global_model.train(i, old_class=old_class, tf_writer=tf_writer, logger_file=logger_file)
        global_model.afterTrain(i)
