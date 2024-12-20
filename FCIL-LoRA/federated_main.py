#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# Revised by Adonis

import os
import warnings
os.environ["WANDB_MODE"] = "disabled"
warnings.filterwarnings('ignore')
from options import args_parser
from utils import exp_details
from VLT import *
from VITLORA import vitlora
import random
import numpy as np
import deepspeed
from utils import compute_forgetting_rate, configure_logging, compute_final_acc, initialize_model
import logging
logger = logging.getLogger(__name__)
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import (
    set_seed,
)

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
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.is_CL:
        task_size = int((args.total_classes - args.fg_nc) / args.task_num)
    else:
        task_size = 0
    args.type = 'iid' if args.iid == 1 else 'non-iid'

    if args.mode == 'federated':
        e = args.epochs
        local_e = args.local_ep
        if args.is_peft:
            nam = 'lora'
            args.base_dir = args.base_dir + '/FL' + '/' + nam
            if args.type == 'iid':
                args.base_dir = args.base_dir + '/iid' + f'/epochs_{e}local_ep{local_e}'
            else:
                args.base_dir = args.base_dir + '/non-iid' + '/beta_' + str(args.beta) + f'/epochs_{e}local_ep{local_e}'
        else:
            nam = 'FullTuning'
            args.base_dir = args.base_dir + '/FL' + '/' + nam
            if args.type == 'iid':
                args.base_dir = args.base_dir + '/iid' + f'/epochs_{e}local_ep{local_e}'
            else:
                args.base_dir = args.base_dir + '/non-iid' + '/beta_' + str(args.beta) + f'/epochs_{e}local_ep{local_e}'
    elif args.mode == 'centralized':
        e = args.epochs
        if args.is_peft:
            nam = 'lora'
            args.base_dir = args.base_dir + '/CL' + '/' + nam + f'/epochs_{e}'
        else:
            nam = 'FullTuning'
            args.base_dir = args.base_dir + '/CL' + '/' + nam + f'/epochs_{e}'



    args.total_num = args.task_num + 1
    exp_details(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=args.mix_precision, kwargs_handlers=[ddp_kwargs])
    args.device = accelerator.device
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if args.log_dir is not None:
        handler = logging.FileHandler(args.log_dir)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # BUILD MODEL
    args.total_classes = 300
    # file_name = args.store_name
    logger.info('==> Building model..')
    args.task = 0
    model, data_collator, tokenizer = initialize_model(args)
    # model = model.to(args.device)
    # --------------------
    if args.mode == 'federated':
        # federated_model = model
        # federated_model = federated_model.to(args.device)

        # print("Beginning federated learning...")
        # print("Whether use peft? {}".format(args.is_peft))
        #
        # num_params = np.sum([p.numel() for p in model.parameters() if p.requires_grad or not p.requires_grad])
        # learnable_params = np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])
        # print('# of learnable params: {}; total params: {}; Proportion: {:.4f}%'.format(learnable_params, num_params, (
        #         learnable_params / num_params) * 100))

        global_model = vitlora(args, model, task_size, args.device, data_collator)
        global_model.setup_data(shuffle=True, tokenizer=tokenizer)
        # global_model.preprocess_test_set(tokenizer)

        old_class = 0

        # 每个任务
        for i in range(args.total_num):

            global_model.beforeTrain(i, logger_file=None, device=args.device)
            if global_model.all_tasks_completed:
                break

            global_model.train(i, logger_file=None, accelerator=accelerator, dev_loader=None)
            # global_model.afterTrain(i, logger_file=None)



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

        logger.info('==> Preparing data..')
        # 初始化数据
        centralized_trainer = vitlora(args, centralized_model, task_size, args.device, data_collator)
        centralized_trainer.setup_data(shuffle=True, tokenizer=tokenizer)

        # centralized_trainer.preprocess_test_set(tokenizer)

        # 每个任务的训练过程
        for i in range(args.total_num):
            # filename = 'log_task_raw{}.txt'.format(i)
            # logger_file = open(os.path.join(cur_path + keyname, args.store_name, filename), 'w')

            centralized_trainer.beforeTrain_raw(i, logger_file=None, device=args.device)
            if centralized_trainer.all_tasks_completed:
                break
            centralized_trainer.raw_train(current_task=i, old_class=0, tf_writer=None, logger_file=None,
                                          accelerator=accelerator, dev_loader=None)
            # centralized_trainer.afterTrain_raw(current_task=i, logger_file=None)

        # acc = compute_final_acc(args, centralized_trainer)

        # 计算 FGT
        # fgt = compute_forgetting_rate(centralized_trainer.task_accuracies, centralized_trainer.previous_task_accuracies)

        # print(f"Final Average Accuracy (ACC): {acc:.4f}%")
        # print(f"Final Forgetting (FGT): {fgt:.4f}%")

        # logger_file.write('Task_accuracies is {}  \n'.format(centralized_trainer.task_accuracies))
        # logger_file.write('previous_task_accuracies is {}\n'.format(centralized_trainer.previous_task_accuracies))
        #
        # # 将 ACC 和 FGT 写入日志文件
        # output_log = f"Final Average Accuracy (ACC): {acc:.4f}%, Final Forgetting (FGT): {fgt:.4f}%\n"
        # logger_file.write(output_log)
        # logger_file.flush()
        #
        # logger_file.close()
