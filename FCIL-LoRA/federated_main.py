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
from VITLORA import vitlora
import logging
logger = logging.getLogger(__name__)
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import (
    set_seed,
)


if __name__ == '__main__':
    args = args_parser()

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
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    args.device = accelerator.device

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

    args.total_classes = 300
    logger.info('==> Building model..')


    # ------------------------------------------------------------------------------------
    if args.mode == 'federated':

        global_model = vitlora(args, task_size, args.device)
        global_model.setup_data(shuffle=True)

        global_model.beforeTrain(args.task)
        if global_model.all_tasks_completed:
            exit()
        global_model.train(args.task, accelerator=accelerator, dev_loader=None)


    # ------------------------------------------------------------------------------------
    elif args.mode == 'centralized':

        centralized_trainer = vitlora(args, task_size, args.device)
        centralized_trainer.setup_data(shuffle=True)

        centralized_trainer.beforeTrain(args.task)
        if centralized_trainer.all_tasks_completed:
            exit()
        centralized_trainer.raw_train(current_task=args.task, accelerator=accelerator, dev_loader=None)

