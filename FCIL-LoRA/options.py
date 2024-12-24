#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import logging
from transformers import (
    MODEL_MAPPING,
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def args_parser():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--client_local', type=int, default=5,
                        help='the number of clients in a local training: M')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--centers_lr', type=float, default=1e-4,
                        help='learning rate of centers')
    parser.add_argument('--encoders_lr', type=float, default=1e-4,
                        help='learning rate of optimizer')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--r', type=int, default=32,
                        help="rank of lora")
    # parser.add_argument('--mix_precision', type=str, default='no')
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='bart_classification_bart-base')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--baseline', type=str, default='vanilla',
                        help="Whether to use method to overcome time-forgetting")
    parser.add_argument("--saved_output_dir", type=str, help="Where to store the final model.")

    # Trainer arguments
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--num_warmup_steps", type=int, default=5,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_scheduler_type", default="linear", help="The scheduler type to use.",
                        choices=["none", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides epoch.")

    # ?
    parser.add_argument("--idrandom", type=int, help="which sequence to use", default=0)
    parser.add_argument("--ft_task", type=int, help="task id")
    parser.add_argument("--base_dir", default='./outputs', type=str, help="task id")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    parser.add_argument("--ntasks", type=int, help="total number of tasks")

    parser.add_argument("--sequence_file", type=str, help="smax", default='/home/qiuwenqi/LLM/Fedfinetune/CL/VAG-main/sequences/fewrel')
    # parser.add_argument("--ntasks", type=int, help="total number of tasks")
    parser.add_argument('--classifier_lr', type=float)

    # method hyper
    parser.add_argument('--store_ratio', type=float, default=0.1,
                        help='sample ratio of old data for replay')
    parser.add_argument('--lamb', type=int, default=100, help='ewc lamda')
    parser.add_argument('--lamb_distill', type=int, default=1, help='ewc lamda')
    parser.add_argument('--aug_ratio', type=float, help='Ratio of the augmented data.', default=0.1)
    parser.add_argument('--use_dev', action='store_true', help='Use the dev set for early stopping.')
    parser.add_argument("--eval_every_epoch", action="store_true", help="Evaluate in each epoch.")

    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--gpu', type=int, default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--classes_per_client', type=int, default=60, help='non-iid classes per client')
    parser.add_argument('--task_num', default=0, type=int, help='number of sequential tasks')
    parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    parser.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')
    parser.add_argument('--total_classes', default=100, type=int, help='total classes')
    parser.add_argument('--fg_nc', default=7, type=int, help='the number of classes in first task')
    parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')
    parser.add_argument('--niid_type', default='D', type=str, help='Quality or Distributed(non-iid)')
    parser.add_argument('--alpha', default=6, type=int, help='quantity skew')
    parser.add_argument('--beta', default=0.5, type=float, help='distribution skew')
    parser.add_argument('--mode', type=str, default='federated', choices=['federated', 'centralized'],
                        help="Mode: 'federated' or 'centralized'")
    parser.add_argument('--model_name_or_path', type=str, default='/home/qiuwenqi/LLM/models/bart-base',
                        help="moedel_name: roberta or llama2")
    parser.add_argument('--combine', default=False, type=bool, help='Whether to combine the data')
    parser.add_argument('--is_peft', type=int, default=1,
                        help="Whether to use peft such as lora to fine tune model")
    parser.add_argument('--deepspeed', type=str, default=None, help="DeepSpeed configuration file")
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--task', type=int, default=None)
    #Replay
    parser.add_argument('--is_replay', type=int, default=0,
                        help="Whether to use replay to fine tune model")
    # NonCL
    parser.add_argument('--is_CL', type=int, default=1,
                        help="Whether to use continuous learning")

    args = parser.parse_args()

    return args
