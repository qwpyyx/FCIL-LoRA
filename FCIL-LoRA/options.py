#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=30,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--client_local', type=int, default=10,
                        help='the number of clients in a local training: M')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--centers_lr', type=float, default=1e-4,
                        help='learning rate of centers')
    parser.add_argument('--encoders_lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--r', type=int, default=16,
                        help="rank of lora")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
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
    parser.add_argument('--gpu', type=int, default=5, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
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
    parser.add_argument('--niid_type', default='Q', type=str, help='Quality or Distributed(non-iid)')
    parser.add_argument('--alpha', default=6, type=int, help='quantity skew')
    parser.add_argument('--beta', default=0.5, type=float, help='distribution skew')
    parser.add_argument('--mode', type=str, default='federated', choices=['federated', 'centralized'],
                        help="Mode: 'federated' or 'centralized'")
    parser.add_argument('--model_path', type=str, default='/home/qiuwenqi/LLM/models/roberta-base',
                        help="moedel_name: roberta or llama2")
    parser.add_argument('--combine', default=False, type=bool, help='Whether to combine the data')
    parser.add_argument('--is_peft', type=int, default=1,
                        help="Whether to use peft such as lora to fine tune model")
    parser.add_argument('--deepspeed', type=str, default=None, help="DeepSpeed configuration file")

    #Replay
    parser.add_argument('--old_data_replay_ratio', type=float, default=0.1,
                        help='sample ratio of old data for replay')

    parser.add_argument('--is_replay', type=int, default=0,
                        help="Whether to use replay to fine tune model")

    args = parser.parse_args()
    return args
