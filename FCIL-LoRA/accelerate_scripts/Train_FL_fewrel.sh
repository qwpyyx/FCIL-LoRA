#!/bin/bash

for task in  $(seq 0 7);
do
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  accelerate launch --config_file /home/qiuwenqi/LLM/Fedfinetune/FCL/PILoRA-revise/PILoRA-cifar/configs/default_config.yaml \
  --num_processes 4 \
  /home/qiuwenqi/LLM/Fedfinetune/FCL/PILoRA-revise/PILoRA-cifar/federated_main.py \
  --mode=federated \
  --task ${task} \
  --max_seq_length 128 \
  --is_peft=0 \
  --dataset=fewrel \
  --baseline bart_classification_bart-base_vanlia \
  --iid=0 \
  --encoders_lr=1e-5 \
  --epochs=3 \
  --task_num=7 \
  --fg_nc=10 \
  --total_classes=80 \
  --local_ep=3 \
  --num_users=20 \
  --client_local=5 \
  --niid_type=D \
  --beta=1 \
  --local_bs=8 \
  --r=32
done


