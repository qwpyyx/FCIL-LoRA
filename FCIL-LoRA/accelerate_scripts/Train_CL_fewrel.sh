#!/bin/bash

for task in  $(seq 0 7);
do
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  accelerate launch --config_file /home/qiuwenqi/LLM/Fedfinetune/FCL/PILoRA-revise/PILoRA-cifar/configs/default_config.yaml \
  --num_processes 4 \
  --mixed_precision fp16 \
  /home/qiuwenqi/LLM/Fedfinetune/FCL/PILoRA-revise/PILoRA-cifar/federated_main.py \
  --mode=centralized \
  --task ${task} \
  --max_seq_length 128 \
  --is_peft=1 \
  --dataset=fewrel \
  --baseline bart_classification_bart-base_vanlia \
  --encoders_lr=1e-3 \
  --epochs=5 \
  --task_num=7 \
  --fg_nc=10 \
  --total_classes=80 \
  --local_bs=8 \
  --r=32
done


