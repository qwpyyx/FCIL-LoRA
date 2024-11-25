# FCIL-LoRA
federated continuous incremental learning combine lora

注意encoders_lr的影响，他会影响lora矩阵的更新学习率，非常影响准确率

命令行
--mode centralized --model=FCILLoRA --dataset=banking77 --iid=0 --centers_lr=1e-3 --encoders_lr=1e-4 --epochs=30 --gpu 3 --task_num 6 --fg_nc 11 --total_classes 77 --local_ep 10 --num_users 30 --client_local 10 --niid_type D --beta 0.1 --local_bs=64
