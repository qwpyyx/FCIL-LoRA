# FCIL-LoRA
federated continuous incremental learning combine lora


命令行
--mode centralized --model=FCILLoRA --dataset=banking77 --iid=0 --centers_lr=1e-3 --encoders_lr=1e-5 --epochs=30 --gpu 3 --task_num 6 --fg_nc 11 --total_classes 77 --local_ep 10 --num_users 30 --client_local 10 --niid_type D --beta 0.1 --local_bs=64
