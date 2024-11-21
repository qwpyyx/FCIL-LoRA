python3 federated_main.py --model=FCILLoRA_beta_0.1 --dataset=banking77 --iid=0 --centers_lr=1e-3 --encoders_lr=1e-3 --epochs=5 --gpu 0 --task_num 6 --fg_nc 11 --total_classes 77 --local_ep 3 --num_users 30 --client_local 10 --niid_type D --beta 0.1 --local_bs=32
python3 federated_main.py --model=FCILLoRA_beta_0.5 --dataset=banking77 --iid=0 --centers_lr=1e-3 --encoders_lr=1e-3 --epochs=5 --gpu 2 --task_num 6 --fg_nc 11 --total_classes 77 --local_ep 3 --num_users 30 --client_local 10 --niid_type D --beta 0.5 --local_bs=32
python3 federated_main.py --model=FCILLoRA_beta_1 --dataset=banking77 --iid=0 --centers_lr=1e-3 --encoders_lr=1e-3 --epochs=5 --gpu 3 --task_num 6 --fg_nc 11 --total_classes 77 --local_ep 3 --num_users 30 --client_local 10 --niid_type D --beta 1 --local_bs=32