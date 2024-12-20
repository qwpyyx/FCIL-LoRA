"""Calculate the Fisher matrix for EWC."""
import os

import torch
import torch.distributed as dist
from tqdm.auto import tqdm


def gather_importance(head_importance):
    head_importance_list = [torch.zeros_like(head_importance) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=head_importance_list, tensor=head_importance.contiguous())  # everyone need to do this
    head_importance_list = torch.stack(head_importance_list)
    head_importance = torch.mean(head_importance_list, dim=0)
    return head_importance


# def fisher_compute(train_dataloader_prune, model, self_fisher, accelerator, args):
#     torch.cuda.empty_cache()
#     if args.mode == 'centralized':
#         fisher_path = os.path.join(args.output_dir, 'fisher')
#     # os.makedirs(fisher_path, exist_ok=True)
#     if args.task > 0:
#         fisher_old = {}
#         for n, _ in model.named_parameters():
#             fisher_old[n] = self_fisher[n].clone().cpu()  # Move fisher_old to cpu to save gpu memory.
#
#     # Init
#     progress_bar = tqdm(range(len(train_dataloader_prune)), disable=not accelerator.is_local_main_process)
#
#     fisher = {}
#     for n, p in model.named_parameters():
#         fisher[n] = 0 * p.data
#     # Compute
#     model.train()
#
#     for step, inputs in enumerate(train_dataloader_prune):
#         model.zero_grad()
#         input_ids = inputs['input_ids']
#         sbatch = input_ids.size(0)
#
#         if 'bart' in model.args.baseline or 't5' in model.args.baseline:
#             outputs = model(**inputs, self_fisher=self_fisher)
#         else:
#             outputs = model(inputs, self_fisher=self_fisher)
#
#         loss = outputs.loss  # loss 1
#
#         loss = loss / args.gradient_accumulation_steps
#
#         accelerator.backward(loss)  # sync
#         progress_bar.update(1)
#         progress_bar.set_description('EWC Fisher Compute Iter (loss=%5.3f)' % loss.item())
#         # Get model
#         for n, p in model.named_parameters():
#             if p.grad is not None:
#                 fisher[n] += sbatch * p.grad.data.pow(2)
#
#     # Mean
#     for n, _ in model.named_parameters():
#         fisher[n] = fisher[n] / len(train_dataloader_prune)
#         fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)
#
#     self_fisher = fisher
#
#     if args.task > 0:
#         for n, _ in model.named_parameters():
#             self_fisher[n] = (self_fisher[n] + fisher_old[n].cuda() * args.task) / (
#                     args.task + 1)
#
#     accelerator.wait_for_everyone()
#
#     if args.mode == 'centralized':
#         if accelerator.is_main_process:
#             torch.save(self_fisher, fisher_path)
#
#     return fisher
def fisher_compute(train_dataloader_prune, model, self_fisher, accelerator, args):
    torch.cuda.empty_cache()

    # 判断模式是集中式还是联邦学习
    if args.mode == 'centralized':
        fisher_path = os.path.join(args.output_dir, 'fisher')
        # 集中式训练：如果是后续任务，保存之前的 Fisher
        if args.task > 0:
            fisher_old = {}
            for n, _ in model.named_parameters():
                fisher_old[n] = self_fisher[n].clone().cpu()  # 保存旧的 Fisher

    # 如果是联邦学习模式，处理自适应 Fisher
    elif args.mode == 'federated':
        if self_fisher is None:
            # 如果self_fisher为None，表示这是该节点第一次参与任务，像集中式第一个任务一样计算 Fisher
            fisher_old = None
        else:
            # 如果self_fisher已存在，保存之前的 Fisher（跟集中式处理一样）
            fisher_old = {}
            for n, _ in model.named_parameters():
                fisher_old[n] = self_fisher[n].clone().cpu()  # 保存旧的 Fisher

    # 初始化 Fisher
    progress_bar = tqdm(range(len(train_dataloader_prune)), disable=not accelerator.is_local_main_process)

    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = 0 * p.data  # 初始化为0
    # 计算 Fisher
    model.train()

    for step, inputs in enumerate(train_dataloader_prune):
        model.zero_grad()
        input_ids = inputs['input_ids']
        sbatch = input_ids.size(0)

        if 'bart' in args.baseline or 't5' in args.baseline:
            outputs = model(**inputs, self_fisher=self_fisher)
        else:
            outputs = model(inputs, self_fisher=self_fisher)

        loss = outputs.loss  # loss 1
        loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)  # sync
        progress_bar.update(1)
        progress_bar.set_description('EWC Fisher Compute Iter (loss=%5.3f)' % loss.item())

        # 累加 Fisher
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += sbatch * p.grad.data.pow(2)

    # 计算 Fisher 的均值
    for n, _ in model.named_parameters():
        fisher[n] = fisher[n] / len(train_dataloader_prune)
        fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)

    self_fisher = fisher

    # 更新 Fisher
    if args.task > 0:
        # 只有在集中式或者联邦学习模式下，self_fisher 存在时才进行更新
        if args.mode == 'federated' and fisher_old is not None:
            # 如果是 FL 模式并且 self_fisher 存在，按集中式的方式更新 Fisher
            for n, _ in model.named_parameters():
                self_fisher[n] = (self_fisher[n] + fisher_old[n].cuda() * args.task) / (args.task + 1)
        elif args.mode == 'centralized':
            # 集中式更新 Fisher
            for n, _ in model.named_parameters():
                self_fisher[n] = (self_fisher[n] + fisher_old[n].cuda() * args.task) / (args.task + 1)

    accelerator.wait_for_everyone()

    # 如果是集中式模式，保存 Fisher
    if args.mode == 'centralized' and accelerator.is_main_process:
        accelerator.save(self_fisher, fisher_path)

    return fisher
