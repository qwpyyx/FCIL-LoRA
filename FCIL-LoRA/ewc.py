import torch

class EWC:
    def __init__(self, model, lamda=1000, is_peft=False):
        self.model = model
        self.lamda = lamda
        self.is_peft = is_peft  # 是否使用LoRA
        self.fisher_information = None
        self.params = None

    def compute_fisher_information(self, dataloader, is_peft, task_id, prev_fisher=None):
        self.model.eval()
        fisher_information = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        device = next(self.model.parameters()).device
        print("Computing Fisher Information Matrix for task {}".format(task_id))
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            self.model.zero_grad()
            logits = self.model(**inputs)
            loss = torch.nn.CrossEntropyLoss()(logits, inputs['labels'])
            loss.backward()

            for name, param in self.model.named_parameters():
                if is_peft and 'lora' not in name.lower():  # 如果是LoRA微调，只计算LoRA的参数
                    continue
                fisher_information[name] += param.grad ** 2

        self.fisher_information = {name: fisher_information[name] / len(dataloader) for name in fisher_information}

    def ewc_loss(self, is_peft=False, current_task=0):
        loss = 0
        if self.fisher_information and self.params:
            for name, param in self.model.named_parameters():
                if is_peft and 'lora' not in name.lower():  # 只对LoRA参数计算损失
                    continue
                if name in self.fisher_information:
                    if current_task > 0:
                        # 使用旧的 Fisher 信息来加权
                        loss += (self.fisher_information[name] * (param - self.params[name]) ** 2).sum()
                    else:
                        loss += (self.fisher_information[name] * (param - self.params[name]) ** 2).sum()

        return self.lamda * loss

    def save_params(self, task_id, is_peft=False):
        """保存当前任务的模型参数和Fisher信息"""
        if is_peft:
            self.params = {name: param.clone() for name, param in self.model.named_parameters() if
                           'lora' in name.lower() and param.requires_grad}
        else:
            self.params = {name: param.clone() for name, param in self.model.named_parameters() if param.requires_grad}

        torch.save(self.params, f'params_task_{task_id}.pt')
        torch.save(self.fisher_information, f'fisher_info_task_{task_id}.pt')


