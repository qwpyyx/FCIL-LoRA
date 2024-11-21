from timm import create_model
from functools import reduce
from operator import mul
import math
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.parameter import Parameter
from torch import Tensor
import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    # add
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
    AutoModelForSequenceClassification,
    Seq2SeqTrainingArguments,
    set_seed, )

from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig


class LLMWithLoRA(nn.Module):
    def __init__(self,
                 modelname: str,
                 num_classes: int,
                 r: int = 4,
                 lora_layer=None,
                 return_feature=True,
                 ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(
            modelname,
            num_labels=num_classes,
            cache_dir=None,
            output_hidden_states=True,
            revision="main",
            use_auth_token=None,
        )

        # 使用预训练LLM模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            modelname,
            cache_dir=None,
            use_fast=True,
            revision="main",
            use_auth_token=None,
        )
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        self.model = AutoModelForSequenceClassification.from_pretrained(
            modelname,
            config=self.config,
            cache_dir=None,
            revision="main",
            use_auth_token=None,
        )

        self.num_classes = num_classes
        self.return_feature = return_feature

        # 使用LoRA配置参数
        self.lora_layer = lora_layer if lora_layer else ["q_proj", "v_proj"]

        # PEFT的LoRA配置
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                 inference_mode=False, r=r,
                                 lora_alpha=32,
                                 target_modules=self.lora_layer,
                                 lora_dropout=0.1)

        self.model = get_peft_model(self.model, lora_config)

        self.model.resize_token_embeddings(len(self.tokenizer))

        # 初始化类别中心（如果需要使用）
        # self.feat_dim = 768
        # self.centers = nn.ParameterList([nn.Parameter(0.1 * torch.randn(self.feat_dim, 1)) for _ in range(self.num_classes)])

    def centers_initial(self, current_tasks):
        # 假设任务是某种增量文本分类，可以通过初始化特定类别的嵌入来实现类似的功能
        current_task = [i for i in range(current_tasks[0], current_tasks[1])]
        no_grad_idx = [i for i in range(self.num_classes) if i not in current_task]
        for i in no_grad_idx:
            self.centers[i].requires_grad = False
        for i in current_task:
            self.centers[i].requires_grad = True

    def compute_distance(self, x):
        # 计算输入特征和类嵌入（或其他参考嵌入）之间的距离
        centers_list = [i for i in self.centers]
        centers = torch.cat(centers_list, dim=1)
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, centers)
        dist = features_square + centers_square - features_into_centers
        dist = dist / float(x.shape[1])
        dist = torch.sqrt(dist)

        return dist, centers

    def forward_features(self, input_ids, attention_mask, decoder_input_ids=None):
        # 使用LLM提取特征，返回隐藏状态
        # 添加decoder_input_ids来支持编码器-解码器架构
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                               output_hidden_states=True)
        hidden_states = outputs.encoder_last_hidden_state  # 提取编码器最后一层的隐藏状态作为特征
        return hidden_states

    # def forward(self, input_ids, attention_mask):
    #     # 前向传播，获取分类 logits
    #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #     logits = outputs.logits
    #
        # if self.return_feature:
        #     hidden_states = outputs.hidden_states[-1]  # 提取最后一层隐藏状态作为特征
        #     return logits, hidden_states
        # else:
        #     return logits

    def forward(self, input_ids, attention_mask):
        # 前向传播，获取分类 logits
        # print("Forward pass is called with input_ids shape:", input_ids.shape)
        # # 打印 LoRA 部分的输出或者检查 LoRA 部分的具体参与
        # for name, param in self.model.named_parameters():
        #     if 'lora_B' in name:
        #         print(f"Using parameter {name}: {param[0][0].item()} before forward pass")
        #
        # for name, param in self.model.named_parameters():
        #     if 'lora_A' in name:
        #         print(f"Using parameter {name}: {param[0][0].item()} before forward pass")

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 检查模型输出和 lora 的状态
        # print("Logits shape after forward pass:", logits.shape)

        return logits

    def forward_from_text(self, texts, mode='train'):
        # 将文本转为token
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = inputs['input_ids'].to(self.encoder.device)
        attention_mask = inputs['attention_mask'].to(self.encoder.device)

        # 前向传播
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.return_feature:
            return outputs.logits, outputs.encoder_last_hidden_state
        else:
            return outputs.logits

    def save_lora_parameters(self, filename: str) -> None:
        r"""保存LoRA的参数，只支持safetensors格式"""
        assert filename.endswith(".safetensors")
        self.encoder.save_pretrained(filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""加载LoRA的参数，只支持safetensors格式"""
        assert filename.endswith(".safetensors")
        self.encoder.from_pretrained(filename)

    def switch_lora(self, idx: int):
        # 在多任务场景中切换不同的 LoRA 设置
        # 由于模型结构基于 LLMWithLoRA，假设模型中所有 LoRA 层都使用了 get_peft_model
        for name, module in self.encoder.named_modules():
            if hasattr(module, 'lora_id'):
                module.lora_id = idx




