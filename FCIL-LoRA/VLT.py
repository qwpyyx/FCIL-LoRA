import torch
import torch.nn as nn

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
                 is_peft: bool,
                 num_classes: int,
                 r: int = 4,
                 lora_layer=None,
                 return_feature=True):
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

        if is_peft:
            # 使用LoRA配置参数
            self.lora_layer = lora_layer if lora_layer else ["q_proj", "v_proj"]

            # PEFT的LoRA配置
            # TODO CAUSAL_LM是不是一种选择？
            lora_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                     inference_mode=False, r=r,
                                     lora_alpha=32,
                                     target_modules=self.lora_layer,
                                     bias="none",
                                     lora_dropout=0.1,
                                     fan_in_fan_out=True)

            self.model = get_peft_model(self.model, lora_config)

            # 固定所有参数，只更新lora参数
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                # TODO 改成lora_b
                if "lora" in name.lower():
                    param.requires_grad = True
        else:
            # full fine tune
            for param in self.model.parameters():
                param.requires_grad = True

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad == True:
        #         print(name)

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

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        return logits



    def save_lora_parameters(self, filename: str) -> None:
        r"""保存LoRA的参数，只支持safetensors格式"""
        assert filename.endswith(".safetensors")
        self.encoder.save_pretrained(filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""加载LoRA的参数，只支持safetensors格式"""
        assert filename.endswith(".safetensors")
        self.encoder.from_pretrained(filename)

    def switch_lora(self, idx: int):
        for name, module in self.encoder.named_modules():
            if hasattr(module, 'lora_id'):
                module.lora_id = idx




