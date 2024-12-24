import torch
import torch.nn as nn

from transformers import (
    DataCollatorWithPadding,
    AutoConfig,
    PreTrainedModel,
    BartConfig,
    PretrainedConfig,
    AutoModelForSeq2SeqLM,
    # add
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
    AutoModelForSequenceClassification,
    Seq2SeqTrainingArguments,
    set_seed, )
from transformers.file_utils import ModelOutput
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from transformers.models.bart.modeling_bart import BartClassificationHead
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqSequenceClassifierOutput
import numpy as np
import torch.nn.functional as F

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss

class LLMWithLoRA(PreTrainedModel):
    def __init__(self,
                 modelname: str,
                 is_peft: bool,
                 num_classes: int,
                 r: int = 4,
                 lora_layer=None,
                 return_feature=True):
        config = AutoConfig.from_pretrained(
            modelname,
            num_labels=num_classes,
            cache_dir=None,
            output_hidden_states=True,
            revision="main",
            use_auth_token=None,
        )
        super().__init__(config)

        # 使用预训练LLM模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            modelname,
            cache_dir=None,
            use_fast=True,
            revision="main",
            use_auth_token=None,
        )
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        self.data_collator = DataCollatorWithPadding(self.tokenizer)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            modelname,
            config=config,
            cache_dir=None,
            revision="main",
            use_auth_token=None,
        )

        self.num_classes = num_classes
        self.return_feature = return_feature
        self.is_peft = is_peft
        if is_peft:
            self.lora_layer = lora_layer if lora_layer else ["q_proj", "v_proj"]
            lora_alpha = 2 * r
            lora_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                     inference_mode=False, r=r,
                                     lora_alpha=lora_alpha,
                                     target_modules=self.lora_layer,
                                     bias="none",
                                     lora_dropout=0.1,
                                     fan_in_fan_out=True)

            self.model = get_peft_model(self.model, lora_config)

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.masked_label = None




    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            adapter_names=None,
            restrict_label=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )
        if self.is_peft:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                adapter_names=adapter_names,
            )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = outputs['decoder_hidden_states'][-1]  # last hidden state

        # eos_mask = input_ids.eq(self.config.eos_token_id)
        # eos_mask = self.model.encoder.adjust_attention_mask_for_parallel(hidden_states, eos_mask)
        #
        # if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
        #     raise ValueError("All examples must have the same number of <eos> tokens.")
        # sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
        #                           :, -1, :
        #                           ]
        # logits = self.classification_head(sentence_representation)

        logits = outputs.logits



        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                if self.masked_label is not None and restrict_label is True:
                    # We restrict the softmax probability to a reduced label set.
                    for tmp_i in labels:
                        if self.masked_label[tmp_i] == 0:  # Sanity check.
                            print('Masked label sanity check fail!')
                            import pdb
                            pdb.set_trace()
                    logits = logits * self.masked_label.to(logits.device)
                    logits = torch.where(logits != 0, logits, torch.tensor(-np.inf).to(logits.device))
                    # print('Restrict label successfully!')
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class MyBart(nn.Module):
    """Wrapper on top of MyBartForSequenceClassification."""

    def __init__(self, model, teacher=None, args=None):
        super().__init__()
        self.model = model
        self.teacher = teacher
        self.kd_loss = DistillKL(1)
        self.config = model.config
        self.args = args
        self.mse = torch.nn.MSELoss()
        self.dropout = nn.Dropout(0.1)
        self.tokenizer = self.model.tokenizer

        if 'ldbr' in args.baseline:
            self.General_Encoder = nn.Sequential(
                nn.Linear(self.model.config.d_model, 128),
                nn.Tanh()
            )
            self.Specific_Encoder = nn.Sequential(
                nn.Linear(self.model.config.d_model, 128),
                nn.Tanh()
            )
            self.task_classifier = nn.Sequential(
                nn.Linear(128, args.ntasks)
            )
            self.cls_classifier = nn.Sequential(
                nn.Linear(2 * 128, self.model.config.num_labels)
            )

    def set_masked_label(self, masked_label):
        self.model.set_masked_label(masked_label)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            adapter_names=None,
            buffer=None,
            restrict_label=False,
            **kwargs
    ):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             head_mask=head_mask,
                             decoder_head_mask=decoder_head_mask,
                             cross_attn_head_mask=cross_attn_head_mask,
                             encoder_outputs=encoder_outputs,
                             inputs_embeds=inputs_embeds,
                             decoder_inputs_embeds=decoder_inputs_embeds,
                             labels=labels,
                             use_cache=use_cache,
                             output_attentions=output_attentions,
                             output_hidden_states=True,
                             restrict_label=restrict_label)
        loss = outputs.loss
        logits = outputs.logits
        distill_loss = None

        # For experience replay, interleaving old samples with current data in training batches.
        if 'experience_replay' in self.args.baseline and buffer is not None and buffer.num_seen_examples > 0:
            replay_input_ids, replay_labels, replay_attention_mask = buffer.get_data(input_ids.size(0))
            replay_input_ids = replay_input_ids.to(self.model.device)
            replay_labels = replay_labels.to(self.model.device)
            replay_attention_mask = replay_attention_mask.to(self.model.device)
            replay_outputs = self.model(input_ids=replay_input_ids,
                                        labels=replay_labels,
                                        attention_mask=replay_attention_mask)
            # print('Add replay data successfully!')
            loss += replay_outputs.loss

        if 'derpp' in self.args.baseline and buffer is not None and buffer.num_seen_examples > 0:
            replay_input_ids, replay_labels, replay_logits, replay_attention_mask = buffer.get_data(input_ids.size(0))
            replay_input_ids = replay_input_ids.to(self.model.device)
            replay_labels = replay_labels.to(self.model.device)
            replay_logits = replay_logits.to(self.model.device)
            replay_attention_mask = replay_attention_mask.to(self.model.device)
            replay_outputs = self.model(input_ids=replay_input_ids,
                                        labels=replay_labels,
                                        attention_mask=replay_attention_mask)
            # Set alpha, beta to 0.5, 0.5
            loss = loss + 0.5 * replay_outputs.loss + 0.5 * F.mse_loss(replay_logits, outputs.logits)

        if 'distill' in self.args.baseline:
            student_ori = outputs
            teacher_ori = self.teacher(input_ids,
                                       attention_mask=attention_mask,
                                       decoder_input_ids=decoder_input_ids,
                                       decoder_attention_mask=decoder_attention_mask,
                                       head_mask=head_mask,
                                       decoder_head_mask=decoder_head_mask,
                                       cross_attn_head_mask=cross_attn_head_mask,
                                       encoder_outputs=encoder_outputs,
                                       inputs_embeds=inputs_embeds,
                                       decoder_inputs_embeds=decoder_inputs_embeds,
                                       labels=labels,
                                       use_cache=use_cache,
                                       output_attentions=output_attentions,
                                       output_hidden_states=True)
            distill_loss = self.kd_loss(teacher_ori.decoder_hidden_states[-1], student_ori.decoder_hidden_states[-1])

        if 'ewc' in self.args.baseline and 'self_fisher' in kwargs:  # We don't need to do this in evaluation.
            loss_reg = 0
            if self.args.mode == 'centralized':
                if self.args.task > 0:
                    for (name, param), (_, param_old) in zip(self.model.named_parameters(),
                                                             self.teacher.named_parameters()):
                        try:
                            loss_reg += torch.sum(
                                kwargs['self_fisher']['model.' + name] * (param_old.cuda() - param.cuda()).pow(2)) / 2
                        except KeyError:
                            loss_reg += torch.sum(
                                kwargs['self_fisher']['module.model.' + name] * (param_old.cuda() - param.cuda()).pow(2)) / 2
                loss += self.args.lamb * loss_reg
            elif self.args.mode == 'federated':
                if kwargs.get('self_fisher') is not None:  # Only execute EWC if self_fisher exists
                    loss_reg = 0
                    if self.args.task > 0:
                        for (name, param), (_, param_old) in zip(self.model.named_parameters(),
                                                                 self.teacher.named_parameters()):
                            try:
                                loss_reg += torch.sum(
                                    kwargs['self_fisher']['model.' + name] * (param_old.cuda() - param.cuda()).pow(2)) / 2
                            except KeyError:
                                loss_reg += torch.sum(
                                    kwargs['self_fisher']['module.model.' + name] * (param_old.cuda() - param.cuda()).pow(2)) / 2
                    loss += self.args.lamb * loss_reg

        if 'ldbr' in self.args.baseline:
            sentence_embedding = outputs.decoder_hidden_states[-1][:, -1, :]
            general_features = self.General_Encoder(sentence_embedding)
            specific_features = self.Specific_Encoder(sentence_embedding)

            task_pred = self.task_classifier(specific_features)
            features = torch.cat([general_features, specific_features], dim=1)
            logits = self.cls_classifier(features)
            loss_fct = nn.CrossEntropyLoss()
            if labels is not None:
                loss = loss_fct(logits, labels)
            else:
                loss = None
        else:
            sentence_embedding = None
            task_pred = None
            general_features = None
            specific_features = None

        return ModelOutput(
            loss=loss,
            distill_loss=distill_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            total_g_fea=general_features,
            total_s_fea=specific_features,
            task_pred=task_pred,
            sentence_embedding=sentence_embedding
        )