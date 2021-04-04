'''
Author: your name
Date: 2021-04-03 14:27:12
LastEditTime: 2021-04-04 17:06:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/multipleChoice.py
'''
from model import *
import torch.nn as nn
import torch


class GroundedModelForMultiplceChoice(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.grounded_lm = GroundedModel(args)
        self.dropout = nn.Dropout(p=0.3)
        self.dropout_g = nn.Dropout(p=0.3)

        # pooler
        self.hidden_size = self.grounded_lm.pretrained_lm.model.config.hidden_size
        self.pooler_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.pooler_activation = nn.Tanh()

        # classifier
        self.classifier = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        """
        input_ids: (real_bs, num_choice, seq_length)
        labels: (real_bs, )
        """
        real_bs = input_ids.shape[0]
        num_choice = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)
                                   ) if input_ids is not None else None
        attention_mask = attention_mask.view(
            -1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(
            -1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)
                                         ) if position_ids is not None else None
        grounded_output, original_output = self.grounded_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        assert grounded_output.shape[0] == real_bs * num_choice
        assert original_output.shape[0] == real_bs * num_choice

        # original LM output
        # pool
        original_pooled_output = self.pooler_dense(original_output[:, 0, :])
        original_pooled_output = self.pooler_activation(original_pooled_output)
        original_pooled_output = self.dropout(original_pooled_output)

        # adapter output
        # pool
        grounded_pooled_output = torch.mean(grounded_output, dim=1)
        grounded_pooled_output = self.dropout_g(grounded_pooled_output)

        # (real_bs * num_choice, 2*hidden_size)
        concatenated_output = torch.cat(
            [original_pooled_output, grounded_pooled_output], dim=-1)
        assert concatenated_output.shape[-1] == 2 * self.hidden_size
        # (real_bs * num_choice, 1)
        logits = self.classifier(concatenated_output)
        # (real_bs, num_choice)
        reshaped_logits = logits.reshape(-1, num_choice)
        assert reshaped_logits.shape[0] == real_bs and reshaped_logits.shape[1] == num_choice

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        else:
            loss = None

        return loss, reshaped_logits

    def load_adapter_from_ckpt(self, ckpt_path: str):
        self.grounded_lm.load_from_ckpt(ckpt_path)


if __name__ == '__main__':
    pass
