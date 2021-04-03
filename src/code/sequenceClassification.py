'''
Author: your name
Date: 2021-04-03 14:27:22
LastEditTime: 2021-04-03 16:26:28
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/sequenceClassification.py
'''
from model import *
import torch
import torch.nn as nn


class GroundedModelForSequenceClassification(nn.Module):

    def __init__(self, args, num_classes: int):
        super().__init__()
        self.grounded_lm = GroundedModel(args)
        self.dropout = nn.Dropout(p=0.3)
        self.dropout_g = nn.Dropout(p=0.3)
        self.num_classes = num_classes

        # pooler
        self.hidden_size = self.grounded_lm.pretrained_lm.model.config.hidden_size
        self.pooler_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.pooler_activation = nn.Tanh()

        # classifier
        self.classifier = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        """
        input_ids: (bs, seq_length)
        labels: (bs, )
        """
        bs = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        grounded_output, original_output = self.grounded_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        assert grounded_output.shape[0] == bs and grounded_output.shape[1] == seq_length
        assert original_output.shape[0] == bs and original_output.shape[1] == seq_length

        # original LM output
        # pool
        original_pooled_output = self.pooler_dense(original_output[:, 0, :])
        original_pooled_output = self.pooler_activation(original_pooled_output)
        original_pooled_output = self.dropout(original_pooled_output)

        # adapter output
        # pool
        grounded_pooled_output = torch.mean(grounded_output, dim=1)
        grounded_pooled_output = self.dropout_g(grounded_pooled_output)

        concatenated_output = torch.cat(
            [original_pooled_output, grounded_pooled_output], dim=-1)
        logits = self.classifier(concatenated_output)  # (bs, num_classes)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        else:
            loss = None

        return loss
