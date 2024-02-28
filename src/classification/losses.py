import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import compute_class_weight

#
# Loss Functions
#


class LabelLoss(nn.Module):
    def __init__(self, classes, target_train):
        super().__init__()
        self._classes = classes
        self._target_train = target_train
        self._xe_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def __repr__(self):
        return f'<{self.__class__.__name__}: loss=XEnt, num_classes={len(self._classes)}>'

    def forward(self, logits, targets):

        target_labels = torch.LongTensor(targets).to(logits.device)
        loss = self._xe_loss(logits, target_labels)

        return loss

    def get_accuracy(self, predictions, targets):

        target_labels = torch.LongTensor(targets).to(predictions.device)

        # compute label accuracy
        num_label_matches = torch.sum(predictions == target_labels)
        accuracy = float(num_label_matches / predictions.shape[0])

        return accuracy
