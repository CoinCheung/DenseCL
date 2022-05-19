
'''
    cross entropy loss with input specially designed for moco/densecl
'''

import torch
import torch.nn as nn
import torch.nn.functional as F



class CrossEntropyLoss(nn.Module):

    def __init__(self, ):
        super(CrossEntropyLoss, self).__init__()
        self.crit = nn.CrossEntropyLoss()

    def forward(self, pos, neg):
        N, _, *M = pos.size()
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros((N, *M), dtype=torch.long).cuda()
        loss = self.crit(logits, labels)
        return loss
