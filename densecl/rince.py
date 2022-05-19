
'''
Proposed in this paper: https://arxiv.org/abs/2201.04309

This loss is not always greater than 0, and maybe we should add warmup to stablize training.

Sadly, if we use this to train denseCL from scratch, the loss would become nan if we use identical training parameters as original implementation(tuned for using info-nce). From my observation, the loss generates fierce gradient thus the model output logits becomes nan.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp



class RINCE(nn.Module):

    def __init__(self, q=0.5, lam=0.025):
        super(RINCE, self).__init__()
        self.q = q
        self.lam = lam

    def forward(self, pos, neg):
        loss = RINCEFunc.apply(pos, neg, self.lam, self.q)
        return loss.mean()


class RINCEFunc(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, pos, neg, lam, q):
        div_q = 1./q
        exp_pos = pos.exp().squeeze(1)
        exp_sum = exp_pos + neg.exp().sum(dim=1)
        term1 = exp_pos.pow(q).neg_()
        term2 = exp_sum.mul_(lam).pow_(q)
        loss = (term1 + term2).mul_(div_q)

        ctx.vars = pos, neg, lam, q

        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        pos, neg, lam, q = ctx.vars
        exp_pos = pos.exp().squeeze(1)
        exp_neg = neg.exp()
        exp_sum = exp_pos + exp_neg.sum(dim=1)

        d_pos = exp_sum.mul(lam).pow(q-1.).mul(lam).mul(exp_pos) - exp_pos.pow(q)
        d_neg = exp_sum.mul(lam).pow(q-1.).mul(lam).unsqueeze(1).mul(exp_neg)

        d_pos = d_pos.mul(grad_output).unsqueeze(1)
        d_neg = d_neg.mul(grad_output.unsqueeze(1))

        return d_pos, d_neg, None, None


class RINCEV2(nn.Module):

    def __init__(self, q=0.5, lam=0.025):
        super(RINCEV2, self).__init__()
        self.q = q
        self.div_q = 1./q
        self.lam = lam

    def forward(self, pos, neg):
        pos = pos.float()
        neg = neg.float()
        exp_pos = pos.exp().squeeze(1)
        exp_sum = exp_pos + neg.exp().sum(dim=1)

        term1 = exp_pos.pow(self.q).neg()
        term2 = exp_sum.mul(self.lam).pow(self.q)
        loss = (term1 + term2).mul(self.div_q)
        return loss.mean()


#  class RINCE(nn.Module):
#
#      def __init__(self, q=0.5, lam=0.025):
#          super(RINCE, self).__init__()
#          self.q = q
#          self.div_q = 1./q
#          self.lam = lam
#
#      def forward(self, logits, labels):
#          N, *M = labels.size()
#          C = logits.size(1)
#          lb_one_hot = torch.zeros_like(logits).bool().scatter_(1, labels.unsqueeze(1), True).detach()
#          exp = logits.exp()
#          exp_pos = exp[lb_one_hot].view(N, *M)
#          exp_sum = exp.sum(dim=1)
#
#          term1 = exp_pos.pow(self.q).neg()
#          term2 = exp_sum.mul(self.lam).pow(self.q)
#          loss = (term1 + term2).mul(self.div_q)
#          return loss.mean()


if __name__ == "__main__":
    logits = torch.randn(3, 4,5 ,6)
    #  labels = torch.randint(0, 3, (3, 5, 6))
    labels = torch.zeros((3, 5, 6)).long()
    crit = RINCE()

    N, *M = labels.size()
    C = logits.size(1)
    lb_one_hot = torch.zeros_like(logits).bool().scatter_(1, labels.unsqueeze(1), True).detach()
    pos = logits[lb_one_hot].view(N, 1, *M)
    neg = logits[~lb_one_hot].view(N, C-1, *M)
    print(pos.size())
    print(neg.size())
    loss = crit(pos, neg)

    #  logits = torch.randn(2, 3,2)
    #  labels = torch.zeros((2, 2)).long()
    #  loss = crit(logits, labels)
    print(loss)

    #  crit = RINCEV2()
    #  print(crit(logits, labels))


    import torchvision
    import torch
    import numpy as np
    import random

    net1 = torchvision.models.resnet18(pretrained=False)
    net1.cuda()
    net1.train()
    net1.double()
    net2 = torchvision.models.resnet18(pretrained=False)
    net2.load_state_dict(net1.state_dict())
    net2.cuda()
    net2.train()
    net2.double()

    criteria1 = RINCE()
    criteria2 = RINCEV2()

    optim1 = torch.optim.SGD(net1.parameters(), lr=1e-3)
    optim2 = torch.optim.SGD(net2.parameters(), lr=1e-3)

    for it in range(1000):
        inten = torch.randn(16 ,3, 224, 224).cuda().double()

        logits1 = net1(inten)
        logits1 = logits1.tanh()
        pos1, neg1 = logits1[:, 0:1, ...], logits1[:, 1:, ...]
        loss1 = criteria1(pos1, neg1)
        optim1.zero_grad()
        loss1.backward()
        optim1.step()

        logits2 = net2(inten)
        logits2 = logits2.tanh()
        pos2, neg2 = logits2[:, 0:1, ...], logits2[:, 1:, ...]
        loss2 = criteria2(pos2, neg2)
        optim2.zero_grad()
        loss2.backward()
        optim2.step()
        with torch.no_grad():
            if (it+1) % 50 == 0:
                print('iter: {}, ================='.format(it+1))
                print('out.weight: ', torch.mean(torch.abs(net1.fc.weight - net2.fc.weight)).item())
                print('conv1.weight: ', torch.mean(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
                print('loss: ', loss1.item() - loss2.item())
                print('loss1: ', loss1.item())
                print('loss2: ', loss2.item())
