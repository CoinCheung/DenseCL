

import torch
import torch.nn as nn



class CutMixer(nn.Module):

    def __init__(self, divisor=32, lower=3, upper=6, T=0.07):
        super(CutMixer, self).__init__()
        '''
        cropsize is within [lower*32, upper*32)
        '''
        self.divisor = divisor
        self.lower, self.upper = lower, upper
        self.T = T
        self.crit = nn.CrossEntropyLoss()

    ## TODO: see if use different crop for each images
    @torch.no_grad()
    def mix_img(self, ims):
        bs, C, H, W = ims.size()
        assert H % self.divisor == 0 and W % self.divisor == 0
        nh, nw = H // self.divisor, W // self.divisor
        h, w = torch.randint(self.lower, self.upper, (2, )).tolist()
        hst = torch.randint(0, nh-h, (1,))[0]
        wst = torch.randint(0, nw-w, (1,))[0]
        hst_, wst_ = hst * self.divisor, wst * self.divisor
        h_, w_ = h * self.divisor, w * self.divisor

        perm = torch.randperm(bs).cuda()
        perm_unshuf = perm.argsort()
        ims_mix = ims.clone()
        ims_mix[:, :, hst_:hst_+h_, wst_:wst_+w_] = ims[perm, :, hst_:hst_+h_, wst_:wst_+w_]
        return ims_mix.detach(), perm, perm_unshuf, h, w, hst, wst

    def forward_mix(self, model, ims, q, k, queue):
        mix_res = self.mix_img(ims)
        ims_mix, perm, perm_unshuf, h, w, hst, wst = mix_res

        p, c = model.forward_cutmix(mix_res)
        p = nn.functional.normalize(p, dim=1)
        c = nn.functional.normalize(c, dim=1)
        p_unshuf = p[perm_unshuf]

        queue = queue.clone().detach()
        p_pos = torch.einsum('nc,nc->n', p_unshuf, k).unsqueeze(-1)
        p_neg1 = torch.einsum('nc,ck->nk', p_unshuf, queue)
        p_neg2 = torch.einsum('nc,nc->n', p_unshuf, c[perm].detach()).unsqueeze(-1)
        p_neg = torch.cat([p_neg1, p_neg2], dim=1)

        c_pos = torch.einsum('nc,nc->n', c, k).unsqueeze(-1)
        c_neg1 = torch.einsum('nc,ck->nk', c, queue)
        c_neg2 = torch.einsum('nc,nc->n', c, p.detach()).unsqueeze(-1)
        c_neg = torch.cat([c_neg1, c_neg2], dim=1)

        cp_pos = torch.cat([p_pos, c_pos], dim=0)
        cp_neg = torch.cat([p_neg, c_neg], dim=0)

        logits = torch.cat([cp_pos, cp_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.crit(logits, labels)
        return loss




