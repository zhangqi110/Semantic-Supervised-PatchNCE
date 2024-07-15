from cmath import exp
import torch
import numpy as np


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Home device: {}'.format(device))

# def get_negative_mask(batch_size):
#     negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
#     for i in range(batch_size):
#         negative_mask[i, i] = 0
#         negative_mask[i, i + batch_size] = 0

#     negative_mask = torch.cat((negative_mask, negative_mask), 0)
#     return negative_mask

# def criterion(out_1,out_2, batch_size, temperature):
#         # neg score
#         out = torch.cat([out_1, out_2], dim=0)
#         neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
#         old_neg = neg.clone()
#         mask = get_negative_mask(batch_size).to(device)
#         neg = neg.masked_select(mask).view(2 * batch_size, -1)

#         # pos score
#         pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
#         pos = torch.cat([pos, pos], dim=0)
        
#         # negative samples similarity scoring
        
#         Ng = neg.sum(dim=-1)
        
            
#         # contrastive loss
#         loss = (- torch.log(pos / (pos + Ng) )).mean()

#         return loss

from packaging import version
import torch
from torch import nn


class SSPLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt


    def compute_nceloss(self, feat_q, feat_k):

        feat_k = feat_k.detach()
        neg = torch.exp(torch.div(torch.matmul(feat_q, feat_k.T), self.opt.nce_T))
        # for numerical stability
        # logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        # logits = dot_contrast - logits_max.detach()
        # exp_logits = torch.exp(logits)
        mask = torch.ones((feat_q.shape[0], feat_q.shape[0])).to(feat_q.device) - torch.eye(feat_q.shape[0]).to(feat_q.device)

        neg = neg.masked_select(mask.bool()).view(feat_q.shape[0], -1)
        pos = torch.exp(torch.div(torch.sum(feat_q * feat_k, dim=-1) , self.opt.nce_T))


        Ng = neg.sum(dim=-1)

        return pos, Ng


    def forward(self, feat_q, feat_k, comp_f=None, out=False):

        # feat_q: 256 256
        # feqt_k 256 256
        # feat_comp_f 256 self.opt.patches 256
        
        pos, Ng = self.compute_nceloss(feat_q, feat_k)
        # pos_copy = pos.clone()
        pos_list = [pos]
        pos_sum = pos.clone()
        if comp_f is not None:
            for i in range(self.opt.ssp_patches):
                comp_f_now = comp_f[:,i,:].squeeze(1)
                pos_comp, Ng_comp = self.compute_nceloss(feat_q, comp_f_now)
                pos_sum = pos_sum + pos_comp
                pos_list.append(pos_comp)
                Ng = Ng + Ng_comp
        # negative samples similarity scoring
        loss = 0.0
        if out:
            for pos_now in pos_list:
                loss = loss + (- torch.log(pos_now / (pos_sum + Ng) )).mean()
        # contrastive loss
        else:
          loss = (- torch.log(pos_sum / (pos_sum + Ng) )).mean()

        return loss / len(pos_list)