from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss']

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = Variable(targets, requires_grad=False)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

def augment_surfaces(inputs, targets, surfaces, thresh_cos=0.95, aug_ratio=0.5):
    n = inputs.size(0)
    #print(n)
    #print('surfaces.size')
    #print(surfaces.size(0))
    #print(surfaces.size(1))
    n, d = surfaces.size(0), surfaces.size(1)

    '''surfaces_np = surfaces.data.cpu().numpy()
                mask = targets.expand(n, n).eq(targets.expand(n, n).t())
                mask_np = mask.data.cpu().numpy()
                import sklearn
                cosine_sim = sklearn.metrics.pairwise.cosine_similarity(surfaces_np,surfaces_np)
                cosine_sim -= mask_np.astype(np.float32)'''

    cosine_sim = F.cosine_similarity(surfaces.view(1, n, d).expand(n, n, d), surfaces.view(n, 1, d).expand(n, n, d), 2)
    '''cosine_sim = torch.pow(surfaces, 2).sum(dim=1, keepdim=True).expand(n, n)
                cosine_sim = cosine_sim + cosine_sim.t()
                cosine_sim.addmm_(1, -2, surfaces, surfaces.t())
                cosine_sim = cosine_sim.clamp(min=1e-12).sqrt()
                cosine_sim = 1 - cosine_sim
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                cosine_sim = cos(surfaces, surfaces)'''
    #print(cosine_sim.data.cpu().numpy())
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    #mask_np = mask.data.cpu().numpy()
    #print('mask_np')
    #print(mask_np)
    cosine_sim = cosine_sim - mask.float()
    m = nn.Threshold(thresh_cos, -1, inplace=True)
    cosine_sim = m(cosine_sim)
    cosine_sim = cosine_sim.data.cpu().numpy()
    targets_np = targets.data.cpu().numpy()
    #print('cosine_sim.shape')
    #print(cosine_sim.shape)
    #print(cosine_sim)
    #print('targets_np.shape')
    #print(targets_np.shape)
    #print(targets_np)
    num_pids = np.unique(targets_np).shape[0]
    #print('num_pids')
    #print(num_pids)
    aug_pairs = []
    aug_idxs = []
    inputs_aug = inputs.clone()
    while (np.max(cosine_sim) > thresh_cos and len(aug_pairs) < num_pids*aug_ratio):
        imax = np.argmax(cosine_sim)
        imax, jmax = np.unravel_index(imax, (n, n))
        i = targets_np[imax]
        j = targets_np[jmax]
        assert i != j
        aug_pairs.append((i,j))
        aug_pairs.append((j,i))
        idxi = np.where(targets_np == i)[0].tolist()
        idxj = np.where(targets_np == j)[0].tolist()
        aug_idxs.extend(idxi)
        aug_idxs.extend(idxj)       
        dfij = inputs[jmax,:] - inputs[imax,:]
        for idx in idxi:
            inputs_aug[idx,:] = inputs[idx,:] + dfij
            targets[idx] = j
            cosine_sim[idx,:] = -1
            cosine_sim[:,idx] = -1
        for idx in idxj:
            inputs_aug[idx,:] = inputs[idx,:] - dfij
            targets[idx] = i
            cosine_sim[idx,:] = -1
            cosine_sim[:,idx] = -1
    for idx in range(n):
        if idx not in aug_idxs:
            inputs_aug[idx,:] = inputs[idx,:]
    #print('aug_pairs')
    #print(aug_pairs)
    #print('aug_idxs')
    #print(aug_idxs)
    #targets_np = targets.data.cpu().numpy()
    #print(targets_np)

    return inputs_aug, targets

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, surfaces=None):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        if surfaces is not None:
            inputs, targets = augment_surfaces(inputs, targets, surfaces)
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

def _apply_margin(x, m):
    if isinstance(m, float):
        #return (x + m).clamp(min=0)
        return torch.mean((x + m).clamp(min=0))
    elif m.lower() == "soft":
        return F.softplus(x)
    elif m.lower() == "none":
        return x
    else:
        raise NotImplementedError("The margin %s is not implemented in BatchHard!" % m)

def batch_soft(cdist, pids, margin, T=1.0):
    """Calculates the batch soft.
    Instead of picking the hardest example through argmax or argmin,
    a softmax (softmin) is used to sample and use less difficult examples as well.
    Args:
        cdist (2D Tensor): All-to-all distance matrix, sized (B,B).
        pids (1D tensor): PIDs (classes) of the identities, sized (B,).
        margin: The margin to use, can be 'soft', 'none', or a number.
        T (float): The temperature of the softmax operation.
    """
    # mask where all positivies are set to true
    mask_pos = pids[None, :] == pids[:, None]
    mask_neg = 1 - mask_pos.data

    # only one copy
    cdist_max = cdist.clone()
    cdist_max[mask_neg] = -float('inf')
    cdist_min = cdist.clone()
    cdist_min[mask_pos] = float('inf')

    # NOTE: We could even take multiple ones by increasing num_samples,
    #       the following `gather` call does the right thing!
    idx_pos = torch.multinomial(F.softmax(cdist_max/T, dim=1), num_samples=1)
    idx_neg = torch.multinomial(F.softmin(cdist_min/T, dim=1), num_samples=1)
    positive = cdist.gather(dim=1, index=idx_pos)[:,0]  # Drop the extra (samples) dim
    negative = cdist.gather(dim=1, index=idx_neg)[:,0]

    return _apply_margin(positive - negative, margin)

class BatchSoft(nn.Module):
    """BatchSoft implementation using softmax.
    
    Also by Tristani as Adaptivei Weighted Triplet Loss.
    """

    def __init__(self, m, T=1.0, **kwargs):
        """
        Args:
            m: margin
            T: Softmax temperature
        """
        super(BatchSoft, self).__init__()
        self.name = "BatchSoft(m={}, T={})".format(m, T)
        self.m = m
        self.T = T

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        return batch_soft(dist, targets, self.m, self.T)

        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

    #def forward(self, dist, pids):
    #    return batch_soft(dist, pids, self.m, self.T)

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        classes = Variable(classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

if __name__ == '__main__':
    pass