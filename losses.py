import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import cuda_as, categorical_entropy


def multiset_loss(scores, labels, preds):
    B, T, C = scores.size()
    loss = 0.0

    # `label_counts[i, j]` = # of free j labels in i'th example.
    label_counts = th.stack([cuda_as(th.zeros(B, C+1), labels).scatter_(1, labels.data[:, t].unsqueeze(1), 1)
                             for t in range(T)], 2).sum(2)[:, :-1]
    mask = (labels != scores.size(2)).float()

    for t in range(T):
        label_counts = Variable(cuda_as(label_counts, scores))
        p_hat = F.log_softmax(scores[:, t, :])
        p = label_counts / th.clamp(label_counts.sum(1, keepdim=True), 1.0).expand_as(label_counts)
        loss_ = (p*p_hat).sum(1, keepdim=True)
        loss_ = mask[:, t].unsqueeze(1)*loss_
        loss += loss_

        # Remove this step's predictions from the label counts (i.e. from free labels multiset)
        label_counts = label_counts.data + cuda_as(th.zeros((B, C)), preds).scatter_(1, preds.data[:, t].unsqueeze(1), -1)
        label_counts[label_counts < 0] = 0

    loss = loss / mask.sum(1, keepdim=True)
    loss = -loss.mean()
    return loss


def _distribution_matching(scores, labels, divergence_type):
    B, T, C = scores.size()

    # Make label distribution `Q`
    label_counts = th.stack([cuda_as(th.zeros(B, C+1), labels).scatter_(1, labels.data[:, t].unsqueeze(1), 1)
                             for t in range(T)], 2).sum(2)
    mask = cuda_as(th.cat((th.ones(B, C), th.zeros(B, 1)), 1), labels)  # Set STOP count to zero
    label_counts = Variable((label_counts * mask)[:, :-1])
    Q = label_counts / label_counts.sum(1, keepdim=True).expand_as(label_counts)

    # Make aggregated prediction distribution `P`
    scores_t = th.chunk(scores, T, 1)
    P = []
    label_mask = (labels != scores.size(2)).float()
    for t, st in enumerate(scores_t):
        st = F.softmax(st.squeeze(1)).unsqueeze(1)
        # Don't use probabilities from steps corresponding to END labels
        st = label_mask[:, t].unsqueeze(1).unsqueeze(1)*st

        P.append(st)
    P = th.sum(th.cat(P, 1), 1).squeeze(1)
    P = P / label_mask.sum(1, keepdim=True)

    if divergence_type == 'l1':
        loss = nn.L1Loss()(P, Q)
    elif divergence_type == 'kl':
        loss = nn.KLDivLoss()(th.log(P), Q)
    return loss


def l1_loss(scores, labels, _):
    return _distribution_matching(scores, labels, 'l1')


def kl_loss(scores, labels, _):
    return _distribution_matching(scores, labels, 'kl')


def _masked_ce(scores, labels, mask):
    # Manual until this issue is fixed: https://github.com/pytorch/pytorch/issues/563
    # `scores` (N, C), `labels` (N,), `mask` (N,)
    # Add extra column of zeros to account for the END class. Otherwise `gather` crashes
    # since the ground truth labels contain the END class.
    log_p = _pad_column(F.log_softmax(scores))
    log_py = th.gather(log_p, 1, labels.unsqueeze(1)).squeeze(1)
    log_py = mask*log_py
    denom = th.clamp(mask.sum(), 1.0)
    mean_loss = -log_py.sum()/denom
    return mean_loss


def ce_loss(scores, labels, placeholder):
    # For k labels, we mask k+1,...,T
    mask = (labels != scores.size(2)).float()
    B, T, _ = scores.size()
    loss = 0.0
    for t in range(T):
        loss += _masked_ce(scores[:, t, :], labels[:, t], mask[:, t])
    loss /= float(T)
    return loss


def _pad_column(tensor2d):
    B, C = tensor2d.size()
    scores = th.cat((tensor2d, Variable(cuda_as(th.zeros(B, 1), tensor2d))), 1)
    return scores


def rl(scores, labels, samples, values, entropy_coeff=.01):
    # For k labels, we mask k+1,...,T
    mask = (labels != scores.size(2)).float()
    B, T = labels.size()
    reward = _multiset_reward(samples.data, labels, C=scores.size(2))

    # Get log probabilities of sampled classes
    log_ps = []
    for t in range(T):
        log_ps.append(th.gather(F.log_softmax(scores[:, t, :]), 1, samples[:, t].unsqueeze(1)))
    log_ps = th.cat(log_ps, 1)

    advantage = Variable(reward - values.data)

    entropies = []
    for t in range(T):
        entropy = categorical_entropy(F.log_softmax(scores[:, t, :]))
        entropies.append(entropy)
    entropies = th.stack(entropies, 1)
    entropy_reg = entropies*mask

    pg_loss = -log_ps*advantage*mask - entropy_coeff*entropy_reg
    pg_loss = pg_loss.sum(1).mean()
    v_loss = ((values - Variable(reward))**2)*mask
    v_loss = v_loss.sum() / mask.sum()
    loss = pg_loss + v_loss
    return loss


def _multiset_reward(samples, labels, C):
    B, T = labels.size()
    # `label_counts[i, j]` = # of j'th label in i'th example.
    label_counts = th.stack([cuda_as(th.zeros(B, C+1), labels).scatter_(1, labels.data[:, t].unsqueeze(1), 1)
                             for t in range(T)], 2).sum(2)[:, :-1]
    rewards = cuda_as(th.zeros(samples.size()), samples)
    for t in range(T):
        rewards[:, t] = label_counts.gather(1, samples[:, t].unsqueeze(1))
        # Remove this step's samples from the label counts
        label_counts = label_counts + cuda_as(th.zeros((B, C)), samples).scatter_(1, samples[:, t].unsqueeze(1), -1)
        label_counts[label_counts < 0] = 0

    # Set to -1 or 1
    rewards[rewards > 0] = 1
    rewards[rewards <= 0] = -1
    rewards = rewards.float()
    return rewards


def stop_loss(stop, labels, stop_class):
    # For k labels, we keep k+1 (i.e. the labels plus 1st STOP label) and mask the rest (k+2,...,T).
    # Since we mask `pred` and `true`, the loss will be zero at masked locations.
    mask = 1 - (((labels == stop_class).cumsum(1).float() - 1) > 0).float()
    pred = stop
    true = (labels == stop_class).float()*mask

    # Element-wise loss (from F.binary_cross_entropy_with_logits)
    max_val = (-pred).clamp(min=0)
    loss = pred - pred*true + max_val + ((-max_val).exp() + (-pred - max_val).exp()).log()

    # Mask and average
    loss = loss*mask
    loss = (loss.sum(1)/th.clamp(mask.sum(1), 1.0)).mean()
    return loss
