import torch
import torch.nn.functional as F
from copy import deepcopy


def get_label(args, logits, num_labels=-1):
    if args.loss_type == 'balance_softmax':
        th_logit = torch.zeros_like(logits[..., :1])
    else:
        th_logit = logits[:, 0].unsqueeze(1)
    output = torch.zeros_like(logits).to(logits)
    mask = (logits > th_logit)
    if num_labels > 0:
        top_v, _ = torch.topk(logits, num_labels, dim=1)
        top_v = top_v[:, -1]
        mask = (logits >= top_v.unsqueeze(1)) & mask
    output[mask] = 1.0
    output[:, 0] = (output.sum(1) == 0.).to(logits)
    return output


def get_at_loss(logits, labels):
    """
    ATL
    """
    labels = deepcopy(labels)
    # TH label
    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
    th_label[:, 0] = 1.0
    labels[:, 0] = 0.0
    p_mask = labels + th_label
    n_mask = 1 - labels
    # Rank positive classes to TH
    logit1 = logits - (1 - p_mask) * 1e30
    loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
    # Rank TH to negative classes
    logit2 = logits - (1 - n_mask) * 1e30
    loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
    # Sum two parts
    loss = loss1 + loss2
    loss = loss.mean()
    return loss


def get_balance_loss(logits, labels):
    """
    Balanced Softmax
    """
    y_true = labels
    y_pred = logits
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e30
    y_pred_pos = y_pred - (1 - y_true) * 1e30
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    loss = neg_loss + pos_loss
    loss = loss.mean()
    return loss


def get_af_loss(logits, labels):
    """
    AFL
    """
    # TH label
    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
    th_label[:, 0] = 1.0
    labels[:, 0] = 0.0
    n_mask = 1 - labels
    num_ex, num_class = labels.size()

    # Rank each class to threshold class TH
    th_mask = torch.cat(num_class * [logits[:, :1]], dim=1)
    logit_th = torch.cat([logits.unsqueeze(1), 1.0 * th_mask.unsqueeze(1)], dim=1)
    log_probs = F.log_softmax(logit_th, dim=1)
    probs = torch.exp(F.log_softmax(logit_th, dim=1))

    # Probability of relation class to be negative (0)
    prob_0 = probs[:, 1, :]
    prob_0_gamma = torch.pow(prob_0, 1)
    log_prob_1 = log_probs[:, 0, :]

    # Rank TH to negative classes
    logit2 = logits - (1 - n_mask) * 1e30
    rank2 = F.log_softmax(logit2, dim=-1)

    loss1 = - (log_prob_1 * (1 + prob_0_gamma) * labels)
    loss2 = -(rank2 * th_label).sum(1)

    loss = 1.0 * loss1.sum(1).mean() + 1.0 * loss2.mean()
    return loss


def get_sat_loss(logits, labels):
    """
    SAT
    """
    exp_th = torch.exp(logits[:, 0].unsqueeze(dim=1))

    p_prob = torch.exp(logits) / (torch.exp(logits) + exp_th)
    n_prob = exp_th / (exp_th + torch.exp(logits))

    p_num = labels[:, 1:].sum(dim=1)
    n_num = 96 - p_num

    p_item = -torch.log(p_prob + 1e-30) * labels
    p_item = p_item[:, 1:]
    n_item = -torch.log(n_prob + 1e-30) * (1 - labels)
    n_item = n_item[:, 1:]

    p_loss = p_item.sum(1)
    n_loss = n_item.sum(1)
    loss = p_loss + n_loss
    loss = loss.mean()
    return loss


def get_mean_sat_loss(logits, labels):
    exp_th = torch.exp(logits[:, 0].unsqueeze(dim=1))

    p_prob = torch.exp(logits) / (torch.exp(logits) + exp_th)
    n_prob = exp_th / (exp_th + torch.exp(logits))

    p_num = labels[:, 1:].sum(dim=1)
    n_num = 96 - p_num

    p_item = -torch.log(p_prob + 1e-30) * labels
    p_item = p_item[:, 1:]
    n_item = -torch.log(n_prob + 1e-30) * (1 - labels)
    n_item = n_item[:, 1:]

    p_loss = p_item.sum(1) / (p_num + 1e-30)
    n_loss = n_item.sum(1) / (n_num + 1e-30)
    loss = p_loss + n_loss
    loss = loss.mean()
    return loss


def get_relu_sat_loss(logits, labels, m=5):
    """
    HingeABL
    """
    p_num = labels[:, 1:].sum(dim=1)

    p_logits_diff = logits[:, 0].unsqueeze(dim=1) - logits
    p_logits_imp = F.relu(p_logits_diff + m)
    p_logits_imp = p_logits_imp * labels
    p_logits_imp = p_logits_imp[:, 1:]
    p_logits_imp = p_logits_imp / (p_logits_imp.sum(dim=1).unsqueeze(dim=1) + 1e-30)

    n_logits_diff = logits - logits[:, 0].unsqueeze(dim=1)
    n_logits_imp = F.relu(n_logits_diff + m)
    n_logits_imp = n_logits_imp * (1 - labels)
    n_logits_imp = n_logits_imp[:, 1:]
    n_logits_imp = n_logits_imp / (n_logits_imp.sum(dim=1).unsqueeze(dim=1) + 1e-30)

    exp_th = torch.exp(logits[:, 0].unsqueeze(dim=1))   # margin=5

    p_prob = torch.exp(logits) / (torch.exp(logits) + exp_th)
    n_prob = exp_th / (exp_th + torch.exp(logits))

    p_item = -torch.log(p_prob + 1e-30) * labels
    p_item = p_item[:, 1:] * p_logits_imp
    n_item = -torch.log(n_prob + 1e-30) * (1 - labels)
    n_item = n_item[:, 1:] * n_logits_imp

    p_loss = p_item.sum(1)
    n_loss = n_item.sum(1)
    loss = p_loss + n_loss
    loss = loss.mean()
    return loss


def get_margin_loss(logits, labels):
    """
    AML
    """
    # TH label
    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
    th_label[:, 0] = 1.0
    labels[:, 0] = 0.0

    # p_mask = labels + th_label
    # n_mask = 1 - labels
    p_mask = labels
    n_mask = 1 - labels
    n_mask[:, 0] = 0.0

    # Rank positive classes to TH
    # print('=====>', logits.shape, p_mask.shape)
    logit1 = logits - (1 - p_mask) * 1e30
    loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

    # Rank TH to negative classes
    logit2 = logits - (1 - n_mask) * 1e30
    loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

    logit3 = 1 - logits + logits[:, 0].unsqueeze(1)
    loss3 = (F.relu(logit3) * p_mask).sum(1)

    logit4 = 1 + logits - logits[:, 0].unsqueeze(1)
    loss4 = (F.relu(logit4) * n_mask).sum(1)

    # import ipdb; ipdb.set_trace()

    # Sum two parts
    # loss = loss1 + loss2 + loss3 + loss4
    # loss = loss3 + 0.5 * loss4
    loss = loss3 + loss4
    # loss = torch.sum(loss * r_mask) / torch.sum(r_mask)
    loss = loss.mean()
    return loss

