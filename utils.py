import torch
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.backends.cudnn as cudnn


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True


def collate_fn(batch):
    # time1 = time.time()
    # 将labels变成one-hot形式
    labels = []
    for f in batch:
        max_rlen = len(f['labels'][0]) if f['labels'] else 1
        neg_label = torch.zeros(f['neg_num'], max_rlen)
        relation = torch.cat((torch.tensor(f['labels']), neg_label), dim=0).long()
        neg_mask = torch.zeros(f['neg_num'], max_rlen)
        neg_mask[:, 0] = 1
        mask = torch.cat((torch.tensor(f['rmasks']), neg_mask), dim=0)
        f_labels = torch.zeros(f['pos_num']+f['neg_num'], 97)
        f_labels.scatter_add_(1, relation, mask)
        labels.append(f_labels)
        # f_labels = []
        # for label in f['labels']:
        #     relation = [0] * 97
        #     for r in label:
        #         relation[r] = 1
        #     f_labels.append(relation)
        # for i in range(f['neg_num']):
        #     relation = [1] + [0] * 96
        #     f_labels.append(relation)
        # labels.append(f_labels)
    labels = torch.cat(labels, dim=0)
    # labels = [f["labels"] for f in batch]
    # time2 = time.time()
    max_len = max([len(f["input_ids"]) for f in batch])
    # 粗暴地padding
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts)
    # time3 = time.time()
    # print("total time cost:{}, label time cost:{}".format(time3-time1, time2-time1))
    return output
