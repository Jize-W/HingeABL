import ujson as json
import numpy as np
from prepro import read_docred
from torch.utils.data import DataLoader
from utils import collate_fn
from tqdm import tqdm
docred_rel2id = json.load(open('meta/rel2id.json', 'r'))


def visualize_doc(file_in, file_out, id_list):
    rel_info_file = '/home/ubuntu/SemiSup_DocRE/dataset/docred/rel_info.json'
    with open(rel_info_file, "r") as f:
        rel_info = json.load(f)
    with open(file_in, "r") as fh:
        data = json.load(fh)
    docs = list()
    for i in id_list:
        sample = dict()
        sample['id'] = i
        sample['title'] = data[i]['title']
        sents = data[i]['sents']
        vertexset = data[i]['vertexSet']
        labels = data[i]['labels']
        sample['sents'] = '\n'.join([' '.join(sents[k]) for k in range(len(sents))])
        id2entity = dict()
        for k in range(len(vertexset)):
            id2entity[k] = vertexset[k][0]['name']
        sample['entities'] = id2entity
        sample_labels = list()
        for label in labels:
            sample_label = dict()
            sample_label['subject'] = vertexset[label['h']][0]['name']
            sample_label['object'] = vertexset[label['t']][0]['name']
            sample_label['relation'] = rel_info[label['r']]
            # sample_label['h'] = [vertexset[label['h']][k]['name'] for k in range(len(vertexset[label['h']]))]
            # sample_label['t'] = [vertexset[label['t']][k]['name'] for k in range(len(vertexset[label['t']]))]
            sample_labels.append(sample_label)
        sample['instances'] = sample_labels
        docs.append(sample)
    with open(file_out, "w") as f:
        json.dump(docs, f)
    return


def id2info(file_rel2id, file_rel2info, file_out):
    with open(file_rel2id, "r") as f:
        rel2id = json.load(f)
    with open(file_rel2info, "r") as f:
        rel2info = json.load(f)
    id2info = dict()
    for rel in rel2id.keys():
        if rel == 'Na':
            continue
        id2info[rel2id[rel]] = rel2info[rel]
    id2info_sorted = dict()
    for i in range(1,97):
        id2info_sorted[i] = id2info[i]
    with open(file_out, "w") as f:
        json.dump(id2info_sorted, f)
    return


def visualize_predicts(file_out, hts, dl, lt):
    predicts = dict()
    for i in range(len(hts)):
        distant_label = np.nonzero(dl[i] == 1).squeeze(dim=1).cpu().numpy().tolist()
        teacher_label = np.nonzero(lt[i] > lt[i][0]).squeeze(dim=1).cpu().numpy().tolist()
        teacher_logit = lt[i].cpu().numpy().tolist()
        distant_label_logit = []
        distant_label_rank = []
        threshold_logit = teacher_logit[0]
        for label in distant_label:
            count = 0
            distant_label_logit.append(teacher_logit[label])
            for k in range(1, len(teacher_logit)):
                if k == label:
                    continue
                if teacher_logit[k] > teacher_logit[label]:
                    count += 1
            distant_label_rank.append(count+1)
        predict = dict()
        predict['hts'] = hts[i]
        predict['distant_label'] = distant_label
        predict['dl_logit'] = distant_label_logit
        predict['dl_rank'] = distant_label_rank
        predict['th_logit'] = threshold_logit
        predict['teacher_label'] = teacher_label
        predicts[i] = predict
    with open(file_out, "w") as f:
        json.dump(predicts, f)
    return


def visualize_test_result(file_out, hts, logits, labels):
    predicts = dict()
    for i in range(len(hts)):
        label = np.nonzero(labels[i] == 1).squeeze(dim=1).cpu().numpy().tolist()
        pred = np.nonzero(logits[i] > logits[i][0]).squeeze(dim=1).cpu().numpy().tolist()
        logit = logits[i].cpu().numpy().tolist()
        label_logit = []
        label_rank = []
        threshold_logit = logit[0]
        for lb in label:
            count = 0
            label_logit.append(logit[lb])
            for k in range(1, len(logit)):
                if k == lb:
                    continue
                if logit[k] > logit[lb]:
                    count += 1
            label_rank.append(count+1)
        predict = dict()
        predict['hts'] = hts[i]
        predict['pred'] = pred
        predict['label'] = label
        predict['lb_logit'] = label_logit
        predict['lb_rank'] = label_rank
        predict['th_logit'] = threshold_logit
        predicts[i] = predict
    with open(file_out, "w") as f:
        json.dump(predicts, f)
    return


def visualize_dev_wrong(dev_file, file_out, model, tokenizer):
    dev_features = read_docred(dev_file, tokenizer, max_seq_length=1024)
    dev_dataloader = DataLoader(dev_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=True)
    wrong_result = dict()
    for idx, batch in enumerate(dev_dataloader):
        predicts = dict()
        inputs = {'input_ids': batch[0].to('cuda'),
                  'attention_mask': batch[1].to('cuda'),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }
        labels = batch[2].to('cuda')
        logits = model(**inputs)
        hts = inputs['hts'][0]
        for i in range(len(hts)):
            label = np.nonzero(labels[i] == 1).squeeze(dim=1).cpu().numpy().tolist()
            pred = np.nonzero(logits[i] > logits[i][0]).squeeze(dim=1).cpu().numpy().tolist()
            if label == [0] and pred == []:
                continue
            if set(label) == set(pred):
                continue
            logit = logits[i].cpu().numpy().tolist()
            label_logit = []
            label_rank = []
            threshold_logit = logit[0]
            for lb in label:
                count = 0
                label_logit.append(logit[lb])
                for k in range(1, len(logit)):
                    if k == lb:
                        continue
                    if logit[k] > logit[lb]:
                        count += 1
                label_rank.append(count+1)
            predict = dict()
            predict['hts'] = hts[i]
            predict['pred'] = pred
            predict['label'] = label
            predict['lb_logit'] = label_logit
            predict['lb_rank'] = label_rank
            predict['th_logit'] = threshold_logit
            predicts[i] = predict
        wrong_result[idx] = predicts
    with open(file_out, "w") as f:
        json.dump(wrong_result, f)


def visualize_dev(dev_file, file_out_prefix, model, tokenizer):
    dev_features = read_docred(dev_file, tokenizer, max_seq_length=1024)
    dev_dataloader = DataLoader(dev_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=True)
    tp, tn, fp, fn_rank_wrong, fn_rank_right = [], [], [], [], []
    num, num_tp, num_tn, num_fp, num_fn_rank_wrong, num_fn_rank_right = 0, 0, 0, 0, 0, 0
    num_tp_exp, num_tn_exp, num_fp_exp, num_fn_rank_wrong_exp, num_fn_rank_right_exp = 0, 0, 0, 0, 0
    for idx, batch in enumerate(dev_dataloader):
        inputs = {'input_ids': batch[0].to('cuda'),
                  'attention_mask': batch[1].to('cuda'),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }
        labels = batch[2].to('cuda')
        logits = model(**inputs)
        hts = inputs['hts'][0]
        input_ids = inputs['input_ids'][0]
        entity_pos = inputs['entity_pos'][0]
        for i in range(len(hts)):
            num += 1
            label = np.nonzero(labels[i] == 1).squeeze(dim=1).detach().cpu().numpy().tolist()
            pred = np.nonzero(logits[i] > logits[i][0]).squeeze(dim=1).detach().cpu().numpy().tolist()
            logit = logits[i].detach().cpu().numpy().tolist()
            item = dict()
            item['batch_id'] = idx
            item['hts_id'] = i
            item['text'] = tokenizer.decode(input_ids)
            item['hts'] = hts[i]
            h_pos = entity_pos[hts[i][0]][0]    # 选第0个mention可视化一下
            t_pos = entity_pos[hts[i][1]][0]
            item['subject'] = tokenizer.decode(input_ids[h_pos[0] + 1: h_pos[1] + 1])
            item['object'] = tokenizer.decode(input_ids[t_pos[0] + 1: t_pos[1] + 1])
            item['logit'] = logit
            item['label'] = label
            item['pred'] = pred
            item['th_logit'] = logit[0]
            if label == [0] and pred == []:
                # tn
                num_tn += 1
                # if num_tn_exp <= 20:
                if idx == 9 and num_tn_exp <= 5:
                    num_tn_exp += 1
                    tn.append(item)
                continue
            label_logit = []
            label_rank = []
            for lb in label:
                count = 0
                label_logit.append(logit[lb])
                for k in range(1, len(logit)):
                    if k == lb:
                        continue
                    if logit[k] > logit[lb]:
                        count += 1
                label_rank.append(count + 1)
            if set(label) == set(pred):
                # tp
                num_tp += 1
                # if len(label) > 1 and num_tp_exp <= 20:
                if idx == 9 and num_tp_exp <= 5:
                    num_tp_exp += 1
                    item['label_logit'] = label_logit
                    tp.append(item)
                continue
            if pred == []:
                # fn
                if set(label_rank) in [{1}, {1, 2}, {1, 2, 3}]:
                    num_fn_rank_right += 1
                    # if len(label) > 1 and num_fn_rank_right_exp <= 20:
                    if idx == 9:
                        num_fn_rank_right_exp += 1
                        item['label_logit'] = label_logit
                        item['label_rank'] = label_rank
                        fn_rank_right.append(item)
                else:
                    num_fn_rank_wrong += 1
                    # if len(label) > 1 and num_fn_rank_wrong_exp <= 20:
                    if idx == 9:
                        num_fn_rank_wrong_exp += 1
                        item['label_logit'] = label_logit
                        item['label_rank'] = label_rank
                        fn_rank_wrong.append(item)
            else:
                # fp
                num_fp += 1
                # if len(label) > 1 and num_fp_exp <= 20:
                if idx == 9:
                    num_fp_exp += 1
                    item['label_logit'] = label_logit
                    item['label_rank'] = label_rank
                    pred_logit = []
                    for p in pred:
                        pred_logit.append(logit[p])
                    item['pred_logit'] = pred_logit
                    fp.append(item)
    assert num == num_tp + num_tn + num_fp + num_fn_rank_wrong + num_fn_rank_right
    stat = dict()
    stat['num'], stat['num_tp'], stat['num_tn'], stat['num_fp'], stat['num_fn_rank_wrong'], \
        stat['num_fn_rank_right']= num, num_tp, num_tn, num_fp, num_fn_rank_wrong, num_fn_rank_right
    with open(file_out_prefix+'_stat.json', 'w') as f:
        json.dump(stat, f)
    with open(file_out_prefix+'_tp.json', 'w') as f:
        json.dump(tp, f)
    with open(file_out_prefix+'_tn.json', 'w') as f:
        json.dump(tn, f)
    with open(file_out_prefix+'_fp.json', 'w') as f:
        json.dump(fp, f)
    with open(file_out_prefix+'_fn_rank_wrong.json', 'w') as f:
        json.dump(fn_rank_wrong, f)
    with open(file_out_prefix+'_fn_rank_right.json', 'w') as f:
        json.dump(fn_rank_right, f)


def visualize_dev_example(dev_file, file_out_prefix, model, tokenizer, batch_id, hts_id):
    dev_features = read_docred(dev_file, tokenizer, max_seq_length=1024)
    dev_dataloader = DataLoader(dev_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=True)
    for idx, batch in enumerate(dev_dataloader):
        if idx != batch_id:
            continue
        else:
            inputs = {'input_ids': batch[0].to('cuda'),
                      'attention_mask': batch[1].to('cuda'),
                      'entity_pos': batch[3],
                      'hts': batch[4],
                      }
            labels = batch[2].to('cuda')
            logits = model(**inputs)
            hts = inputs['hts'][0]
            input_ids = inputs['input_ids'][0]
            entity_pos = inputs['entity_pos'][0]
            for i in range(len(hts)):
                if i != hts_id:
                    continue
                else:
                    label = np.nonzero(labels[i] == 1).squeeze(dim=1).detach().cpu().numpy().tolist()
                    pred = np.nonzero(logits[i] > logits[i][0]).squeeze(dim=1).detach().cpu().numpy().tolist()
                    logit = logits[i].detach().cpu().numpy().tolist()
                    item = dict()
                    item['batch_id'] = idx
                    item['hts_id'] = i
                    item['text'] = tokenizer.decode(input_ids)
                    item['hts'] = hts[i]
                    h_pos = entity_pos[hts[i][0]][0]    # 选第0个mention可视化一下
                    t_pos = entity_pos[hts[i][1]][0]
                    item['subject'] = tokenizer.decode(input_ids[h_pos[0] + 1: h_pos[1] + 1])
                    item['object'] = tokenizer.decode(input_ids[t_pos[0] + 1: t_pos[1] + 1])
                    item['logit'] = logit
                    item['label'] = label
                    item['pred'] = pred
                    item['th_logit'] = logit[0]
                    label_logit = []
                    label_rank = []
                    for lb in label:
                        count = 0
                        label_logit.append(logit[lb])
                        for k in range(1, len(logit)):
                            if k == lb:
                                continue
                            if logit[k] > logit[lb]:
                                count += 1
                        label_rank.append(count + 1)
                    item['label_logit'] = label_logit
                    item['label_rank'] = label_rank
                    with open(file_out_prefix+'_example.json','w') as f:
                        json.dump(item, f)
                    break
            break


def relation_hist(file_in):
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)
    relation_hist = [0] * 97
    for sample in tqdm(data, desc="Example"):
        # label里给的三元组，以字典形式收集，train_triple[(h,t)]=[{r,evidence},...]
        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'evidence': evidence}]
                    relation_hist[r] += 1
                else:
                    train_triple[(label['h'], label['t'])].append({'relation': r, 'evidence': evidence})
                    relation_hist[r] += 1
    return relation_hist
