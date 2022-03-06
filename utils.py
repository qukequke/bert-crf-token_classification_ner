# -*- coding: utf-8 -*-
import numpy as np
import os

import logging
from collections import Counter
from importlib import import_module

import torch
import torch.nn as nn
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from config import *


def eval_object(object_):
    if '.' in object_:
        module_, class_ = object_.rsplit('.', 1)
        module_ = import_module(module_)
        return getattr(module_, class_)
    else:
        module_ = import_module(object_)
        return module_


def generate_sent_masks(enc_hiddens, source_lengths):
    """ Generate sentence masks for encoder hidden states.
    @param enc_hiddens (Tensor): encodings of shape (b, src_len, h), where b = batch size,
                                 src_len = max source length, h = hidden size. 
    @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.len = batch size
    @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                where src_len = max source length, b = batch size.
    """
    enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.  shape [b, sl, out_dize]
        targets: The indices of the actual target classes. shape [b, sl]
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=-1)
    correct = (out_classes == targets).sum()
    return correct.item()


class SeqEntityScore(object):
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id2label, self.markup)
            pre_entities = get_entities(pre_path, self.id2label, self.markup)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])


def validate(model, dataloader):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    print('正在对验证集进行测试')
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    eval_loss = 0.0
    nb_eval_steps = 0
    # Deactivate autograd for evaluation.
    # pbar = ProgressBar(n_total=len(dataloader), desc="Evaluating")
    prefix = ''
    with torch.no_grad():
        for step, tokened_data_dict in enumerate(dataloader):
            tokened_data_dict = {k: v.to(device) for k, v in tokened_data_dict.items()}
            labels = tokened_data_dict['labels']
            tmp_eval_loss, logits = model(**tokened_data_dict)
            tags = model.crf.decode(logits, tokened_data_dict['attention_mask'])
            running_loss += tmp_eval_loss.item()

            # _, out_classes_raw = probabilities.max(dim=-1)
            out_classes_raw = tags[0] # [1, bs, num_lables]->[bs, num_labels] 第一代表1个序列解码出几个结果，一般是1个
            out_classes = [(torch.masked_select(i, mask.type(torch.bool)))[1:-1] for i, mask in
                           zip(out_classes_raw, tokened_data_dict['attention_mask'])]
            labels = [torch.masked_select(i, mask.type(torch.bool))[1:-1] for i, mask in
                      zip(labels, tokened_data_dict['attention_mask'])]
            all_prob.extend(out_classes)
            all_labels.extend(labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)

    all_prob = [[j.item() for j in i.cpu()] for i in all_prob]
    all_labels = [[j.item() for j in i.cpu()] for i in all_labels]
    # data_zip = [[i, j] for i, j in zip(all_prob, all_labels) if len(i) == len(j)]
    # pred_one_list = sum([i[0] for i in data_zip], start=[])
    # label_one_list = sum([i[1] for i in data_zip], start=[])
    # print(pred_one_list)
    # print(label_one_list)
    # epoch_accuracy = accuracy_score(all_labels, all_prob)
    # epoch_accuracy = accuracy_score(label_one_list, pred_one_list)
    # 评估
    label2id = load_json(json_dict)
    id2label = {v: k for k, v in label2id.items()}
    decode_labels = [[id2label[j].replace('M', 'I').replace('E', 'I') for j in i] for i in all_labels]
    decode_preds = [[id2label[j].replace('M', 'I').replace('E', 'I') for j in i] for i in all_prob]
    dict_all, dict_every_type = my_metrics(decode_labels, decode_preds)
    print(dict_all)
    print(dict_every_type)
    return epoch_time, epoch_loss, dict_all['f1']


def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    all_pred = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for tokened_data_dict in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            tokened_data_dict = {k: v.to(device) for k, v in tokened_data_dict.items()}
            labels = tokened_data_dict['labels']
            # seqs, masks, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_labels.to(device)
            _, _, probabilities = model(**tokened_data_dict)  # [batch_size, n_label]
            accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start
            # all_prob.extend(probabilities[:, 1].cpu().numpy())
            _, out_classes = probabilities.max(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_pred.extend(out_classes.cpu().numpy())
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, all_labels, all_pred


def seq_decode(tokenizer, data):
    decoded_preds = tokenizer.batch_decode(data, skip_special_tokens=True)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # decoded_labels = [''.join(i.split(' ')) for i in decoded_labels]
    decoded_preds = [''.join(i.split(' ')) for i in decoded_preds]
    return decoded_preds


import json


def load_json(file_name, value_is_array=False):
    with open(file_name, 'r', encoding='utf-8') as f:
        dict_ = json.load(f)
        return dict_


from sklearn.metrics import classification_report


def classication_report(true_labels, preds):
    label2id = load_json(json_dict)
    id2label = {v: k for k, v in label2id.items()}
    labels = list(set(true_labels))
    labels.sort(key=lambda x: true_labels.index(x))
    target_names = [id2label[i] for i in labels]
    print(classification_report(true_labels, preds, labels=labels, target_names=target_names))


def compute(origin, found, right):
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, f1


def my_metrics(decode_labels, decode_preds, id2label=None):
    markup = 'bio'
    origins, founds, rights = [], [], []
    for label_path, pre_path in zip(decode_labels, decode_preds):
        label_entities = get_entities(label_path, id2label, markup)
        pre_entities = get_entities(pre_path, id2label, markup)
        origins.extend(label_entities)
        founds.extend(pre_entities)
        rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])

    class_info = {}
    origin_counter = Counter([x[0] for x in origins])
    found_counter = Counter([x[0] for x in founds])
    right_counter = Counter([x[0] for x in rights])
    for type_, count in origin_counter.items():
        origin = count
        found = found_counter.get(type_, 0)
        right = right_counter.get(type_, 0)
        recall, precision, f1 = compute(origin, found, right)
        class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), 'num': count}
    origin = len(origins)
    found = len(founds)
    right = len(rights)
    recall, precision, f1 = compute(origin, found, right)
    return {'acc': precision, 'recall': recall, 'f1': f1, 'num': origin}, class_info


def calc_decode_acc(decode_labels, decode_preds):
    new_labels = []
    pre_alpha = 'BIO'
    pre_type = '症状234'
    start = 0
    for ii, i in enumerate(decode_labels):
        if i == 'O':
            if pre_alpha == 'B' or pre_alpha == 'I':
                #             print(new_labels)
                new_labels.append([pre_type, [start, ii]])
            #         continue
            pre_alpha = 'O'
            pre_type = ''
        else:
            # print(i)
            alpha, type_ = i.rsplit('-', 1)
            if alpha == 'B':
                if pre_alpha == 'I':
                    new_labels.append([pre_type, [start, ii]])
                start = ii
            pre_alpha = alpha
            pre_type = type_
    ## 得到labels的解码结果，每个实体和对应的位置，然后pred必须每个都和它一样
    # [['C', [1, 3]], ['病毒', [4, 10]], ['B', [10, 12]], ['aa', [17, 19]]]

    right_num = 0
    right_list = []
    for i in new_labels:
        all_right = True
        for ii, index_ in enumerate(range(i[1][0], i[1][1])):
            if decode_preds[index_] == 'O':
                all_right = False
            else:
                if ii == 0:
                    if decode_preds[index_] != f"B-{i[0]}":
                        all_right = False
                else:
                    if decode_preds[index_] != f"I-{i[0]}":
                        all_right = False
        if all_right:
            right_num += 1
            right_list.append(i[0])
    from collections import Counter
    acc = right_num / len(new_labels)
    right_counter = Counter(right_list)
    print(f"完整实体ACC为{acc}, 实体{right_num}/{len(new_labels)}, {right_counter}")
    # print(new_labels)
    return acc
    # print(right_list)
    # right_num


def ner_decode(data):
    label2id = load_json(json_dict)
    id2label = {v: k for k, v in label2id.items()}
    id2label[-100] = '-'
    # d = [[id2label[j.item()] for j in torch.masked_select(i, mask.type(torch.bool))] for i, mask in zip(data, attention_mask)]
    d = [[id2label[j.item()] for j in i] for i in data]
    d = [','.join(i) for i in d]
    return d


def test_data(model, dataloader, tokenizer):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    all_pred = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for tokened_data_dict in tqdm(dataloader):
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            tokened_data_dict = {k: v.to(device) for k, v in tokened_data_dict.items()}
            labels = tokened_data_dict.get('labels')
            # seqs, masks, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_labels.to(device)
            # _, _, probabilities = model(**tokened_data_dict)  # [batch_size, n_label]
            if labels is not None:
                loss, logits = model(**tokened_data_dict)
            else:
                logits = model(**tokened_data_dict)[0]
            tags = model.crf.decode(logits, tokened_data_dict['attention_mask'])
            # accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start
            # all_prob.extend(probabilities[:, 1].cpu().numpy())
            out_classes = tags[0]
            # decoded_preds = tokenizer.batch_decode(out_classes, skip_special_tokens=True)
            # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # decoded_labels = [''.join(i.split(' ')) for i in decoded_labels]
            # decoded_preds = [''.join(i.split(' ')) for i in decoded_preds]

            # out_classes = [(torch.masked_select(i, mask.type(torch.bool)))[:-2] for i, mask in zip(out_classes, tokened_data_dict['attention_mask'])]
            out_classes = [(torch.masked_select(i, mask.type(torch.bool)))[1:-1] for i, mask in
                           zip(out_classes, tokened_data_dict['attention_mask'])]
            # print(out_classes)
            if labels is not None:
                labels = [torch.masked_select(i, mask.type(torch.bool))[1:-1] for i, mask in
                          zip(labels, tokened_data_dict['attention_mask'])]
                all_labels.extend(labels)
            # decoded_preds = ner_decode(out_classes, tokened_data_dict['attention_mask'])
            # decoded_labels = ner_decode(labels, tokened_data_dict['attention_mask'])
            all_pred.extend(out_classes)
            # all_labels.extend(decoded_labels)
            # all_pred.extend(decoded_preds)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    # accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, all_labels, all_pred


def train(model, dataloader, optimizer, epoch_number, max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device
    model.to(device)
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (tokened_data_dict) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        tokened_data_dict = {k: v.to(device) for k, v in tokened_data_dict.items()}
        # labels = tokened_data_dict['labels']
        optimizer.zero_grad()
        # for k, v in tokened_data_dict.items():
        #     print(k)
        #     print(v)
        loss, logits = model(**tokened_data_dict)
        tqdm_batch_iterator.set_description(f'loss:{loss.cpu().item()}')
        # print(loss)
        # print(logits)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        # print(probabilities.shape)
        # print(labels.shape)
        # correct_preds += correct_predictions(probabilities, labels)
        # description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
        #     .format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
        # tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    # epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss


def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    # print(seq)
    # print(id2label)
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.rsplit('-', 1)[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.rsplit('-', 1)[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2label, markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


import sys
import time


class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='Training')
        >>> step = 2
        >>> pbar(step=step,info={'loss':20})
    '''

    def __init__(self, n_total, width=30, desc='Training', num_epochs=None):

        self.width = width
        self.n_total = n_total
        self.desc = desc
        self.start_time = time.time()
        self.num_epochs = num_epochs

    def reset(self):
        """Method to reset internal variables."""
        self.start_time = time.time()

    def _time_info(self, now, current):
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'
        return time_info

    def _bar(self, now, current):
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1: recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        return bar

    def epoch_start(self, current_epoch):
        sys.stdout.write("\n")
        if (current_epoch is not None) and (self.num_epochs is not None):
            sys.stdout.write(f"Epoch: {current_epoch}/{self.num_epochs}")
            sys.stdout.write("\n")

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        bar = self._bar(now, current)
        show_bar = f"\r{bar}" + self._time_info(now, current)
        if len(info) != 0:
            show_bar = f'{show_bar} ' + " [" + "-".join(
                [f' {key}={value:.4f} ' for key, value in info.items()]) + "]"
        if current >= self.n_total:
            show_bar += '\n'
        sys.stdout.write(show_bar)
        sys.stdout.flush()


def get_logger(fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger()
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    fh = logging.FileHandler('log.txt', encoding='utf-8')
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger



def save_json(file_name, dict_):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(dict_, f, ensure_ascii=False)


# def load_json(file_name, value_is_array=False):
#     with open(file_name, 'r', encoding='utf-8') as f:
#         dict_ = json.load(f)
#         return dict_
def get_max(x, y):
    max_x_index = np.argmax(y)
    max_x = x[max_x_index]
    max_y = y[max_x_index]
    return max_x, max_y
def my_plot(train_acc_list, losses):
    plt.figure()
    plt.plot(train_acc_list, color='r', label='dev_f1')
    x = [i for i in range(len(train_acc_list))]
    for add, list_ in enumerate([train_acc_list, ]):
        max_x, max_y = get_max(x, list_)
        plt.text(max_x, max_y, f'{(max_x, max_y)}')
        plt.vlines(max_x, min(train_acc_list), max_y, colors='r' if add==0 else 'b', linestyles='dashed')
        plt.hlines(max_y, 0, max_x, colors='r' if add==0 else 'b', linestyles='dashed')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(train_file), f'{MODEL}_dev_f1.png'))
    plt.figure()
    plt.plot(losses)
    plt.savefig(os.path.join(os.path.dirname(train_file), f'{MODEL}_loss.png'))
logger = get_logger()
