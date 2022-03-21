# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2022/3/6 9:40
@Author  : quke
@File    : infer.py
@Description:
---------------------------------------
'''
from collections import Counter

import time
from flask import Flask, request

from sys import platform

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

from config import *
from utils import load_json

label_2_id = load_json(json_dict)

dir_name = 'cner'
target_file = f'models/{dir_name}/best.pth.tar'  # 模型存储路径
label_file = f'data/{dir_name}/label2id.json'
bert_path_or_name = 'bert-base-chinese'  # 使用模型
batch_size = 64
csv_rows = ['raw_sen', 'label']
max_seq_len = 150


# problem_type = 'multi_label_classification'
# problem_type = 'single_label_classification'  # 单分类


class DataPrecessForSentence(Dataset):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, LCQMC_file, is_sentence=False):
        """
        bert_tokenizer :分词器
        LCQMC_file     :语料文件
        """
        self.bert_tokenizer = bert_tokenizer
        self.data = self.get_input(LCQMC_file, is_sentence)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]
        return {k: v[idx] for k, v in self.data.items()}

    # 获取文本与标签
    def get_input(self, file, is_sentence=False):
        """
        通对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        入参:
            dataset     : pandas的dataframe格式，包含三列，第一,二列为文本，第三列为标签。标签取值为{0,1}，其中0表示负样本，1代表正样本。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
            labels      : 标签取值为{0,1}，其中0表示负样本，1代表正样本。
        """
        if isinstance(file, list):
            df = pd.DataFrame([{'raw_sen': i} for i in file])
        else:
            if n_nums:
                df = pd.read_csv(file, engine='python', encoding=csv_encoding, error_bad_lines=False, nrows=n_nums)
            else:
                df = pd.read_csv(file, engine='python', encoding=csv_encoding, error_bad_lines=False)
            # print(df)
        self.bert_tokenizer.model_max_length = max_seq_len
        print(f"数据集个数为{len(df)}")
        sentences = df[csv_rows[0]].tolist()
        self.length = len(df)
        # print(self.bert_tokenizer)
        for ii, i in enumerate(sentences):
            assert isinstance(i, str), f"{i}, {ii}"

        data = self.my_bert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # 返回结果为类字典 {'input_ids':[[1,1,1,], [1,2,1]], 'token_type_ids':矩阵, 'attention_mask':矩阵,...}
        seq_len = data['input_ids'].size(1)
        if csv_rows[-1] in df.columns:
            labels = df[csv_rows[-1]].tolist()
            self.labels = labels.copy()
            labels = [eval(i) for i in labels]
            labels = [(i + ([label_2_id['pad'], ] * (seq_len - len(i)))) if seq_len > len(i) else i[:seq_len]
                      for i in labels]  # pad labels
            labels = torch.Tensor(labels).type(torch.long)
            data['labels'] = labels
        print('输入例子')
        print(sentences[0] if isinstance(sentences[0], str) else sentences[0][0])
        for k, v in data.items():
            print(k)
            print(v[0])
        print(f"实际序列转换后的长度为{len(data['input_ids'][0])}, 设置最长为{max_seq_len}")
        return data

    def my_bert_tokenizer(self, sentences, padding, truncation, return_tensors):
        """
        自己实现的类似tokenizer.__call__结果, 返回字典里面包括Input_ids和attention_mask
        """
        sen_max_len = max([len(i) for i in sentences])
        max_len = min(max_seq_len, sen_max_len)
        sentences = [i[:max_len - 2] for i in sentences]
        sentences_list = [['[CLS]'] + [i for i in sen] + ['[SEP]'] for sen in sentences]
        input_ids = [self.bert_tokenizer.convert_tokens_to_ids(sen) for sen in sentences_list]
        # print(input_ids)
        # print(f"max_len 为{max_len}")
        attention_mask = [[1, ] * len(i) + [0, ] * (max_len - len(i)) for i in input_ids]
        input_ids = [i + [0, ] * (max_len - len(i)) for i in input_ids]
        # for i, j in zip(input_ids, attention_mask):
        #     print(len(i))
        #     print(len(j))
        data_dict = {'attention_mask': attention_mask, 'input_ids': input_ids}
        data_dict = {k: torch.Tensor(v).type(torch.long) for k, v in data_dict.items()}
        # for i in input_ids:
        #     a = self.bert_tokenizer.decode(i)
        #     print(a)
        return data_dict


from torch import nn
from config import *
from utils import eval_object
import torch
from typing import List, Optional

model_dict = {
    'bert': ('transformers.BertTokenizer',
             'transformers.BertModel',
             'transformers.BertConfig',
             'bert-base-chinese'  # 使用模型
             ),

    'roberta': (
        'transformers.BertTokenizer',
        'transformers.RobertaModel',
        'transformers.RobertaConfig',
        'hfl/chinese-roberta-wwm-ext',
    ),
    'albert': ('transformers.BertTokenizer',
               'transformers.AlbertModel',
               'transformers.AlbertConfig'
               ),
}
MODEL = 'bert'
ClassifyClass = eval_object(model_dict[MODEL][1])
ClassifyConfig = eval_object(model_dict[MODEL][2])


class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = 'mean') -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None,
               nbest: Optional[int] = None,
               pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            nbest (`int`): Number of most probable paths for each sequence
            pad_tag (`int`): Tag at padded positions. Often input varies in length and
                the length will be padded to the maximum length in the batch. Tags at
                the padded positions will be assigned with a padding tag, i.e. `pad_tag`
        Returns:
            A PyTorch tensor of the best tag sequence for each batch of shape
            (nbest, batch_size, seq_length)
        """
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(self, emissions: torch.Tensor,
                  tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor,
                       tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        seq_length, batch_size = tags.shape
        mask = mask.float()

        score = self.start_transitions[tags[0]]  # [batch_size,]
        score += emissions[0, torch.arange(batch_size), tags[0]]  # [batch_size]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor,
                        pad_tag: Optional[int] = None) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size), pad_tag,
                             dtype=torch.long, device=device)

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score
        # shape: (batch_size, num_tags)
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
                             end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size),
                                    dtype=torch.long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)

        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor,
                              mask: torch.ByteTensor,
                              nbest: int,
                              pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags, nbest),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest), pad_tag,
                             dtype=torch.long, device=device)

        # + score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # + history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)

            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags, nbest)
            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size, nbest),
                                    dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device) \
            .view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest

        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)


class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = ClassifyClass.from_pretrained(bert_path_or_name)
        # self.bert = AutoModelForSequenceClassification.from_pretrained(bert_path_or_name, num_labels=10)
        self.device = torch.device("cuda")
        # self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, **input_):
        if 'labels' in input_:
            labels = input_.pop('labels', '')
        else:
            labels = None
        bert_output = self.bert(**input_)  # [bs, seq_len, num_labels]
        sequence_output = bert_output[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=input_['attention_mask'])
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores


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


def compute(origin, found, right):
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, f1


def data_t(model, dataloader, tokenizer):
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
            batch_time += time.time() - batch_start
            out_classes = tags[0]

            out_classes = [(torch.masked_select(i, mask.type(torch.bool)))[1:-1] for i, mask in
                           zip(out_classes, tokened_data_dict['attention_mask'])]
            if labels is not None:
                labels = [torch.masked_select(i, mask.type(torch.bool))[1:-1] for i, mask in
                          zip(labels, tokened_data_dict['attention_mask'])]
                all_labels.extend(labels)
            all_pred.extend(out_classes)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    # accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, all_labels, all_pred


def ner_decode(data):
    label2id = load_json(json_dict)
    id2label = {v: k for k, v in label2id.items()}
    id2label[-100] = '-'
    # d = [[id2label[j.item()] for j in torch.masked_select(i, mask.type(torch.bool))] for i, mask in zip(data, attention_mask)]
    d = [[id2label[j.item()] for j in i] for i in data]
    d = [','.join(i) for i in d]
    return d


class Inferenve:
    def __init__(self):
        label2id = load_json(json_dict)
        self.id2label_dict = {v: k for k, v in label2id.items()}
        self.device = torch.device("cuda")
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path_or_name)
        print(20 * "=", " Preparing for testing ", 20 * "=")
        print(target_file)
        if platform == "linux" or platform == "linux2":
            checkpoint = torch.load(target_file)
        else:
            checkpoint = torch.load(target_file, map_location=self.device)
        # Retrieving model parameters from checkpoint.
        self.model = BertModel().to(self.device)
        self.model.load_state_dict(checkpoint["model"])

    def get_ret(self, sentences):
        if isinstance(sentences, str):
            sentences = list(sentences)
        test_dataset = DataPrecessForSentence(self.bert_tokenizer, sentences)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        # for i in test_loader:
        #     print(i)
        batch_time, total_time, accuracy, all_labels, all_pred = data_t(self.model, test_loader, self.bert_tokenizer)
        # decoded_preds = ner_decode(all_pred)
        all_pred = [[j.item() for j in i.cpu()] for i in all_pred]
        label_entities = [get_entities(i, self.id2label_dict, 'bio') for i in all_pred]
        return label_entities


from flask_restplus import Api, Resource, reqparse

app = Flask(__name__)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add("Expries", -1)
    response.headers.add("Cache-Control", "no-cache")
    response.headers.add("Pragma", "no-cache")
    return response


api = Api(app, version='1.0', title='api调用', description='python swagger', )
ns = api.namespace('api', description='实体识别')  # 模块命名空间

tts_parser = reqparse.RequestParser()  # 参数模型
tts_parser.add_argument('text', type=str, required=True, help="想要转语音的文字", default='高勇：男，中国国籍')

infer = Inferenve()


class Ner(Resource):
    @ns.expect(tts_parser)
    def get(self):
        """
        实体识别
        """
        d = request.args
        text = d.get('text')
        ret = infer.get_ret([text, ])[0]
        ret = [[i[0], i[1], i[2] + 1] for i in ret]
        return {'data': ret, 'is_success': 1}

ns.add_resource(Ner, "/ner", endpoint="ner")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)