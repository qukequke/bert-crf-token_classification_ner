# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import pandas as pd
import torch
from config import *
from utils import load_json

label_2_id = load_json(json_dict)


class DataPrecessForSentence(Dataset):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, LCQMC_file):
        """
        bert_tokenizer :分词器
        LCQMC_file     :语料文件
        """
        self.bert_tokenizer = bert_tokenizer
        self.data = self.get_input(LCQMC_file)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]
        return {k: v[idx] for k, v in self.data.items()}

    # 获取文本与标签
    def get_input(self, file):
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
        if n_nums:
            df = pd.read_csv(file, engine='python', encoding=csv_encoding, error_bad_lines=False, nrows=n_nums)
        else:
            df = pd.read_csv(file, engine='python', encoding=csv_encoding, error_bad_lines=False)
        # print(df)
        self.length = len(df)
        self.bert_tokenizer.model_max_length = max_seq_len
        print(f"数据集个数为{len(df)}")
        sentences = df[csv_rows[0]].tolist()
        # print(self.bert_tokenizer)
        for ii, i in enumerate(sentences):
            assert isinstance(i, str), f"{i}, {ii}"

        data = self.my_bert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # 返回结果为类字典 {'input_ids':[[1,1,1,], [1,2,1]], 'token_type_ids':矩阵, 'attention_mask':矩阵,...}
        # from torch.utils.t
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
        sentences = [i[:max_len-2] for i in sentences]
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


if __name__ == '__main__':
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader
    bert_path_or_name = model_dict[MODEL][-1]
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path_or_name)
    dataset = DataPrecessForSentence(bert_tokenizer, train_file)
    # for i in dataset:
    #     print(i)
    #     break
    # print(len(dataset))
    # d = DataLoader(dataset, batch_size=20)
    # print(len(d))
    # for ii, i in enumerate(d):
    #     print(i)
    #     print(ii)
    #     break

    # tokenizer(["Hello, my dog is cute", 'plase call me shen'], return_tensors="pt", padding=True, truncation=True)
