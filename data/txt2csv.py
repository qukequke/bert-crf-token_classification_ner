# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/11/29 16:39
@Author  : quke
@File    : txt2csv.py
@Description:
---------------------------------------
'''

# from config import *
import os

import json

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


def save_json(file_name, dict_):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(dict_, f, ensure_ascii=False)


def load_json(file_name, value_is_array=False):
    with open(file_name, 'r', encoding='utf-8') as f:
        dict_ = json.load(f)
        return dict_


def get_dict(file_name, dict_out, split_=' '):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = [i.split(split_) for i in f.readlines() if len(i.split(split_)) == 2]
        document_pair = [[i[0], i[1].strip()] for i in data]
    label_list = [i[1] for i in document_pair]
    all_label_set = list(set(label_list))
    all_label_set.sort(key=lambda x: label_list.index(x))
    label_2_id = {j: i for i, j in enumerate(all_label_set)}
    save_json(dict_out, label_2_id)


def combine(sen_list, label_list):
    """
    合并长度太短的 和 不一O或B开始的, 去掉全为O的
    """
    new_sen_list, new_label_list = [], []
    sen_list, label_list = list(zip(*[[ii, jj] for ii, jj in zip(sen_list, label_list) if jj and ii]))
    for i, j in zip(sen_list, label_list):
        if new_sen_list and ((j[0] != 'O' or (len(j[0]) > 1 and j[0].split('-')[-1] != 'B')) or len(j) < 5):
            new_sen_list[-1].extend(i)
            new_label_list[-1].extend(j)
        else:
            new_sen_list.append(i)
            new_label_list.append(j)
    new_sen_list, new_label_list = list(zip(*[[ii, jj] for ii, jj in zip(new_sen_list, new_label_list)
                                             if len(set([label.strip()[-1] for label in jj])) > 1]))
    return new_sen_list, new_label_list


def change(file_name, file_out, dict_file, split_=' '):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = [i.split(split_) for i in f.readlines() if len(i.split(split_)) == 2]
        document_pair = [[i[0], i[1].strip()] for i in data]
    # document_pair = [i for i in document_pair if i[0].strip()]
    label_2_id = load_json(dict_file)
    # print(label_2_id)
    # save_json('label_2_id.json', label_2_id)
    sen_list = []
    label_list = []
    cur_sen_list = []
    cur_label_list = []
    if False:  # 合并字母和数字
        pair_2 = []
        for i, j in document_pair:
            if i.encode('utf-8').isalnum() and pair_2 and pair_2[-1][0].encode('utf-8').isalnum():
                pair_2[-1][0] += i
            else:
                pair_2.append([i, j])

        document_pair = pair_2

    for char, label in document_pair:
        cur_sen_list.append(char)
        cur_label_list.append(label)
        if char in (',', '，', '。'):
            # if sum(i != 'O' for i in cur_label_list):  # 不全为O，加入
            sen_list.append(cur_sen_list[:-1].copy())
            label_list.append(cur_label_list[:-1].copy())
            cur_label_list = []
            cur_sen_list = []

    sen_list, label_list = combine(sen_list, label_list)
    # for i in label_list:
    #     print(i)
    #     break
    # print(sen_list)
    raw_sen_list = [''.join(i) for i in sen_list]
    sen_list = [i for i in sen_list]
    ret_label_list = [i for i in label_list]
    label2_list = [[label_2_id[ii] for ii in i] for i in label_list]
    #  label前面后面添加O  因为前面后面是cls  和 sep
    data_dict = [{'sen': i, 'label_decode': j, 'label': [label_2_id['O'], ] + k + [label_2_id['O'], ], 'raw_sen': q}
                 for i, j, k, q in zip(sen_list, ret_label_list, label2_list, raw_sen_list) if i]


    df = pd.DataFrame(data_dict)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # df['bert_tokenize'] = df['raw_sen'].apply(tokenizer.tokenize)

    # df['len'] = df['raw_sen'].apply(lambda x: sum(tokenizer(x)['attention_mask']) - 2)  # 筛掉一些和bert不符合的
    # df['new'] = df.apply(lambda x: x.len == len(x.label), axis=1)
    # print(f"去掉的个数为{(~df['new']).sum()}")
    # df = df.loc[df['new'], :]
    # df.drop(['new', 'len'], inplace=True, axis=1)
    print(df)

    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df.to_csv(file_out, index=False, encoding='utf-8')
    test_df.to_csv('blood_data/dev.csv', index=False, encoding='utf-8')


# def get_test_csv():

if __name__ == '__main__':
    # json_dict = 'med_data/label_2_id.json'
    # get_dict('med_data/train.char.bmes', json_dict)
    # change('med_data/train.char.bmes', 'med_data/train.csv', json_dict)
    # change('med_data/test.char.bmes', 'med_data/test.csv', json_dict)
    # change('med_data/dev.char.bmes', 'med_data/dev.csv', json_dict)


    # dict_file = 'blood_data/label_2_id.json'
    json_dict = 'blood_data/label_2_id.json'
    from config import json_dict
    if not os.path.exists(json_dict):
        print('创建字典')
        get_dict('blood_data/result.txt', json_dict, split_='\t')
    change('blood_data/result.txt', 'blood_data/train.csv', json_dict, split_='\t')

    # change('blood_data/test.char.bmes', 'blood_data/test.csv')
    # change('blood_data/dev.char.bmes', 'blood_data/dev.csv')
