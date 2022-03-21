# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2022/3/5 16:21
@Author  : quke
@File    : txt2csv2.py
@Description:
将txt类型(每行都是一个字一个标注结果类型)转换为csv类型
'''
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import save_json, load_json


def get_dict(file_name, dict_out, split_=' '):
    """
    生成label2id字典
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        data = [i.split(split_) for i in f.readlines() if len(i.split(split_)) == 2]
        document_pair = [[i[0], i[1].strip()] for i in data]
    label_list = [i[1].replace('M-', 'I-').replace('E-', 'I-').replace("S-", 'I-') for i in document_pair]
    all_label_set = list(set(label_list))
    all_label_set.sort(key=lambda x: label_list.index(x))
    label_2_id = {'start': 0, 'end': 1, 'pad': 2}  # start,end是crf需要的
    label_2_id.update({j: i for i, j in enumerate(all_label_set, start=3)})
    print(f'生成{dict_out}')
    save_json(dict_out, label_2_id)

def change(file_name, file_out, dict_file, split_=' ', split_3=False):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = [i.split(split_) if len(i.split(split_)) == 2 else ['###', '###'] for i in f.readlines() ]
        document_pair = [[i[0], i[1].strip().replace('M-', 'I-').replace('E-', 'I-').replace("S-", 'I-')] for i in data]
    label_2_id = load_json(dict_file)
    sen_list = []
    label_list = []
    cur_sen_list = []
    cur_label_list = []
    for char, label in document_pair:
        cur_sen_list.append(char)
        cur_label_list.append(label)
        # if char in (',', '，', '。', ';', '；', ':', '：'):
        # print(char)
        if char in ('###', ):
            sen_list.append(cur_sen_list[:-1].copy())
            label_list.append(cur_label_list[:-1].copy())
            cur_label_list = []
            cur_sen_list = []
    # print(sen_list)
    # print(label_list)

    raw_sen_list = [''.join(i) for i in sen_list]
    label2_list = [[label_2_id[ii] for ii in i] for i in label_list]  # label2id
    # #  label前面后面添加O  因为前面后面是cls  和 sep
    data_dict = [{'sen': i, 'label_decode': j, 'label': [label_2_id['O'], ] + k + [label_2_id['O'], ], 'raw_sen': q}
                 for i, j, k, q in zip(sen_list, label_list, label2_list, raw_sen_list) if i]
    df = pd.DataFrame(data_dict)
    df['length'] = df['sen'].apply(lambda x: len(x))
    if split_3:
        train_df, test_df = train_test_split(df, test_size=0.2)
        test_df, dev_df = train_test_split(test_df, test_size=0.1)
        train_df.to_csv(file_out, index=False, encoding='utf-8')
        test_df.to_csv('test.csv', index=False, encoding='utf-8')
        dev_df.to_csv('dev.csv', index=False, encoding='utf-8')
    else:
        df.to_csv(file_out, index=False, encoding='utf-8')

if __name__ == '__main__':
    json_dict = 'cner/label_2_id.json'
    get_dict('cner/train.char.bmes', json_dict)  # 生成label2id字典，存储到json_dict
    change('cner/train.char.bmes', 'cner/train.csv', json_dict)
    change('cner/test.char.bmes', 'cner/test.csv', json_dict)
    change('cner/dev.char.bmes', 'cner/dev.csv', json_dict)