# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/12/6 8:50
@Author  : quke
@File    : gen_test.py
@Description:
---------------------------------------
'''
import pandas as pd

file_name = 'blood_data/031.txt'
with open(file_name, 'r', encoding='utf-8') as f:
    d = f.read()
d = d.replace('\r', ' ').replace('\n', ' ').replace('\u3000', ' ')
# raw_txt_list = [[i.replace('\n', '#').replace('\r', '#').replace('\u3000', '#'), 'O'] for i in raw_txt]  # \n需要单独替换
# print(d)
split_list = [',', '。', '，', '；']
sen_list = []
start_list = [0, ]
from copy import copy

temp_str = ''
for ii, char in enumerate(d):
    if char not in split_list:
        temp_str += char
    else:
        sen_list.append(copy(temp_str))
        temp_str = ''
        start_list.append(ii + 1)

# print(len(start_list), len(sen))
assert len(start_list) == len(sen_list) + 1
for i, j in zip(sen_list, start_list):
    print(i, d[j:j + len(i)])
# print(split_list)

data_dict = [{'raw_sen': i, 'start': j} for i, j in zip(sen_list, start_list)]
df = pd.DataFrame(data_dict)

# 去掉前面的0，且start变化
df['start'] = df.apply(lambda x:x.start + x.raw_sen.index(x.raw_sen.strip()[0]), axis=1)
df['raw_sen'] = df['raw_sen'].apply(lambda x:x.strip())
df.to_csv('blood_data/test.csv', encoding='utf-8', index=False)
