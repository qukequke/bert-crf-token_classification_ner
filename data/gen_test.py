# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/12/6 8:50
@Author  : quke
@File    : gen_test.py
@Description:
---------------------------------------
'''
import re

import pandas as pd
import sys
import io
from copy import copy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')
file_name = 'blood_data/brat/032.txt'


# def drop_space(s):
    # s = re.sub('([a-zA-Z])\s+([a-zA-Z])', r'\1####\2', s)
    # s = re.sub('([a-zA-Z])\s+([a-zA-Z])', r'\1####\2', s)
    # s = s.replace(' ', '').replace('####', ' ')
    # return s
with open(file_name, 'r', encoding='utf-8') as f:
    d = f.read()
# 去掉前面不是英文的空格
d = ''.join([d[i] for i in range(1, len(d)) if not ((d[i] == ' ') and (ord(d[i - 1].lower()) not in range(ord('a'), ord('z'))))])
with open(file_name, 'w', encoding='utf-8') as f:
    f.write(d)
d = d.replace('\r', ' ').replace('\n', ' ').replace('\u3000', ' ')
# raw_txt_list = [[i.replace('\n', '#').replace('\r', '#').replace('\u3000', '#'), 'O'] for i in raw_txt]  # \n需要单独替换
# print(d)
# split_list = [',', '。', '，', '；']
split_list = (',', '，', '。', ';', '；')
sen_list = []
start_list = [0, ]

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

def drop_space(d):
    d = ''.join([d[i] for i in range(1, len(d)) if not ((d[i] == ' ') and (ord(d[i - 1].lower()) not in range(ord('a'), ord('z'))))])
    return d.strip()

# 去掉前面的0，且start变化
df['start'] = df.apply(lambda x: x.start + x.raw_sen.index(x.raw_sen.strip()[0]), axis=1)
df['raw_sen'] = df['raw_sen'].apply(lambda x: x.strip())
# df = df.drop_duplicates(subset=['raw_sen'])
# df['raw_sen'] = df['raw_sen'].apply(drop_space)

df['length'] = df['raw_sen'].apply(lambda x: len(x))
df.to_csv('blood_data/test.csv', encoding='utf-8', index=False)
