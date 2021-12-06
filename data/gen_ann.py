# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/12/6 9:40
@Author  : quke
@File    : gen_ann.py
@Description:
---------------------------------------
'''
import pandas as pd

df = pd.read_csv('blood_data/test_data_predict.csv', encoding='utf-8', usecols=['ann_data'])
ann_list = df['ann_data'].tolist()
# print(ann_list)
ann_list = [eval(i) for i in ann_list]
ann_list = sum(ann_list, start=[])
print(ann_list)

data_list = []
for i, ann_one in enumerate(ann_list):
    data_list.append(f"T{i}\t{ann_one[0]} {ann_one[1]} {ann_one[2]}\t{ann_one[3]}\n")


with open('blood_data/123.ann', 'w', encoding='utf-8') as f:
    f.writelines(data_list)

