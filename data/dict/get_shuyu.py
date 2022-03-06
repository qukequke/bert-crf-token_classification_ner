# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/12/15 11:25
@Author  : quke
@File    : get_shuyu.py
@Description:
---------------------------------------
'''
import re

# import sys
# import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
file_name = '../blood_data/93 输血医学术语.txt'

with open(file_name, 'r', encoding='gbk') as f:
    d = f.readlines()

# print(d)
# d = re.findall('\d\.\d\.\d\.\n+?(.*)\n', d)
# d = re.findall('\d+\.\d+\.\d+\.', d)
# print(d)
start, end = '英文对应词索引', '汉语拼音索引'
ret_list = []
start_flag = 0
for i in d:
    if i.strip() == start:
        start_flag = 1
    if i.strip() == end:
        # print('23423423')
        break
    if start_flag:
        if len(i.strip()) >= 1:
            ret_list.append(i.strip())
ret_list = ret_list[1:]
ret_list = [i.split('\t')[0] for i in ret_list if len(i.split('\t')) == 2]
# print(ret_list)
abbreviation_list = [i.split('；')[1] for i in ret_list if '；' in i]
ret_list = [i if '；' not in i else i.split('；')[0] for i in ret_list]
ret_list = [i for i in ret_list if all([ord(ii) in range(ord('A'), ord('z')) or ii == ' ' for ii in i])]
print(ret_list)
print(abbreviation_list)
en_name = '英文名.txt'
abbreviation_name = '缩写词.txt'
with open(en_name, 'w', encoding='utf-8') as f:
    f.writelines([i + '\n' for i in ret_list])
with open(abbreviation_name, 'w', encoding='utf-8') as f:
    f.writelines([i + '\n' for i in abbreviation_list])


