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
import requests


def belong(x1, x2):
    start1, end1 = int(x1[1]), int(x1[2])
    start2, end2 = int(x2[1]), int(x2[2])
    if start1 == start2 and end1 == end2:
        return False

    if start1 >= start2 and end1 <= end2:
        return True
    else:
        return False


def drop_union_ann(region_wds):
    final_list = []
    region_wds.sort(key=lambda x: (int(x[1]), int(x[2])))
    for i in range(1, len(region_wds) - 1):
        if belong(region_wds[i], region_wds[i + 1]):
            print(f"2:{region_wds[i]} {region_wds[i + 1]}去掉{region_wds[i]}")
            continue
        elif belong(region_wds[i], region_wds[i - 1]):
            print(f"{region_wds[i - 1]} {region_wds[i]}去掉{region_wds[i]}")
            continue
        else:
            # print(region_wds[i])
            final_list.append(region_wds[i])
    return final_list


def get_vocab_ner(input_data):
    vocabulary_glob = 'dict/*.txt'
    # input_data = 'data/产科病历样例.txt'
    # print(input_data)
    re_pair = [
        ['(\d{4} ?- ?\d{2} ?- ?\d{2}) ?实施', '实施日期'],
        ['(\d{4} ?- ?\d{2} ?- ?\d{2}) ?发布', '发布日期'],
        ['(\w{2}/?T? ?\d{1,3}.?\d?—\d{4})', '编号'],
        ["[^\d/a-z·](([A-Z]?[a-z]{2,}[ -]{,3})+)[^·]", '英文名'],
        ["[\(（]([A-Z]{1,8})[\)）]", '缩写词'],
        # ['(\w{2}/?T? ?\d{1,3}.?\d?—\d{4})', '编号'],
        # ['(\w{2}\d{1,5}—\d{4})', '编号'],
    ]
    # ann_out = 'data/chanke.ann'
    # json_out = 'data/chanke.json'

    import glob
    import json
    import os
    import ahocorasick
    import re


    class ACTree:
        def __init__(self, file_list):
            print(file_list)
            self.wd_2_label = {}
            for file in file_list:
                print(file)
                data_list = open(file, 'r', encoding='utf-8').read().split('\n')
                data_list.sort(key=lambda x: len(x))
                open(file, 'w', encoding='utf-8').writelines([i + '\n' for i in data_list])
                for i in data_list:
                    self.wd_2_label[i] = os.path.basename(file).rsplit('.', 1)[0]
                # print(self.wd_2_label)
            self.build_actree(list(self.wd_2_label.keys()))


        def build_actree(self, wordlist):
            self.actree = ahocorasick.Automaton()
            for index, word in enumerate(wordlist):
                self.actree.add_word(word, (index, word))
            self.actree.make_automaton()

        def ner_dict(self, question):
            region_wds = []
            for i in self.actree.iter(question):
                wd = i[1][1]
                end = i[0]
                region_wds.append([wd, end - len(wd) + 1, end])
            final_list = drop_union_ann(region_wds)
            # stop_wds = []
            # final_list = []
            # region_wds.sort(key=lambda x: (x[1], x[2]))
            # for i in range(1, len(region_wds) - 1):
            #     if self.belong(region_wds[i], region_wds[i + 1]):
            #         # print(f"2:{region_wds[i]} {region_wds[i + 1]}去掉{region_wds[i]}")
            #         continue
            #     elif self.belong(region_wds[i], region_wds[i - 1]):
            #         # print(f"{region_wds[i - 1]} {region_wds[i]}去掉{region_wds[i]}")
            #         continue
            #     else:
            #         # print(region_wds[i])
            #         final_list.append(region_wds[i])
            final_dict = [{"name": i[0], "start": i[1], "end": i[2] + 1, "label": self.wd_2_label[i[0]]} for
                          i in final_list]
            return final_dict

    def re_data(data):
        # data = data.replace('\n', ' ').replace('\r', ' ')
        final_d = []
        for re_, label in re_pair:
            d = re.finditer(re_, data)
            # for ii in d:
            #     print(ii)
            d = [{'name': i.group(1), 'start': i.regs[1][0], 'end': i.regs[1][1], 'label': label} for i in d]
            print(label)
            print(d)
            final_d.extend(d)
        return final_d

    ac_tree = ACTree(glob.glob(vocabulary_glob))
    # data = open(input_data).read()
    # data = '既往史:2018年在本院孕足月剖宫产取一女活婴，现体健。否认肝炎、结核等急慢性传染病史，否认高血压、肾炎、糖尿病、心脏病史，否认食物及药物过敏史，否认输血史。按时预防接种'
    # print(data)
    d = ac_tree.ner_dict(input_data)
    if re_pair:
        re_d = re_data(input_data)
        d.extend(re_d)

    # for k in d:
    #     print(k['name'])
    #     print(k['start'])
    # file_name = 'data/chanke.ann'

    data = [[i['label'], i['start'], i['end'], i['name']] for i in d]
    return data

def get_data_api(document):
    json_data = {
        'document': document
    }
    r = requests.post('http://172.0.34.62:5000/api/entity_recognition', json=json_data).json()
    # print(r)
    data = [[i['classify'][0], i['start'], i['end'], i['name']] for i in r['data']]
    return data

def drop_dup(ann_list):
    ann_list = list(set(['###'.join([str(jj) for jj in i]) for i in ann_list]))
    ann_list = [i.split('###') for i in ann_list]
    return ann_list


df = pd.read_csv('blood_data/test_data_predict.csv', encoding='utf-8', usecols=['ann_data'])
ann_list = df['ann_data'].tolist()
# print(ann_list)
ann_list = [eval(i) for i in ann_list]
ann_list = sum(ann_list, start=[])
bert_ann_len = len(ann_list)

document = open('blood_data/brat/032.txt', 'r', encoding='utf-8').read()
data = get_data_api(document)
api_ann_len = len(data)

# print(len(ann_list))
# print(data)
data_vocab = get_vocab_ner(document)
vocab_ann_len = len(data_vocab)

ann_list.extend(data)
ann_list.extend(data_vocab)
pre_drop_len = len(ann_list)
ann_list = drop_dup(ann_list)
ann_list = drop_union_ann(ann_list)
print(f"drop前{pre_drop_len}， 最终长度{len(ann_list)}, bert_len{bert_ann_len}， api_len{api_ann_len}， vocab_len{vocab_ann_len}")

print(ann_list)
print(len(ann_list))
ann_list.sort(key=lambda x: int(x[1]))
data_list = []
for i, ann_one in enumerate(ann_list):
    if ann_one[3].strip():
        data_list.append(f"T{i}\t{ann_one[0]} {ann_one[1]} {ann_one[2]}\t{ann_one[3]}\n")

# for i in ann_list:
#     print(i)
#     print(document[int(i[1]):int(i[2])]==i[3])
#     print(i[1:3])
#     print(document[int(i[1]):int(i[2])])
#     print(i[3])
#     print()
with open('blood_data/brat/032.ann', 'w', encoding='utf-8') as f:
    f.writelines(data_list)
