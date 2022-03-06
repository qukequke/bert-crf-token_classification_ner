# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/11/18 15:24
@Author  : quke
@File    : config.py
@Description:
---------------------------------------
'''
import json

model_dict = {
    'bert': ('transformers.BertTokenizer',
             'transformers.BertModel',
             'transformers.BertConfig',
             'bert-base-chinese'  # 使用模型
             ),
    'ernie': (
        'transformers.AutoTokenizer',
        'transformers.BertModel',
        'transformers.AutoConfig',
        "nghuyong/ernie-1.0",  # 使用模型参数
    ),
    'roberta': (
        'transformers.BertTokenizer',
        'transformers.RobertaModel',
        'transformers.RobertaConfig',
        'hfl/chinese-roberta-wwm-ext',
    ),
    'albert': ('transformers.AutoTokenizer',
               'transformers.AlbertModel',
               'transformers.AutoConfig',
               "voidful/albert_chinese_tiny",  # 使用模型参数
               ),
}
MODEL = 'roberta'
# MODEL = 'ernie'
# MODEL = 'albert'
# MODEL = 'bert'

epochs = 20
batch_size = 32
# batch_size = 1
# batch_size = 10
lr = 1e-5  # 学习率
patience = 40  # early stop 不变好 就停止
max_grad_norm = 10.0  # 梯度修剪
target_file = 'models/best.pth.tar'  # 模型存储路径
# target_file = 'models/best_med.pth.tar'  # 模型存储路径
# checkpoint = 'models/blood_best5.pth.tar'   # 设置模型路径  会继续训练
checkpoint = None  # 设置模型路径  会继续训练
n_nums = None  # 读取csv行数，因为有时候测试需要先少读点 None表示读取所有
# n_nums = 20 # 读取csv行数，因为有时候测试需要先少读点 None表示读取所有
freeze_bert_head = False  # freeze bert提取特征部分的权重


# 切换任务时 数据配置
csv_rows = ['raw_sen', 'label']  # csv的行标题，文本 和 类（目前类必须是列表）

dir_name = 'cner'
train_file = f"data/{dir_name}/train.csv"
dev_file = f"data/{dir_name}/dev.csv"
# dev_file = f"data/{dir_name}/train.csv"
test_file = f"data/{dir_name}/test.csv"
# test_file = f"data/{dir_name}/dev.csv"
# csv_encoding = 'utf-8'
csv_encoding = 'utf-8'
# csv_encoding = 'gbk'
json_dict = f'data/{dir_name}/label_2_id.json'
test_pred_out = f"data/{dir_name}/test_data_predict.csv"
# csv_encoding = 'gbk'


PREFIX = ''
# max_src_length = 400
max_seq_len = 150
# max_src_length = 128
# max_target_length = 64
ignore_pad_token_for_loss = True
overwrite_cache = None


use_crf = True

with open(json_dict, 'r', encoding='utf-8') as f:
    dict_ = json.load(f)
num_labels = len(dict_)
print(f"num_labels 是{num_labels}")