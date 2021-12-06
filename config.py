# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/11/18 15:24
@Author  : quke
@File    : config.py
@Description:
---------------------------------------
'''
model_dict = {
    'bert': ('transformers.BertTokenizer', 'transformers.BertModel', 'transformers.BertConfig'),
    'roberta': (
        'transformers.BertTokenizer', 'transformers.RobertaForSequenceClassification', 'transformers.RobertaConfig'),
    'bart': ('transformers.BertTokenizer', 'transformers.BartForConditionalGeneration', 'transformers.BartConfig'),
}
epochs = 100
batch_size = 32
# batch_size = 10
lr = 1e-5  # 学习率
patience = 40  # early stop 不变好 就停止
max_grad_norm = 10.0  # 梯度修剪
# target_file = 'models/best.pth.tar'  # 模型存储路径
target_file = 'models/best_med.pth.tar'  # 模型存储路径
# checkpoint = 'models/best.pth.tar'   # 设置模型路径  会继续训练
checkpoint = None  # 设置模型路径  会继续训练
# max_seq_len = 103
n_nums = None  # 读取csv行数，因为有时候测试需要先少读点 None表示读取所有
freeze_bert_head = False  # freeze bert提取特征部分的权重


# 切换任务时 数据配置
num_labels = 28  # 文本分类个数
# num_labels = 139  # 文本分类个数
# num_labels = 2  # 文本分类个数
# csv_rows = ['premise', 'hypothesis', 'label']
csv_rows = ['raw_sen', 'label']  # csv的行标题，文本 和 类（目前类必须是数字）
# csv_rows = ['sentence1', 'sentence2', 'label']  # csv的行标题，文本 和 类（目前类必须是数字）

# train_file = "data/med_data/train.csv"
# dev_file = "data/med_data/dev.csv"
# test_file = "data/med_data/train.csv"
# test_file = "data/dev.csv"
# dir_name = 'blood_data'
dir_name = 'med_data'
train_file = f"data/{dir_name}/train.csv"
dev_file = f"data/{dir_name}/dev.csv"
# dev_file = f"data/{dir_name}/train.csv"
test_file = f"data/{dir_name}/test.csv"
# test_file = f"data/{dir_name}/dev.csv"
csv_encoding = 'utf-8'
json_dict = f'data/{dir_name}/label_2_id.json'
# test_pred_out = f"data/{dir_name}/train_data_predict.csv"
test_pred_out = f"data/{dir_name}/test_data_predict.csv"
# csv_encoding = 'gbk'

# 训练还是测试任务
MODE = 'train'
# MODE = 'test'
# MODEL = 'roberta'
MODEL = 'bert'
# bert_path_or_name = 'fnlp/bart-base-chinese'  # 使用模型
bert_path_or_name = 'bert-base-chinese'  # 使用模型
# bert_path_or_name = 'hfl/chinese-roberta-wwm-ext'  # 使用模型

PREFIX = ''
max_src_length = 400
max_seq_len = 100
# max_src_length = 128
max_target_length = 64
ignore_pad_token_for_loss = True
overwrite_cache = None


use_crf = True