# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/11/24 11:03
@Author  : quke
@File    : tt.py
@Description:
---------------------------------------
'''
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer, \
    BartForConditionalGeneration

data_files = {}
data_files["train"] = 'data/summary_train.csv'
extension = 'csv'
raw_datasets = load_dataset(extension, data_files=data_files)
# train_dataset = raw_datasets['train']

text_column = 'content'
summary_column = 'title'
prefix = ''
max_target_length = 128
padding = 'max_length'
ignore_pad_token_for_loss = True
def preprocess_function(examples):
    inputs = examples[text_column]
    print(len(inputs))
    targets = examples[summary_column]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)
    print(model_inputs[0])

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print(raw_datasets['train'])
# for i,j in raw_datasets.items():

column_names = raw_datasets["train"].column_names
# BartForConditionalGeneration
# tokenizer = AutoTokenizer.from_pretrained('fnlp/bart-base-chinese')
tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
# model = AutoModelForSeq2SeqLM.from_pretrained('fnlp/bart-base-chinesek')
model = BartForConditionalGeneration.from_pretrained('fnlp/bart-base-chinese')
print(tokenizer)
print(model)
overwrite_cache = None
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    # pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    pad_to_multiple_of=None
)

processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=column_names,
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=10
)
for batch_dict in train_dataloader:
    for k, v in batch_dict.items():
        print(k)
        print(v.shape)
    break
