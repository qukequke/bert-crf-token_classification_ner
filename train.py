# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import DataPrecessForSentence
from utils import train, validate, eval_object, my_plot
# from transformers import BertTokenizer, AutoTokenizer, DataCollatorForSeq2Seq
from model import BertModel
from transformers.optimization import AdamW
from config import *

Tokenizer = eval_object(model_dict[MODEL][0])
bert_path_or_name = model_dict[MODEL][-1]



def main():
    tokenizer = Tokenizer.from_pretrained(bert_path_or_name)
    device = torch.device("cuda")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    target_dir = os.path.dirname(target_file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    processed_datasets_train = DataPrecessForSentence(tokenizer, train_file)
    processed_datasets_dev = DataPrecessForSentence(tokenizer, dev_file)
    print("\t* Loading validation data...")
    # dev_data = DataPrecessForSentence(tokenizer, dev_file)
    # dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = BertModel().to(device)
    # print(processed_datasets_train)
    # train_dataset = processed_datasets_train["train"]
    # dev_dataset = processed_datasets_dev["train"]
    train_loader = DataLoader(processed_datasets_train, shuffle=True, batch_size=batch_size)
    dev_loader = DataLoader(processed_datasets_dev, shuffle=True, batch_size=batch_size)
    # train_loader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    # )
    # dev_loader = DataLoader(
    #     dev_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size

    # )

    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    bert_lr = 3e-5
    crf_lr = 1e-3
    linear_lr = 1e-3
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': bert_lr},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': bert_lr},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': crf_lr},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_lr},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': crf_lr},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': linear_lr}
        ]

    # optimizer = AdamW(model.parameters(), lr=lr)
    optimizer = AdamW(optimizer_grouped_parameters, lr=bert_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    train_f1_list, dev_f1_list = [], []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        print(f'载入checkpoint文件{checkpoint}')
        checkpoint_save = torch.load(checkpoint)
        # print(checkpoint_save.keys())
        start_epoch = checkpoint_save["epoch"] + 1
        # start_epoch = 1
        best_score = checkpoint_save["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint_save["model"])
        # optimizer.load_state_dict(checkpoint_save["optimizer"])
        epochs_count = checkpoint_save["epochs_count"]
        train_losses = checkpoint_save["train_losses"]
        valid_losses = checkpoint_save["valid_losses"]
    # Compute loss and accuracy before starting (or resuming) training.
    epoch_time, epoch_loss, f1_score = validate(model, dev_loader)
    print("before train-> Valid. time: {:.4f}s, loss: {:.4f}, f1_score: {:.4f}% \n"
          .format(epoch_time, epoch_loss, (f1_score * 100)))

    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        # print(model)
        # epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        epoch_time, epoch_loss = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: "
              .format(epoch_time, epoch_loss))
        print("* Validation for epoch {}:".format(epoch))
        # epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(model, dev_loader)

        epoch_time, epoch_loss, f1_score = validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, f1_score: {:.4f}% \n"
              .format(epoch_time, epoch_loss, (f1_score * 100)))
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(f1_score)
        # Early stopping on validation accuracy.
        if f1_score < best_score:
            patience_counter += 1
        else:
            print('save model')
            best_score = f1_score
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        # "optimizer": optimizer.state_dict(),
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       target_file
                       )
        dev_f1_list.append(f1_score)
        my_plot(dev_f1_list, train_losses)
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    # train_file = "../data/TextClassification/mnli/train.csv"
    # df = pd.read_csv(train_file, engine='python', error_bad_lines=False)

    main(
        # "../data/TextClassification/mnli/train.csv",
        # "../data/TextClassification/mnli/dev.csv",
        # "data/TextClassification/qqp/train.csv",
        # "../data/TextClassification/qqp/dev.csv",
        # "../data/TextClassification/imdb/train.csv",
        # "../data/TextClassification/imdb/dev.csv",
        # "models",
        # checkpoint='models/best.pth.tar'
    )
