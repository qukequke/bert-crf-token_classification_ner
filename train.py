# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:55:43 2020

@author: zhaog
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import DataPrecessForSentence
from utils import train, validate, eval_object
from transformers import BertTokenizer, AutoTokenizer, DataCollatorForSeq2Seq
from model import BertModel
from transformers.optimization import AdamW
from config import *

Tokenizer = eval_object(model_dict[MODEL][0])


# print(Tokenizer)

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
    print(processed_datasets_train)
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

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
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
    _, valid_loss, valid_accuracy = validate(model, dev_loader)
    # print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(valid_loss,
    #                                                                               (valid_accuracy * 100),
    #                                                                               ))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training roberta model on device: {}".format(device), 20 * "=")
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
        # scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.
        if f1_score < best_score:
            patience_counter += 1
        else:
            print('save data')
            best_score = f1_score
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        # "best_score": best_score,
                        # "optimizer": optimizer.state_dict(),
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))
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
