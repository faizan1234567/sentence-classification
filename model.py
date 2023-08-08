'''
Author: Muhammad Faizan
-----------------------
python model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class colaModel(pl.LightningModule):
    def __init__(self, model = "google/bert_uncased_L-2_H-128_A-2", lr = 3e-5):
        super(colaModel, self).__init__()
        self.lr = lr
        self.save_hyperparameters()

        self.num_classes = 2
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels= self.num_classes)
        # self.linear = nn.Linear(self.model.config.hidden_size, self.num_classes)
        

        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes = self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average = "macro", num_classes = self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average = "macro", num_classes = self.num_classes
        )

        self.precision_micro_metric = torchmetrics.Precision(average = "micro")
        self.recall_micro_metric = torchmetrics.Recall(average = "micro")
    
    def forward(self, input_ids, attention_mask, labels = None):
       outputs = self.model(input_ids = input_ids, attention_mask = attention_mask,
                       labels = labels)
       return outputs
    
    def training_step(self, batch, batch_index):

       outputs = self.forward(input_ids= batch["input_ids"], 
                              attention_mask= batch["attention_mask"],
                              labels= batch["label"])
       predictions = torch.argmax(outputs.logits, dim=1)
       train_acc = self.train_accuracy_metric(predictions, batch["label"])
       self.log("train/loss", outputs.loss, prog_bar = True, on_epoch = True)
       self.log("train/acc", train_acc, prog_bar = True, on_epoch = True)
       return outputs.loss 
    
    def validation_step(self, batch, batch_index):
        labels = batch["label"]
        outputs = self.forward(input_ids = batch["input_ids"],
                               attention_mask= batch["attention_mask"],
                               labels = labels)
        preds = torch.argmax(outputs.logits, 1)

        # calculate metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # log all these metrics
        self.log("valid/loss", outputs.loss, prog_bar = True, on_step = True)
        self.log("valid/acc", valid_acc, prog_bar = True, on_epoch = True)
        self.log("valid/precision_macro", precision_macro, prog_bar = True, on_epoch = True)
        self.log("valid/recall_macro", recall_macro, prog_bar = True, on_epoch = True)
        self.log("valid/precision_micro", precision_micro , prog_bar = True, on_epoch = True)
        self.log("valid/recall_micro",recall_micro , prog_bar = True, on_epoch = True)
        self.log("valid/f1",f1 , prog_bar = True, on_epoch = True)
        return {"labels": labels, "logits": outputs.logits}
    
    def validation_epoch_end(self, outputs):
        
        labels = torch.cat([x["label"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        
        # plot confusino matrix on w&b
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
            probs = logits.numpy(), y_true = labels.numpy()
                )
            }
        )
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.hparams["lr"])

