'''
Author: Muhammad Faizan
-----------------------
python model.py
'''
import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from sklearn.metrics import accuracy_score

class colaModel(pl.lightningModule):
    def __init__(self, model = "google/bert_uncased_L-2_H-128_A-2", lr = 1e-2):
        super(colaModel, self).__init()
        self.lr = lr
        self.save_hyperparameters()

        self.model = AutoModel.from_pretrained(model)
        self.linear = nn.Linear(self.model.config.hidden_size, 2)
        self.num_classes = 2
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)

        h_cls = outputs.last_hidden_state[:, 0]
        logits = self.linear(h_cls)
        return logits
    
    def training_step(self, batch):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log("training_loss", loss, prog_bar = True)
        return loss
    
    def validation_step(self, batch):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        _, preds = torch.max(logits, dim=1)
        val_accuracy = accuracy_score(batch["label"].cpu(), preds.cpu())
        val_accuracy = torch.tensor(val_accuracy)
        self.log("validation_acc", val_accuracy, prog_bar = True)
        self.log("validation_loss", loss, prog_bar = True)

    def configure_optimizer(self):
        return torch.optim.adam(self.model.parameters(), lr = self.hparams["lr"])



