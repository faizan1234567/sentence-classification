import torch
import pytorch_lightning as pl

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer

class Dataset(pl.LightningDataModule):
    def __init__(self, model = 'google/bert_uncased_L-2_H-128_A-2', batch_size = 32,
                 max_length = 128):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    # prepare dataset
        cola_dataset = load_dataset('glue', 'cola')
        self.train_dataset = cola_dataset['train']
        self.validation_dataset = cola_dataset['validation']
        self.test_dataset = cola_dataset['test']

    def tokenize(self, sample):
        return self.tokenizer(
            sample['sentence'],
            truncation = True, 
            padding = 'max_length',
            max_length = self.max_length)
    
    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.train_dataset.map(self.tokenize, batched = True)
            self.train_dataset.set_format(type = "torch", 
                                        columns = ["input_ids", "attention_mask", "label"])
            
            self.validation_dataset = self.validation_dataset.map(self.tokenize, batched = True)
            self.validation_dataset.set_format(type = "torch", 
                                               columns = ["input_ids", "attention_mask", "label"])
            self.test_dataset = self.test_dataset.map(self.tokenize, batched = True)
            self.test_dataset.set_format(type = "torch", 
                                         columns = ["input_ids", "attention_mask", "label"])
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size,
                                           shuffle = True, num_workers = 8)
    def validation_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_dataset, batch_size = self.batch_size,
                                           shuffle = True, num_workers = 8)
    

if __name__ == "__main__":
    # load the dataset
    dataset = Dataset()
    dataset.prepare_dataset()
    dataset.setup()
    data = dataset.train_dataloader()
    batch = next(iter(data))
    print(batch['input_ids'].shape, batch['label'].shape, batch['attention_mask'].shape)
    print(batch['input_ids'][0])


        


    
