"""
Convert the model to the ONNX format
-----------------------------------

Author: Muahmmad Faizan
----------------------
python convert2onnx.py -h
"""
import torch
import hydra
import logging

from omegaconf.omegaconf import OmegaConf
from model import colaModel
from dataset import Dataset

# configure logger
logger = logging.getLogger(__name__)

# convert to onnx foramt
@hydra.main(config_path="configs/", config_name='configs.yaml', version_base= None)
def convert(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"loading pretrained model from {model_path}")
    model = colaModel.load_from_checkpoint(model_path)

    data = Dataset(model=cfg.model.tokenizer, batch_size= cfg.preprocess.batch_size,
                   max_length= cfg.prprocess.max_length)
    
    data.setup()
    input_batch = next(iter(data.train_dataloader()))
    input_example = {"input_ids": input_batch["input_ids"][0].unsqueeze(0),
                     "attention_mask": input_batch["attention_mask"][0].unsqueeze(0)}
    
    
    


if __name__ == '__main__':
    convert()