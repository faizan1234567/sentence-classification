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

