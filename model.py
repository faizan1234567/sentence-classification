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


