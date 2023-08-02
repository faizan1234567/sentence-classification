import torch
from model import colaModel
from dataset import Dataset

class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.