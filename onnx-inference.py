"""
Run ONNX inference runtime
--------------------------

Author: Muhammad Faizan

python onnx-inference.py -h
"""

# import dependencies
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

from dataset import Dataset
from utils import timing
