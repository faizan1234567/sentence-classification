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


# ----------------------------------
class ONNXPredictor:
    def __init__(self, model_path) -> None:
        """
        ONNX predictor 
        -------------
        Parameters
        ----------
        model_path: str
        """
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = Dataset()
        self.labels = ["unacceptable", "acceptable"]

    
    @timing
    def predict(self, text):
        """
        predict the text as acceptable or unacceptable
        ----------------------------------------------

        Parameters
        ----------
        text: str

        """
        inference_example = {'sentence': text}
        processed = self.processor.tokenize(inference_example)
        ort_input = {'input_ids': np.expand_dims(processed['input_ids'], axis=0), 
                     'attention_mask': np.expand_dims(processed['attention_mask'], axis= 0)}
        # run the ort inference
        ort_outputs = self.ort_session.run(None, ort_input)
        scores = softmax(ort_outputs[0])[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == '__main__':
    # single sentence
    sentence = 'He is eating an Apple'
    predictor = ONNXPredictor('models/model.onnx')
    print(predictor.predict(sentence))

    # for a list of sentences
    sentences = ['Mission impossible is my favourite movie'] * 5
    for sentence in sentences:
        print(predictor.predict(sentence))
    

