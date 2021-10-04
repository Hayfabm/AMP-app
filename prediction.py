"""
WARNINGS: if you run the app locally and don't have a GPU you should choose device='cpu'
"""

from typing import List

from tensorflow.keras.models import load_model
import numpy as np

from utils import word_embedding


class AMPApp:
    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.num_gpus = 0 if device == "cpu" else 1

        self.NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
        self.max_seq_length = 200

        # NOTE: if you have issues at this step, please use h5py 2.10.0
        # by running the following command: pip install h5py==2.10.0
        self.model = load_model("logs/model_20211004154235.hdf5")
        print(self.model.summary())

    @staticmethod
    def score_names() -> List[str]:
        return ["AMP_recognition_probability"]

    def compute_scores(self, sequences_list: List[str]) -> List[float]:
        scores_list = []
        for sequence in sequences_list:
            # sequence embedding
            sequence_embeddings = word_embedding(sequence)

            # forward pass throught the model
            model_output = self.model.predict(sequence_embeddings)
            scores_list.append(
                [{self.score_names()[0]: prob[1]} for prob in model_output]
            )  # model_output[1]: # Antimicrobial recognition 
        return scores_list


if __name__ == "__main__":
    sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQ", "KALEE", "LAGYNIVATPRGYVLAGG"]

    app = AMPApp("cpu")

    scores = app.compute_scores(sequences)
    print (scores)

