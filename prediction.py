"""
WARNINGS: if you run the app locally and don't have a GPU you should choose device='cpu'
"""

from typing import List, Dict
from deepchain.components import DeepChainApp
from tensorflow.keras.models import load_model
import numpy as np

Score = Dict[str, float]
ScoreList = List[Score]

# utils
def word_embedding(
    sequence: str,
    max_seq_length: int = 200,
    CONSIDERED_AA: str = "ACDEFGHIKLMNPQRSTVWYX",
):
    # amino acids encoding
    aa_mapping = {aa: i + 1 for i, aa in enumerate(CONSIDERED_AA)}
    for i, val in enumerate(CONSIDERED_AA):
        if val == "X":
            aa_mapping[val] = 0
        else:
            aa_mapping

    # adapt sequence size
    if len(sequence) < max_seq_length:
        # extent the sequence
        sequence = sequence.zfill(max_seq_length)

    # encode sequence
    encoded_sequence = np.zeros((max_seq_length,))  # (200,)
    for i, amino_acid in enumerate(sequence):
        if amino_acid in CONSIDERED_AA:
            encoded_sequence[i] = aa_mapping[amino_acid]
    model_input = np.expand_dims(encoded_sequence, 0)  # add batch dimension

    return model_input  # (1, 200)


# app


class App(DeepChainApp):
    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.num_gpus = 0 if device == "cpu" else 1
        self.NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
        self.max_seq_length = 200

        # NOTE: if you have issues at this step, please use h5py 2.10.0
        # by running the following command: pip install h5py==2.10.0

        self.model = load_model("logs/model_20211005100737.hdf5")
        print(self.model.summary())

    @staticmethod
    def score_names() -> List[str]:
        return ["AMP_recognition_probability"]

    def compute_scores(self, sequences_list: List[str]) -> ScoreList:
        scores_list = []
        for sequence in sequences_list:
            # sequence embedding
            sequence_embeddings = word_embedding(sequence)

            # forward pass throught the model
            model_output = self.model.predict(sequence_embeddings)
            scores_list.append(
                {self.score_names()[0]: model_output[0][1]}
            )  # model_output[1]: # Antimicrobial recognition
        return scores_list


if __name__ == "__main__":
    sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQ", "KALEE", "LAGYNIVATPRGYVLAGG"]
    app = App("cpu")

    scores = app.compute_scores(sequences)
    print(scores)

