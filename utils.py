"""utils file"""
from typing import List, Tuple

import numpy as np
import pandas as pd


def create_dataset(data_path: str) -> Tuple[List[str], List[int]]:
    dataset = pd.read_csv(data_path)
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # shuffle the dataset
    return list(dataset["sequence"]), list(dataset["label"])


def word_embedding(
    sequence: str,
    max_seq_length: int = 200,
    CONSIDERED_AA: str = "ACDEFGHIKLMNPQRSTVWYX",
):
    # amino acids encoding
    aa_mapping = {aa: i + 1 for i, aa in enumerate(CONSIDERED_AA)}
    for i, val in enumerate(CONSIDERED_AA):
        if val == "X": 
            aa_mapping[val]=0
        else:
            aa_mapping

    # adapt sequence size
    if len(sequence) < max_seq_length:
        # extent the sequence
        sequence = sequence.zfill(max_seq_length)
    else:
        # pad the sequence
        sequence = "." + sequence * (max_seq_length - len(sequence))

    # encode sequence
    encoded_sequence = np.zeros((max_seq_length,))  # (200,)
    for i, amino_acid in enumerate(sequence):
        if amino_acid in CONSIDERED_AA:
            encoded_sequence[i] = aa_mapping[amino_acid]
    model_input = np.expand_dims(encoded_sequence, 0)  # add batch dimension

    return model_input  # (1, 200)