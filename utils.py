"""utils file"""
from typing import List, Tuple

import numpy as np
import pandas as pd
import math

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

    # encode sequence
    encoded_sequence = np.zeros((max_seq_length,))  # (200,)
    for i, amino_acid in enumerate(sequence):
        if amino_acid in CONSIDERED_AA:
            encoded_sequence[i] = aa_mapping[amino_acid]
    model_input = np.expand_dims(encoded_sequence, 0)  # add batch dimension

    return model_input  # (1, 200)


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    sensitivity = float(tp) / (tp + fn + 1e-06)
    specificity = float(tn) / (tn + fp + 1e-06)
    mcc = float(tp * tn - fp * fn) / (
        math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-06
    )
    return acc, sensitivity, specificity, mcc