from AMP_with_bio_transformers import BATCH_S
from biotransformers import BioTransformers
import numpy as np
from utils import create_dataset


DATASET= "Data/all_DATA.csv"

BACKEND_MODEL = "protbert"
POOL_MODE = "mean"
BATCH_S = 2

sequences, labels= np.array(create_dataset(data_path=DATASET))
bio_trans = BioTransformers(backend=BACKEND_MODEL)

sequences_train_embeddings = bio_trans.compute_embeddings(
sequences, pool_mode=(POOL_MODE,), batch_size=BATCH_S
)[
POOL_MODE
]  
print (sequences_train_embeddings)


