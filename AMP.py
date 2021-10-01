"""AMP training script for custom model"""
import datetime
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    Bidirectional,
    Embedding,
    Dense,
    Dropout,
    Flatten,
    Convolution1D,
    MaxPooling1D,
    BatchNormalization,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)



from utils import create_dataset, word_embedding, categorical_probas_to_classes, calculate_performace


def build_model(top_words, maxlen, pool_length, embedding_size):
    """AMP model
    Combined CNN and LSTM, to predict Antimicrobial peptide
    """
    custom_model = Sequential(name="AMP-model")
    # (4042, 200) <- 'preprocess' word embedding sequence encoding
    custom_model.add(Embedding(top_words, embedding_size, input_length=maxlen))
    # (4042, 200, 128) <- word embedding sequence encoding
    custom_model.add(
        Convolution1D(
            64,
            16,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer="random_uniform",
            name="convolution_1d_layer1",
        )
    )
    custom_model.add(MaxPooling1D(pool_size=pool_length))
    custom_model.add(LSTM(100, return_sequences=False, name='lstm1'))
    custom_model.add(Dropout(0.1))
    custom_model.add(Dense(2, name='full_connect'))
    custom_model.add(Flatten())
    custom_model.add(Activation('sigmoid'))
    
    return custom_model


if __name__ == "__main__":
    path = "AMP/"
    if not os.path.exists(path):
        os.makedirs(path)

    Rec_A = open(path + "Antimicrobial_recognition.txt" , "w")

    Rec_A.writelines(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        + "\n"
    )
    Rec_A.write(
        "   acc,            sensitivity,         specificity,          mcc           "
        + "\n"
    )
    Rec_A.writelines(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        + "\n"
    )



    # init neptune logger
    run = neptune.init(project='sophiedalentour/AMP-app',
                    tags=['epochs = 10', 'range(k_fold)']
                )

    # set the seed
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # set amino acids to consider
    CONSIDERED_AA = "ACDEFGHIKLMNPQRSTVWYX"

    # embedding and convolution parameters
    MAX_SEQ_LENGTH = 200
    VOCAB_SIZE = len(CONSIDERED_AA)
    POOL_LENGTH = 5
    EMBEDDING_SIZE = 128

    # split_dataset parameters
    k_fold = 10
    scores = []
    
    # training parameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    SAVED_MODEL_PATH = (
        "logs/model_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".hdf5"
    )
    DATASET= "Data/all_DATA.csv"
    
    # save parameters in neptune
    run["hyper-parameters"] = {
        "encoding_mode": "word embedding",
        "seed": SEED,
        "considered_aa": CONSIDERED_AA,
        "max_seq_length": MAX_SEQ_LENGTH,
        "vocab_size": VOCAB_SIZE,
        "pool_length": POOL_LENGTH,
        "embedding_size": EMBEDDING_SIZE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "saved_model_path": SAVED_MODEL_PATH,
        "dataset": DATASET,
    }

    # create dataset
    sequences, labels= np.array(create_dataset(data_path=DATASET))
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=1024)
    for ((train, test), k) in zip(skf.split(sequences, labels), range(k_fold)):

        # encode sequences
        sequences_train_encoded = np.concatenate(
            [
                word_embedding(seq, MAX_SEQ_LENGTH, CONSIDERED_AA)
                for seq in sequences[train]
            ],
            axis=0,
            )  
        sequences_test_encoded = np.concatenate(
            [
                word_embedding(seq, MAX_SEQ_LENGTH, CONSIDERED_AA)
                for seq in sequences[test]
            ],
            axis=0,
            )  

        # encode labels
        labels_train_encoded = to_categorical(
                labels[train], num_classes=2, dtype="float32"
            )  
        labels_test_encoded = to_categorical(
                labels[test], num_classes=2, dtype="float32",
            )  

        # build model
        model = build_model(VOCAB_SIZE, MAX_SEQ_LENGTH, POOL_LENGTH, EMBEDDING_SIZE)
        print(model.summary())

        # compile model
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy", "AUC", "Precision", "Recall"]
        )

        # define callbacks
        my_callbacks = [
            # ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
            # EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1),
            ModelCheckpoint(
                monitor="val_accuracy",
                mode="max",
                filepath=SAVED_MODEL_PATH,
                save_best_only=True,
            ),
            NeptuneCallback(run=run, base_namespace="metrics"),
        ]

        # fit the model
        history = model.fit(
            sequences_train_encoded,
            labels_train_encoded,
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            verbose=1,
            validation_split=0.1,
            #validation_data=(sequences_test_encoded, labels_test_encoded),
            callbacks=my_callbacks,
        )       

    #run.stop()

        # prediction probability

        predictions = model.predict(sequences_test_encoded)
        predictions_prob = predictions[:, 1]

        y_class = categorical_probas_to_classes(predictions)

        # true_y_C_C=utils.categorical_probas_to_classes(true_y_C)
        true_y = categorical_probas_to_classes(labels_test_encoded)
        (  
            acc, 
            sensitivity, 
            specificity, 
            mcc,
        ) = calculate_performace(len(y_class), y_class, true_y)
        print("======================")
        print("======================")
        print(
            "Iter " + ", " + str(k + 1) + " of " + str(k_fold) + "cv:"
        )
        print(
            "\tacc='%0.4f', sn='%0.4f', sp='%0.4f', mcc='%0.4f'"
            % (acc, sensitivity, specificity, mcc)
        )
        Rec_A.write(
            str(acc)
            + ","
            + str(sensitivity)
            + ","
            + str(specificity)
            + ","
            + str(mcc)
            + "\n"
        )
        scores.append(
            [acc,  sensitivity, specificity, mcc]
        )
    scores = np.array(scores)

    print(len(scores))

    
    Rec_A.close()
    

    run.stop()