"""This file contains function to manipulate the batches created with functions in batch_generators.py
"""
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def pad_output_batch(output_batch):
    """This adds zero padding to the output batch in the beginning of the sequence
    to comply with the RNN input format."""
    return np.concatenate([np.zeros((output_batch.shape[0], 1)), output_batch], axis=1)


def one_hot_encode_output(output_batch):
    lb = LabelBinarizer()
    lb.fit(list(range(19)))  # la somma 1 non può esserci
    new_output_batch = lb.transform(output_batch[0])
    for i in range(1, output_batch.shape[0]):
        new_output_batch = np.stack([new_output_batch, lb.transform(output_batch[i])])
    return new_output_batch
