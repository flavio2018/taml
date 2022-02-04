"""This file contains function to manipulate the batches created with functions in batch_generators.py
"""
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def pad_output_batch(output_batch):
    """This adds zero padding to the output batch in the beginning of the sequence
    to comply with the RNN input format."""
    return np.concatenate([np.zeros((output_batch.shape[0], 1)), output_batch], axis=1)


def one_hot_encode_batch(batch, max_sum=None):
    """Transform a batch of outputs with categorical labels into a batch with one-hot encoded labels."""
    lb = LabelBinarizer()
    if max_sum is None:
        max_sum = max(batch.ravel())
    lb.fit(list(range(max_sum + 1)))  # la somma 1 non può esserci
    new_output_batch = lb.transform(batch[0])
    for i in range(1, batch.shape[0]):
        one_hot_encoded_labels = lb.transform(batch[i])
        # print(np.shape(new_output_batch))
        new_output_batch = np.concatenate([new_output_batch, one_hot_encoded_labels])
    return np.reshape(new_output_batch, (-1, batch.shape[1], max_sum + 1))
