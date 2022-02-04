"""This file contains functions that create batches of input and output data. The inputs are sequences of integers,
and the outputs are other sequences that are pairwise sums of the inputs."""
import numpy as np


def generate_input_batch(int_range=(1, 10), sample_size=2, batch_size=100, seed=123456):
    """The input batch is composed by sequences of sample_size integers randomly selected in int_range
    (right element excluded)."""
    rng = np.random.default_rng(seed=seed)
    return rng.integers(int_range[0], int_range[1], size=(batch_size, sample_size))


def generate_output_batch(input_batch):
    """Given an input sequence of N integers, the inputs are sums of the inputs two by two (e.g. the first two,
    the second two, etc).
    """
    output_batch = input_batch[:, 0:2].sum(axis=1)
    output_batch = np.expand_dims(output_batch, axis=1)
    for i in range(1, input_batch.shape[1]-1):
        sum_of_two_cols = input_batch[:, i:i+2].sum(axis=1)
        sum_of_two_cols = np.expand_dims(sum_of_two_cols, axis=1)
        output_batch = np.concatenate([output_batch, sum_of_two_cols], axis=1)
    return output_batch
