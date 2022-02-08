"""This file contains functions that create batches of input and output data.
The specific inputs and outputs depend on the task."""
import numpy as np


def generate_sum_input_batch(int_range=(1, 10), sample_size=2, batch_size=100, seed=123456):
    """The input batch is composed by sequences of sample_size integers randomly selected in int_range
    (right element excluded)."""
    rng = np.random.default_rng(seed=seed)
    return rng.integers(int_range[0], int_range[1], size=(batch_size, sample_size))


def generate_sum_output_batch(input_batch):
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


def generate_discriminate_input_batch(range=(1, 10), only_int=True, only_these: list = None, batch_size=100, seed=123456):
    """Generate couples of numbers within the specified range, or sampled uniformly within a specified list of integers
     if only_these is not None. Also, the numbers can be either only integers if only_int is True, or also floats
     otherwise."""
    if only_these is not None:
        rng = np.random.default_rng(seed=seed)
        batch = rng.choice(a=only_these, size=(batch_size, 2))
    else:
        batch = generate_sum_input_batch(int_range=range, batch_size=batch_size, seed=seed)

    if only_int:
        return batch
    else:
        rng = np.random.default_rng(seed=seed)
        # __ = (range[1] - range[0]) * rng.random(size=(batch_size, 2)) + range[0]  # alternative
        return batch + rng.random(size=(batch_size, 2))


def generate_discriminate_output_batch(inputs):
    """Generate the targets for the discrimination problem. The problem is to determine whether the first quantity is
    greater than the second one."""
    output_batch = []
    for couple in inputs:
        if couple[0] > couple[1]:
            output_batch.append(True)
        else:
            output_batch.append(False)
    return np.array(output_batch)


if __name__ == '__main__':
    input_batch = generate_discriminate_input_batch(only_int=True, only_these=[1, 3, 5, 7])
    print(input_batch)
    print(generate_discriminate_output_batch(input_batch))
