import numpy as np

from batch_generators import generate_input_batch, generate_output_batch


def main():
    _test_input_batch_shape()
    _test_output_batch_shape()
    _test_sums_in_output_batch()
        

def _test_input_batch_shape():
    input_batch = generate_input_batch(batch_size=300)
    assert(input_batch.shape == (300, 60))


def _test_output_batch_shape():
    ones_matrix = np.ones((10, 60))
    output_batch = generate_output_batch(ones_matrix)
    assert(output_batch.shape == (10, 59))


def _test_sums_in_output_batch():
    ones_matrix = np.ones((10, 60))
    output_batch = generate_output_batch(ones_matrix)
    all_elements_are_twos = (output_batch == 2).all()
    assert(all_elements_are_twos == True)


if __name__ == '__main__':
    main()
