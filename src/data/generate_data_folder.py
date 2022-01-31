"""This script creates a stable version of the dataset to be used for training,
using the data generator functions."""
import click
from codetiming import Timer
from humanfriendly import format_timespan

from src.data.batch_generators import generate_input_batch, generate_output_batch


@click.command()
@click.option('--inputs_filepath', type=click.Path())
@click.option('--targets_filepath', type=click.Path())
@click.option('--num_batches', type=int)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(inputs_filepath, targets_filepath, num_batches):
    for __ in range(num_batches):
        input_batch = generate_input_batch()
        output_batch = generate_output_batch(input_batch)

        with open(inputs_filepath, "w+") as inputs_data_file:
            _write_batch(input_batch, inputs_data_file)

        with open(targets_filepath, "w+") as targets_data_file:
            _write_batch(output_batch, targets_data_file)


def _write_batch(batch, file):
    for sample in batch:
        file.write(" ".join([str(digit) for digit in sample]))  # sample is an array of 59 digits
        file.write("\n")


if __name__ == "__main__":
    main()
