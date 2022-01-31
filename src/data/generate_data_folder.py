"""This script creates a stable version of the dataset to be used for training,
using the data generator functions."""
import click
from codetiming import Timer
from humanfriendly import format_timespan

from src.data.batch_generators import generate_input_batch, generate_output_batch


@click.command()
@click.option('--output_filepath', type=click.Path())
@click.option('--num_batches', type=int)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(output_filepath, num_batches):
    with open(output_filepath, "w+") as input_data_file:
        for __ in range(num_batches):
            input_batch = generate_input_batch()
            output_batch = generate_output_batch(input_batch)
            for sample in output_batch:
                input_data_file.write(" ".join([str(digit) for digit in sample]))  # sample is an array of 59 digits
                input_data_file.write("\n")


if __name__ == "__main__":
    main()
