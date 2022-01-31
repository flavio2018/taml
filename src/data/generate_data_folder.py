"""This script creates a stable version of the dataset to be used for training,
using the data generator functions."""
import click
from codetiming import Timer
from humanfriendly import format_timespan

from src.data.batch_generators import generate_input_batch, generate_output_batch


@click.command()
@click.argument('output_filepath', type=click.Path())
@click.argument('num_batches', type=int)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(output_file, num_batches):
    for __ in range(num_batches):
        input_batch = generate_input_batch()
        output_batch = generate_output_batch()


if __name__ == "__main__":
    main()
