"""This script creates a stable version of the dataset to be used for training,
using the data generator functions."""
import click
from codetiming import Timer
from humanfriendly import format_timespan

from src.data.batch_generators import generate_discriminate_input_batch, generate_discriminate_output_batch, _write_batch


@click.command()
@click.option('--inputs_filepath', type=click.Path())
@click.option('--targets_filepath', type=click.Path())
@click.option('--num_batches', type=int)
@click.option('--n_range', type=tuple, default=(1, 50))
@click.option('--only_int', type=bool)
@click.option('--only_even', type=bool)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(inputs_filepath, targets_filepath, num_batches, n_range, only_int, only_even):
    if only_even:
        only_these = [n for n in range(*n_range) if n % 2 == 0]
    else:
        only_these = None
    for __ in range(num_batches):
        input_batch = generate_discriminate_input_batch(range=n_range, only_int=only_int, only_these=only_these)
        output_batch = generate_discriminate_output_batch(input_batch)

        with open(inputs_filepath, "w+") as inputs_data_file:
            _write_batch(input_batch, inputs_data_file)

        with open(targets_filepath, "w+") as targets_data_file:
            for el in output_batch:
                targets_data_file.write(str(el) + "\n")


if __name__ == "__main__":
    main()
