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
@click.option('--range_start', type=int, default=1)
@click.option('--range_end', type=int, default=50)
@click.option('--only_int', type=bool)
@click.option('--only_even', type=bool)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(inputs_filepath, targets_filepath, num_batches, range_start, range_end, only_int, only_even):
    if only_even:
        only_these = [n for n in range(range_start, range_end) if n % 2 == 0]
    else:
        only_these = None

    for __ in range(num_batches):
        input_batch = generate_discriminate_input_batch(range=(range_start, range_end),
                                                        only_int=only_int,
                                                        only_these=only_these)
        output_batch = generate_discriminate_output_batch(input_batch)

        with open(inputs_filepath, "a+") as inputs_data_file:
            _write_batch(input_batch, inputs_data_file)

        with open(targets_filepath, "a+") as targets_data_file:
            for el in output_batch:
                targets_data_file.write(str(el) + "\n")


if __name__ == "__main__":
    main()
