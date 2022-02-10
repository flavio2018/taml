"""This script trains a model on the task."""
import click
from codetiming import Timer
from humanfriendly import format_timespan

from src.data.batch_generators import generate_sum_input_batch, generate_sum_output_batch
from src.models.models import build_mlp


@click.command()
@click.option("--train_name", default="mlp_raw_1_10")
@click.option("--use_activation", type=bool)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(train_name, use_activation):
    batch_size = 2000 if use_activation else 1000
    inputs = generate_sum_input_batch(batch_size=batch_size)
    outputs = generate_sum_output_batch(inputs)

    model = build_mlp(use_activation=use_activation, n_inputs=2)
    model.fit(inputs, outputs, epochs=4, batch_size=1)

    model.save(f"../models/trained/{train_name}")


if __name__ == "__main__":
    main()
