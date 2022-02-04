"""This script trains a model on the task."""
import keras.losses
import click
from codetiming import Timer
from humanfriendly import format_timespan

from src.data.batch_generators import generate_input_batch, generate_output_batch
from src.models.models import build_mlp
from src.models.metrics import plot_confusion_matrix, evaluate_preds


@click.command()
@click.option("--train_name", default="mlp_raw_1_10")
@click.option("--use_activation", type=bool)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(train_name, use_activation):
    batch_size = 2000 if use_activation else 1000
    inputs = generate_input_batch(batch_size=batch_size)
    outputs = generate_output_batch(inputs)

    model = build_mlp(use_activation=use_activation, n_inputs=2)
    model.fit(inputs, outputs, epochs=4, batch_size=1)

    model.save(f"../models/trained/{train_name}")

    inputs_test = generate_input_batch(int_range=(1, 20), batch_size=100, seed=321)
    outputs_test = generate_output_batch(inputs_test)
    predictions = model.predict(inputs_test)
    evaluate_preds(predictions, outputs_test)
    plot_confusion_matrix(predictions, outputs_test, train_name)


if __name__ == "__main__":
    main()
