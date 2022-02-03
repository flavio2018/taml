"""This script trains a model on the task."""
import keras.losses
import click
from codetiming import Timer
from humanfriendly import format_timespan

from src.data.batch_generators import generate_input_batch, generate_output_batch
from src.models.models import build_mlp


@click.command()
@click.option("--use_activation", type=bool)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(use_activation):
    batch_size = 2000 if use_activation else 1000
    inputs = generate_input_batch(batch_size=batch_size)
    outputs = generate_output_batch(inputs)

    model = build_mlp(use_activation=use_activation)
    model.fit(inputs, outputs, epochs=4, batch_size=1)

    inputs_test = generate_input_batch(batch_size=100, seed=321)
    outputs_test = generate_output_batch(inputs_test)
    predictions = model.predict(inputs_test)
    print(predictions[0], outputs_test[0])
    print("MAE:", keras.losses.mae(predictions, outputs_test))
    print("MSE:", keras.losses.mse(predictions, outputs_test))


if __name__ == "__main__":
    main()
