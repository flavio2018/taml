"""This script implements the testing cases for the MLP models."""
import click
from codetiming import Timer
from humanfriendly import format_timespan

from tensorflow import keras

from src.data.batch_generators import generate_sum_input_batch, generate_sum_output_batch
from src.data.data_ops import one_hot_encode_batch
from src.models.metrics import plot_confusion_matrix, evaluate_preds


@click.command()
@click.option("--train_name", help="Codename of the trained model, also name of the checkpoint directory.")
@click.option("--test_name", help="Codename of the testing setting.")
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(train_name, test_name):
    model = keras.models.load_model(filepath=f"../models/trained/{train_name}")

    test_batch_size = 100
    if test_name == "interpolate":
        inputs_test = generate_sum_input_batch(int_range=(1, 10), batch_size=test_batch_size, seed=321)

    elif test_name == "extrapolate":
        inputs_test = generate_sum_input_batch(int_range=(1, 21), batch_size=test_batch_size, seed=321)

    else:
        print(f"Wrong test_name: {test_name}.")
        return

    outputs_test = generate_sum_output_batch(inputs_test)

    if "raw" in train_name:
        pass
    elif "one_hot" in train_name:
        inputs_test = one_hot_encode_batch(inputs_test, max_sum=40).reshape((test_batch_size, -1))
    else:
        print(f"Unexpected train_name: {train_name}.")

    predictions = model.predict(inputs_test)
    evaluate_preds(predictions, outputs_test)
    plot_confusion_matrix(predictions, outputs_test, "_".join([train_name, test_name]))
    

if __name__ == "__main__":
    main()
