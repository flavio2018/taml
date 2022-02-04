"""This script trains a model on the task."""
import click
from codetiming import Timer
from humanfriendly import format_timespan

from src.data.batch_generators import generate_input_batch, generate_output_batch
from src.data.data_ops import one_hot_encode_batch
from src.models.models import build_mlp
from src.models.metrics import evaluate_preds, plot_confusion_matrix


#  https://stackoverflow.com/questions/44232898/memoryerror-in-tensorflow-and-successful-numa-node-read-from-sysfs-had-negativ#44233285
@click.command()
@click.option("--train_name", default="mlp_one_hot_1_10")
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(train_name):
    batch_size = 1000
    inputs = generate_input_batch(batch_size=batch_size)
    outputs = generate_output_batch(inputs)

    inputs = one_hot_encode_batch(inputs, max_sum=40).reshape((batch_size, -1))
    # outputs = one_hot_encode_output(outputs).reshape((batch_size, -1))

    model = build_mlp(n_inputs=inputs.shape[1], n_outputs=outputs.shape[1])
    model.fit(inputs, outputs, epochs=10, batch_size=1)

    model.save(f"../models/trained/{train_name}")

    test_batch_size = 100
    inputs_test = generate_input_batch(int_range=(1, 21), batch_size=test_batch_size, seed=321)
    outputs_test = generate_output_batch(inputs_test)
    inputs_test = one_hot_encode_batch(inputs_test, max_sum=40).reshape((test_batch_size, -1))
    # outputs_test = one_hot_encode_output(outputs_test).reshape((test_batch_size, -1))

    predictions = model.predict(inputs_test)
    evaluate_preds(predictions, outputs_test)
    plot_confusion_matrix(predictions, outputs_test, train_name)


if __name__ == "__main__":
    main()
