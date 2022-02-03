"""This script trains a model on the task."""
import keras.losses
from codetiming import Timer
from humanfriendly import format_timespan

from src.data.batch_generators import generate_input_batch, generate_output_batch
from src.data.data_ops import pad_output_batch, one_hot_encode_output
from src.models.models import build_rnn, build_mlp


@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main():
    batch_size = 1000
    inputs = generate_input_batch(batch_size=batch_size)
    outputs = generate_output_batch(inputs)

    outputs = pad_output_batch(outputs)
    inputs = one_hot_encode_output(inputs).reshape((batch_size, -1))
    outputs = one_hot_encode_output(outputs).reshape((batch_size, -1))
    print(inputs[0], inputs[0].shape)

    model = build_mlp(n_inputs=inputs.shape[1], n_outputs=outputs.shape[1])
    model.fit(inputs, outputs, epochs=10, batch_size=1)

    test_batch_size = 100
    inputs_test = generate_input_batch(batch_size=test_batch_size, seed=321)
    outputs_test = generate_output_batch(inputs_test)
    inputs_test = one_hot_encode_output(inputs_test).reshape((test_batch_size, -1))
    outputs_test = pad_output_batch(outputs_test)
    outputs_test = one_hot_encode_output(outputs_test).reshape((test_batch_size, -1))

    predictions = model.predict(inputs_test)
    print(predictions[0], outputs_test[0])
    print("MAE:", keras.losses.mae(predictions, outputs_test))
    print("MSE:", keras.losses.mse(predictions, outputs_test))


if __name__ == "__main__":
    main()
