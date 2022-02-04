from tensorflow import keras


def build_rnn(n_units=100, input_size=2):
    model = keras.Sequential(
        layers=[
            keras.layers.Input(shape=(input_size, )),
            keras.layers.Embedding(input_dim=10, output_dim=128, input_length=input_size),
            keras.layers.SimpleRNN(n_units, activation="relu", return_sequences=True),  # TODO mask first output
        ])
    model.compile(optimizer="sgd", loss="mse", metrics="accuracy")
    return model


def build_mlp(n_inputs=60, n_outputs=None, use_activation=False):
    if n_outputs is None:
        n_outputs = n_inputs - 1  # in the case of raw numbers inputs
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_inputs, )))
    model.add(keras.layers.Dense(units=n_outputs))
    if use_activation:
        model.add(keras.layers.ReLU())
    model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
    return model
