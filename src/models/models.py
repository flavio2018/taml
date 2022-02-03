from tensorflow import keras


def build_rnn(n_units=100):
    model = keras.Sequential(
        layers=[
            keras.layers.Input(shape=(60, )),
            keras.layers.Embedding(input_dim=10, output_dim=128, input_length=60),
            keras.layers.SimpleRNN(n_units, activation="relu", return_sequences=True),  # TODO mask first output
        ])
    model.compile(optimizer="sgd", loss="mse", metrics="accuracy")
    return model


def build_mlp(n_inputs=60, n_outputs=59, use_activation=False):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_inputs, )))
    model.add(keras.layers.Dense(units=n_outputs))
    if use_activation:
        model.add(keras.layers.ReLU())
    model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
    return model
