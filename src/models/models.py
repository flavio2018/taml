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
