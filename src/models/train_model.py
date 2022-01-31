"""This script trains a model on the task."""
from codetiming import Timer
from humanfriendly import format_timespan

import numpy as np

from src.data.batch_generators import generate_input_batch, generate_output_batch\
from src.data.data_ops import pad_output_batch
from src.models.models import build_rnn


@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main():
    BATCHSIZE = 100
    rnn = build_rnn()
    inputs = generate_input_batch(batch_size=BATCHSIZE)
    outputs = generate_output_batch(inputs)

    outputs = pad_output_batch(outputs)

    rnn.fit(inputs, outputs, epochs=15)


if __name__ == "__main__":
    main()
