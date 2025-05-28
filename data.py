import tensorflow_datasets as tfds
import numpy as np


def load_dataset(name: str):
    train_data, test_data = tfds.load(
        name,
        split=["train", "test"],
        as_supervised=True,
    )
    return tfds_as_np(train_data), tfds_as_np(test_data)


def tfds_as_np(ds):
    return np.array([(text.decode("utf-8"), label) for text, label in tfds.as_numpy(ds)])
