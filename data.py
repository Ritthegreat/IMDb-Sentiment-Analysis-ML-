import tensorflow_datasets as tfds
import numpy as np
import os, pickle


# cache waste hella storage
def load_dataset(name: str, cache_dir="cache", cache=False):
    train_data, test_data = tfds.load(
        name, split=["train", "test"], as_supervised=True, data_dir=cache_dir
    )
    if not cache:
        return tfds_as_np(train_data), tfds_as_np(test_data)

    if not os.path.exists("cache/train_np.pkl"):
        with open("cache/train_np.pkl", "wb") as f:
            pickle.dump(tfds_as_np(train_data), f)
    if not os.path.exists("cache/test_np.pkl"):
        with open("cache/test_np.pkl", "wb") as f:
            pickle.dump(tfds_as_np(test_data), f)
    with open("cache/train_np.pkl", "rb") as f:
        train_np = pickle.load(f)
    with open("cache/test_np.pkl", "rb") as f:
        test_np = pickle.load(f)
    return train_np, test_np


def tfds_as_np(ds):
    return np.array(
        [(text.decode("utf-8"), label) for text, label in tfds.as_numpy(ds)]
    )
