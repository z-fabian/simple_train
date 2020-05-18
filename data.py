import tensorflow as tf
import pandas as pd
import numpy as np


def get_csv(path):
    df = pd.read_csv(path, header=None)
    return df.values.astype(np.float32)


def create_dataset(x, y, batch_sz=64, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=x.shape[0])
    dataset = dataset.batch(batch_sz)
    return dataset
