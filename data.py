import tensorflow as tf
import pandas as pd
import numpy as np



def get_csv(path):
    df = pd.read_csv(path, header=None)
    return df.values.astype(np.float32)


def create_datasets(x_train, y_train, x_val, y_val, train_batch=64, val_batch=64, shuffle_train=True):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if shuffle_train:
        train_dataset = train_dataset.shuffle(buffer_size=10024)
    train_dataset = train_dataset.batch(train_batch)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).repeat()
    val_dataset = val_dataset.batch(val_batch)
    return train_dataset, val_dataset


# TODO: -fix shuffle buffer size