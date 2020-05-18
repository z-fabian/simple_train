import tensorflow as tf


def normalize(x):
    m = tf.reduce_mean(x, axis=1, keepdims=True)
    s = tf.math.reduce_std(x, axis=1, keepdims=True)
    return (x - m)/s


def shuffle_data(x, y):
    n = x.shape[0]
    random_order = tf.random.shuffle(tf.range(n))
    x = tf.gather(x, random_order, axis=0)
    y = tf.gather(y, random_order, axis=0)
    return x, y


def get_train_val_split(x, y, shuffle, val_split, train_subsample):
    assert 0 < val_split <= 1.0
    assert 0 < train_subsample <= 1.0
    if shuffle:
        x, y = shuffle_data(x, y)
    n = int(x.shape[0]*train_subsample)
    x = x[:n, ...]
    y = y[:n]
    n_val_data = int(x.shape[0] * val_split)
    x_val = x[:n_val_data, ...]
    y_val = y[:n_val_data, ...]
    x_train = x[n_val_data:, ...]
    y_train = y[n_val_data:, ...]
    return x_train, y_train, x_val, y_val


def subsample_classes(x, y, num_classes):
    indices = tf.where(tf.less(y, num_classes))[:, 0]
    indices = tf.cast(indices, tf.int32)
    x = tf.gather(x, indices, axis=0)
    y = tf.gather(y, indices, axis=0)
    return x, y


def get_cifar10_data(val_split=0.1, shuffle=True, train_subsample=1.0, num_classes=10):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = tf.cast(x_train, dtype=tf.float32)
    x_test = tf.cast(x_test, dtype=tf.float32)

    # Subsample classes
    x_train, y_train = subsample_classes(x_train, y_train, num_classes)
    x_test, y_test = subsample_classes(x_test, y_test, num_classes)

    # Flatten images
    x_train = tf.reshape(x_train, shape=(x_train.shape[0], -1))
    x_test = tf.reshape(x_test, shape=(x_test.shape[0], -1))

    # One-hot encode labels
    y_train = tf.squeeze(tf.one_hot(y_train, depth=num_classes), axis=1)
    y_test = tf.squeeze(tf.one_hot(y_test, depth=num_classes), axis=1)

    # Normalize examples
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # Create train and validation datasets
    x_train, y_train, x_val, y_val = get_train_val_split(x_train,
                                                         y_train,
                                                         shuffle=shuffle,
                                                         val_split=val_split,
                                                         train_subsample=train_subsample)
    return x_train, y_train, x_val, y_val, x_test, y_test


def create_dataset(x, y, batch_sz=64, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=x.shape[0])
    dataset = dataset.batch(batch_sz)
    return dataset
