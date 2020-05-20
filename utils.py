import tensorflow as tf
import pickle
from pathlib import Path


def evaluate(model, val_dataset,
             loss_fn=tf.keras.losses.MeanSquaredError(),
             acc_metric=tf.keras.metrics.CategoricalAccuracy()):
    loss_metric = tf.keras.metrics.Mean()
    acc_metric.reset_states()
    loss_metric.reset_states()
    for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        preds = model(x_batch_val)
        loss = loss_fn(preds, y_batch_val)
        loss_metric.update_state(loss)
        acc_metric.update_state(y_batch_val, preds)
    return loss_metric.result(), acc_metric.result()


def save_model_and_config(checkpoint_path, model, config):
    save_path = Path(checkpoint_path)
    model.save_weights(checkpoint_path)
    save_config(str(save_path.parents[0]) + '/config', config)


def load_config(checkpoint_path):
    save_path = Path(checkpoint_path)
    config_path = str(save_path.parents[0]) + '/config'
    with open(config_path, 'rb') as file:
        config = pickle.load(file)
        return config


def save_config(path, config):
    with open(path, 'wb') as file:
        pickle.dump(config, file)
