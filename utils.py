import tensorflow as tf


def evaluate(model, val_dataset,
             loss_fn=tf.keras.losses.MeanSquaredError(),
             acc_metric=tf.keras.metrics.CategoricalAccuracy()):
    loss_metric = tf.keras.metrics.Mean()
    for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        preds = model(x_batch_val)
        loss = loss_fn(preds, y_batch_val)
        loss_metric.update_state(loss)
        acc_metric.update_state(y_batch_val, preds)
    return loss_metric.result(), acc_metric.result()
