import tensorflow as tf
import numpy as np
from data import create_datasets, get_csv
from models import OneHidden

TRAIN_X_PATH = '../data/resnet18-cifar100-embeddings.csv'
TRAIN_Y_PATH = '../data/resnet18-cifar100-labels.csv'
VAL_X_PATH = '../data/resnet18-imagenet-embeddings.csv'
VAL_Y_PATH = '../data/resnet18-imagenet-labels.csv'
TRAIN_BATCH = 64
VAL_BATCH = 64
NUM_CLASSES = 20
HIDDEN_UNITS = 64
LR = 0.001
EPOCHS = 10

x_train, y_train, x_val, y_val = get_csv(TRAIN_X_PATH), get_csv(TRAIN_Y_PATH), get_csv(VAL_X_PATH), get_csv(VAL_Y_PATH)
y_train = tf.squeeze(tf.one_hot(y_train.astype(np.int32), depth=NUM_CLASSES), axis=1)
y_val = tf.squeeze(tf.one_hot(y_val.astype(np.int32), depth=NUM_CLASSES), axis=1)
n_train = x_train.shape[0]
n_val = x_train.shape[0]

train_dataset, val_dataset = create_datasets(x_train=x_train,
                                             y_train=y_train,
                                             x_val=x_val,
                                             y_val=y_val,
                                             train_batch=TRAIN_BATCH,
                                             val_batch=VAL_BATCH,
                                             shuffle_train=True)

model = OneHidden(hidden_units=HIDDEN_UNITS, num_classes=NUM_CLASSES)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
mse_loss_fn = tf.keras.losses.MeanSquaredError()
loss_metric = tf.keras.metrics.Mean()
model.compile(optimizer, loss=mse_loss_fn)

for epoch in range(EPOCHS):
    print('Start of epoch %d' % (epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
          preds = model(x_batch_train)
          loss = mse_loss_fn(preds, y_batch_train)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        loss_metric(loss)
    print('epoch %s: mean loss = %s' % (epoch, loss_metric.result()))

model.summary()

sample_input = x_train[0, :]
sample_input = sample_input[np.newaxis, ...]
print(model.predict(x=sample_input))
