import tensorflow as tf
import numpy as np
from data import create_datasets, get_csv
from models import OneHidden
from utils import evaluate

TRAIN_X_PATH = '../data/resnet18-cifar100-embeddings.csv'
TRAIN_Y_PATH = '../data/resnet18-cifar100-labels.csv'
VAL_X_PATH = '../data/resnet18-imagenet-embeddings.csv'
VAL_Y_PATH = '../data/resnet18-imagenet-labels.csv'
CHECKPOINT_PATH = './checkpoints/resnet_cifar100'
TRAIN_BATCH = 64
VAL_BATCH = 64
NUM_CLASSES = 20
HIDDEN_UNITS = 64
LR = 0.001
EPOCHS = 5

# Create training and validation datasets
x_train, y_train, x_val, y_val = get_csv(TRAIN_X_PATH), get_csv(TRAIN_Y_PATH), get_csv(VAL_X_PATH), get_csv(VAL_Y_PATH)
y_train = tf.squeeze(tf.one_hot(y_train.astype(np.int32), depth=NUM_CLASSES), axis=1)
y_val = tf.squeeze(tf.one_hot(y_val.astype(np.int32), depth=NUM_CLASSES), axis=1)

train_dataset, val_dataset = create_datasets(x_train=x_train,
                                             y_train=y_train,
                                             x_val=x_val,
                                             y_val=y_val,
                                             train_batch=TRAIN_BATCH,
                                             val_batch=VAL_BATCH,
                                             shuffle_train=True)

# Create model and optimizer
model = OneHidden(hidden_units=HIDDEN_UNITS, num_classes=NUM_CLASSES)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

# Create loss function and metrics
mse_loss_fn = tf.keras.losses.MeanSquaredError()
loss_metric = tf.keras.metrics.Mean()
acc_metric = tf.keras.metrics.CategoricalAccuracy()

# Compile model
model.compile(optimizer, loss=mse_loss_fn)

model._set_inputs(inputs=x_train)  # Only needed in TF2.0 due to a bug in saving custom models. Will be fixed in TF2.2.
model.summary()

# Train the model
for epoch in range(EPOCHS):
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            preds = model(x_batch_train)
            loss = mse_loss_fn(preds, y_batch_train)
        grads = tape.gradient(loss, model.trainable_weights)

        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        loss_metric.update_state(loss)
        acc_metric.update_state(y_batch_train, preds)


    print('epoch %s: train mean loss = %s train accuracy = %s' % (epoch, loss_metric.result(), acc_metric.result()))

# Evaluate model
loss, acc = evaluate(model, val_dataset)
print('val mean loss = %s val accuracy = %s' % (loss, acc))

# Save weights
print('Saving model...')
model.save_weights(CHECKPOINT_PATH)

print('Done!')


