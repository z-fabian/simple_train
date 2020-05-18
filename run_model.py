import tensorflow as tf
import numpy as np
from data import create_dataset, get_csv
from models import OneHidden
from utils import evaluate


TEST_X_PATH = '../data/resnet18-cifar100-embeddings.csv'
TEST_Y_PATH = '../data/resnet18-cifar100-labels.csv'
CHECKPOINT_PATH = './checkpoints/resnet_cifar100'
TEST_BATCH = 64
NUM_CLASSES = 20
HIDDEN_UNITS = 64


# Create test dataset
x_test, y_test = get_csv(TEST_X_PATH), get_csv(TEST_Y_PATH)
y_test = tf.squeeze(tf.one_hot(y_test.astype(np.int32), depth=NUM_CLASSES), axis=1)

test_dataset = create_dataset(x=x_test,
                              y=y_test,
                              batch_sz=TEST_BATCH,
                              shuffle=False)

# Load model
model = OneHidden(hidden_units=HIDDEN_UNITS, num_classes=NUM_CLASSES)
model.load_weights(CHECKPOINT_PATH)

# Evaluate model
loss, acc = evaluate(model, test_dataset)
print('val mean loss = %s val accuracy = %s' % (loss, acc))

print('Done!')