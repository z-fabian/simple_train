import tensorflow as tf
from data import create_dataset, get_cifar10_data
from models import OneHidden
from utils import evaluate, save_model_and_config
from args import Args


def train_model(args):
    # Create training and validation datasets
    x_train, y_train, x_val, y_val, _, _ = get_cifar10_data(val_split=args.val_split,
                                                            shuffle=args.shuffle_before_split,
                                                            train_subsample=args.train_subsample,
                                                            num_classes=args.num_classes)
    train_dataset = create_dataset(x_train, y_train, batch_sz=args.train_batch, shuffle=args.shuffle)
    val_dataset = create_dataset(x_val, y_val, batch_sz=args.val_batch, shuffle=False)

    # Create model and optimizer
    if args.architecture == 'onehidden':
        use_relu = True
    elif args.architecture == 'linear':
        use_relu = False

    model = OneHidden(hidden_units=args.hidden_units,
                      num_classes=args.num_classes,
                      use_bias=args.use_bias,
                      use_relu=use_relu)
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Create loss function and metrics
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.CategoricalAccuracy()

    # Compile model
    model.compile(optimizer, loss=mse_loss_fn)

    model._set_inputs(inputs=x_train)  # Only needed in TF2.0 due to a bug in saving custom models. Will be fixed in TF2.2.
    model.summary()

    # Train the model
    for epoch in range(args.epochs):
        loss_metric.reset_states()
        acc_metric.reset_states()
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
    train_loss, train_acc = evaluate(model, train_dataset)
    print('Final train loss = %s train accuracy = %s' % (train_loss, train_acc))

    val_loss, val_acc = evaluate(model, val_dataset)
    print('Final val loss = %s val accuracy = %s' % (val_loss, val_acc))

    # Save weights
    print('Saving model...')
    save_model_and_config(args.checkpoint_path, model, args)

    print('Done!')


if __name__ == '__main__':
    args = Args().parse_args()
    tf.random.set_seed(args.seed)
    train_model(args)

