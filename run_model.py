import tensorflow as tf
from data import create_dataset, get_cifar10_data
from models import OneHidden
from utils import evaluate, load_config
from args import Args

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def run_model(args):
    # Load model configuration
    config = load_config(args.checkpoint_path)
    # Create test dataset
    _, _, _, _, x_test, y_test = get_cifar10_data(num_classes=config.num_classes)

    test_dataset = create_dataset(x=x_test,
                                  y=y_test,
                                  batch_sz=args.test_batch,
                                  shuffle=False)

    # Load model
    if args.architecture == 'onehidden':
        use_relu = True
    elif args.architecture == 'linear':
        use_relu = False
    model = OneHidden(hidden_units=config.hidden_units,
                      num_classes=config.num_classes,
                      use_bias=config.use_bias,
                      use_relu=use_relu)
    model.load_weights(args.checkpoint_path)

    # Evaluate model
    loss, acc = evaluate(model, test_dataset)
    print('Test mean loss = %s test accuracy = %s' % (loss, acc))

    print('Done!')


if __name__ == '__main__':
    args = Args().parse_args()
    tf.random.set_seed(args.seed)
    run_model(args)
