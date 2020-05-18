import argparse
import pathlib


class Args(argparse.ArgumentParser):

    def __init__(self, **overrides):

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--checkpoint-path', type=str, required=True,
                          help='Path to the folder to save the model after training or load the model for inference')
        self.add_argument('--num-classes', default=10, type=int, help='Number of classes. If less than 10, CIFAR-10 '
                                                                      'will be subsampled.')

        # Model parameters
        self.add_argument('--architecture', choices=['onehidden', 'linear'], default='onehidden',
                          help='Model architecture, in case of "onehidden" the model is f(x) = ReLU(x*W+b_1)*V + b2'
                               'x: input, (num_examples x input_dim)'
                               'W: hidden layer, (input_dim x hidden_units)'
                               'b1: hidden bias, (hidden_units,)'
                               'V: output layer, (hidden_units, num_classes)'
                               'b2: output bias, (num_classes,)'
                          'For "linear" there is no ReLU activation after the hidden layer.')
        self.add_argument('--hidden-units', default=128, type=int, help='Number of hidden units')
        self.add_argument('--use-bias', action='store_true',
                          help='If set, trainable bias term is added to the model in both input and output layer')

        # Training
        self.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
        self.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd',
                          help='Optimizer used in training')
        self.add_argument('--lr', default=0.01, type=float, help='Learning rate')
        self.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
        self.add_argument('--train-batch', default=64, type=int, help='Training batch size')
        self.add_argument('--val-batch', default=64, type=int, help='Validation batch size')
        self.add_argument('--test-batch', default=64, type=int, help='Test batch size')
        self.add_argument('--train-subsample', default=1.0, type=float,
                          help='Fraction of training data used for training')
        self.add_argument('--val-split', default=0.1, type=float,
                          help='Fraction of training data used for validation')
        self.add_argument('--shuffle-before-split', action='store_true',
                          help='If set, shuffle dataset before splitting into train and val')
        self.add_argument('--shuffle', action='store_true',
                          help='If set, shuffle train data after each epoch')
        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')

        self.set_defaults(**overrides)
