# simple_train
Simple no-frills code to train and evaluate a one hidden layer neural network in Tensorflow2.0. The supported dataset is currently CIFAR-10. Two models are available for training:
- **'One-hidden'**: one hidden layer neural network with ReLU activations. Implements `f(x) = V*ReLU(W*x + b1) + b2`, where the bias terms `b1` and `b2` are optional.
- **'Linear'**: same as 'One-hidden', but without ReLU activations.

## Usage
To see all options available to customize model architecture and training take a look at `args.py` or run
```
python train_model.py -h
```
This repository includes two scripts to train (`train_model.py`) and run (`run_model.py`) a one hidden layer model. You can train a model as easily as 
```
python train_model.py --checkpoint-path $OUTPUT_PATH
```
and evaluate it on the test dataset using
```
python run_model.py --checkpoint-path $CHECKPOINT_PATH
```
Make sure to replace `$OUTPUT_PATH` with the path where you want the trained model to be saved, and replace `$CHECKPOINT_PATH` with the path to the checkpoint to be loaded (in this example they can be the same). Here is an example how to train a one hidden layer neural network with 128 hidden units using Adam optimizer:
```
python train_model.py --checkpoint-path $OUTPUT_PATH \
--architecture onehidden --hidden-units 128 --use-bias\
--optimizer adam --lr 0.001 --epochs 200
```

To run the network make sure the architecture exactly matches the saved model:
```
python run_model.py --checkpoint-path $CHECKPOINT_PATH \
--architecture onehidden --hidden-units 128 --use-bias
```

## Requirements:

- `tensorflow >= 2.0`
- `numpy`
