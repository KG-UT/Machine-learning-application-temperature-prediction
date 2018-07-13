"""This will contain several functions essential for the training of our data"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
import math
from typing import Optional


def make_feature_columns(features: 'DataFrame') -> set:
    """ Construct the TensorFlow feature columns.

    :param features: The names of the numerical input features to use
    :return: A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in features])


def input_fn(features: 'DataFrame', targets: 'DataFrame', batch_size: int=1, shuffle: bool=True,
             num_epochs: Optional[int]=None) -> tuple:
    """ Trains a neural network model

    :param features: pandas DataFrame of features
    :param targets: pandas DataFrame of targets
    :param batch_size: int that represents size of batch to be passed to model
    :param shuffle: boolean value to determine whether to shuffle the data or not
    :param num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    :return: Tuple of (features, labels) for next data batch
    """
    # convert pandas dataframe into a dict of np arrays
    features = {key:np.array(value) for key,value in dict(features).items()}

    # construct a dataset, and deal with batching/repeating
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # shuffle the data
    # shuffle all data(2000 exceeds total rows)
    if shuffle:
        ds = ds.shuffle(2000)

    # return next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def np_array_predictions(dnn_regressor: 'DNNRegressor', input_fn: callable) -> 'array':
    """ predicts values for examples given in input_fn using dnn_regressor

    :param dnn_regressor: A 'DNNRegressor' object
    :param input_fn: A function to be passed as an argument to dnn_regressor.predict(). Contains relevant examples.
    :return: An Numpy array containing the predictions
    """
    predictions = dnn_regressor.predict(input_fn=input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    return predictions


def make_predict_input_fn(features: 'DataFrame', targets: 'DataFrame') -> callable:
    """ Makes an input function specified for training or prediction, using given features and targets.

    :param features: features to be used by input function
    :param targets: targets to be used by input function
    :param predict: bool representing whether this is for prediction or not
    :return: An input function
    """
    return lambda: input_fn(features=features,
                            targets=targets['met0'],
                            num_epochs=1,
                            shuffle=False)


def train_nn_regression_model(optimizer, steps, batch_size, hidden_units,  training_features, training_targets,
             validation_features, validation_targets):
    """ Trains a neural network regression model. In addition, this function will print training loss
    and validation loss to keep track of training progress.

    :param optimizer: An instance of 'tf.train.Optimizer'. This is the optimizer to be used.
    :param steps: A non-zero int representing the total number of steps we will train the model for.
    :param batch_size: a non-zero int representing the batch size.
    :param hidden_units: A list of ints such that the number of items in represent the number of hidden layers,
                         and the ints represent the number of nodes in that respective hidden layer.
    :param training_features: A pandas DataFrame containing the column(s) to be used as input features for training.
    :param training_targets: A pandas DataFrame containing a single column representing the target, to be used
                             for training.
    :param validation_features: A pandas DataFrame containing the column(s) to be used as input features to be used
                                for validation.
    :param validation_targets: A pandas DataFrame containing a single column representing the target, to be used for
                               validation.
    :return: A tuple of '(estimator, training_losses, validation_losses)':
    estimator: The trained 'DNNRegressor' object
    training_loss: A list containing the training losses during training
    validation_loss: A list containing the validation losses during training
    """
    # we will split training into 10 periods
    periods = 10
    steps_per_period = steps // periods

    # first get rid of outliers
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    # create DNNRegressor object
    dnn_regressor = tf.estimator.DNNRegressor(feature_columns=make_feature_columns(training_features),
                                              hidden_units=hidden_units,
                                              optimizer=optimizer)

    # create the input functions for training and validation
    # I have no clue why we are passing training_targets[target_name] for targets, since that should just be a series
    # and I think we should be passing a DataFrame ?? But google did it in their tutorial, so I'll do it here
    training_input_fn = lambda: input_fn(features=training_features,
                                         targets=training_targets['met0'],
                                         batch_size=batch_size)
    predict_training_input_fn = make_predict_input_fn(features=training_features,
                                                      targets=training_targets)
    predict_validation_input_fn = make_predict_input_fn(features=validation_features,
                                                        targets=validation_targets)

    # root mean squared loss lists
    training_rmsl = []
    validation_rmsl = []

    # Train model in a loop so we can periodically compute loss
    for period in range(0, periods):
        # training
        print('period..'+str(period))
        dnn_regressor.train(input_fn=training_input_fn,
                            steps=steps_per_period)

        # get predictions to compute loss
        training_predictions = np_array_predictions(dnn_regressor=dnn_regressor, input_fn=predict_training_input_fn)
        validation_predictions = np_array_predictions(dnn_regressor=dnn_regressor, input_fn=predict_validation_input_fn)

        # compute loss for training and validation set
        periodic_training_rsml = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        periodic_validation_rsml = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))

        # print loss
        print("training rmsl = {}".format(periodic_training_rsml))
        print("validation rmsl = {}".format(periodic_validation_rsml))

        # add periodic loss to the loss list
        training_rmsl.append(periodic_training_rsml)
        validation_rmsl.append(periodic_validation_rsml)

    print('dnn_regressor fully trained')
    return dnn_regressor, training_rmsl, validation_rmsl
