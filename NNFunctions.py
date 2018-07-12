"""This will contain several functions essential for the training of our data"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset

def make_feature_columns(features):
    """ Construct the TensorFlow feature columns.

    :param features: The names of the numerical input features to use
    :return: A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in features])


def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """ Trains a neural network

    :param features: pandas DataFrame of features
    :param targets: pandas DataFrame of targets
    :param batch_size: size of batch to be passed to model
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
