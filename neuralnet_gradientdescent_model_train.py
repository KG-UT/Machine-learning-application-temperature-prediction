"""This is the main body where we will set up our data and train our model.
This uses neural networks, and gradient descent as the optimizer.
Much of this has been inspired by the google machine learning crash course."""
import tensorflow as tf
import nn_functions
import math
from sklearn import metrics
from get_features_and_targets import ModelSets

data_sets = ModelSets("weather data/past 3 years weather data.csv")

# We'll do stochastic gradient descent, so batch size = 1
nn_model = nn_functions.train_nn_regression_model(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.005),
                                                  batch_size=5,
                                                  steps=1000,
                                                  hidden_units=[3, 3],
                                                  training_features=data_sets.training_features,
                                                  training_targets=data_sets.training_targets,
                                                  validation_features=data_sets.validation_features,
                                                  validation_targets=data_sets.validation_targets)

# We can test on test data
# make the input function
predict_test_input_fn = nn_functions.make_predict_input_fn(data_sets.test_features, data_sets.test_targets)
# get the predictions
test_predictions = nn_functions.np_array_predictions(dnn_regressor=nn_model[0], input_fn=predict_test_input_fn)
test_rmsl = math.sqrt(metrics.mean_squared_error(test_predictions, data_sets.test_targets))
print("test rmsl = {}".format(test_rmsl))
