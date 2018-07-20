"""This is the main body where we will set up our data and train our model"""
import pandas as pd
import numpy as np
import tensorflow as tf
import process_data
import nn_functions
import math
from sklearn import metrics

# get data
toronto_weather_data = pd.read_csv("weather data/neural net past 3 days weather data (csv).csv", sep=",")
# get rid of na values
toronto_weather_data = toronto_weather_data.dropna(0, 'any')
# randomize order
toronto_weather_data = toronto_weather_data.reindex(np.random.permutation(toronto_weather_data.index))
# To get info on data, uncomment the following line:
# print(toronto_weather_data.describe())

# There are 1582 rows, so we take 1150 as training data, 332 as validation data, and 100 as test data
training_features = process_data.process_features(toronto_weather_data.head(1150))
training_targets = process_data.process_target(toronto_weather_data.head(1150))

validation_features = process_data.process_features(toronto_weather_data[1150:1482])
validation_targets = process_data.process_target(toronto_weather_data[1150:1482])

test_features = process_data.process_features(toronto_weather_data.tail(100))
test_targets = process_data.process_target(toronto_weather_data.tail(100))

# We'll do stochastic gradient descent, so batch size = 1
nn_model = nn_functions.train_nn_regression_model(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.005),
                                                  batch_size=1,
                                                  steps=600,
                                                  hidden_units=[3, 3],
                                                  training_features=training_features,
                                                  training_targets=training_targets,
                                                  validation_features=validation_features,
                                                  validation_targets=validation_targets)

# We can test on test data
# make the input function
predict_test_input_fn = nn_functions.make_predict_input_fn(test_features, test_targets)
# get the predictions
test_predictions = nn_functions.np_array_predictions(dnn_regressor=nn_model[0], input_fn=predict_test_input_fn)
test_rmsl = math.sqrt(metrics.mean_squared_error(test_predictions, test_targets))
print("test rmsl = {}".format(test_rmsl))


