"""This is the main body where we will set up our data and train our model"""
import pandas as pd
import numpy as np
import ProcessData

# get data
toronto_weather_data = pd.read_csv("weather data/neural net past 3 days weather data (csv).csv", sep=",")
# get rid of na values
toronto_weather_data = toronto_weather_data.dropna(0, 'any')
# randomize order
toronto_weather_data = toronto_weather_data.reindex(np.random.permutation(toronto_weather_data.index))

# There are 1582 rows, so we take 1200 as training data, 322 as validation data, and 300 as test data
training_features = ProcessData.process_features(toronto_weather_data.head(1200))
training_targets = ProcessData.process_target(toronto_weather_data.head(1200))

validation_features = ProcessData.process_features(toronto_weather_data.loc[1200:1521])
validation_targets = ProcessData.process_target(toronto_weather_data.loc[1200:1521])

test_features = ProcessData.process_features(toronto_weather_data.tail(300))
test_targets = ProcessData.process_target(toronto_weather_data.tail(300))



