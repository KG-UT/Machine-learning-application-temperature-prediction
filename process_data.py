"""Convert given weather data from a DataFrame into processed data that can be used by TensorFlow"""
import pandas as pd


# Legend for data names:
# mat = max temp., mit = min temp., met = mean temp., prec = precipitation in mm
# Numbers indicate how many days ago. Temperature in Celsius.


def process_features(weather_data):
    """ Take weather_data DataFrame and return a DataFrame containing only the features to be used by our model

    :param weather_data: DataFrame containing all relevant weather data
    :return: DataFrame containing only the features
    """
    # get a DataFrame with all the features
    weather_data_features = weather_data[['month', 'mat3', 'mit3', 'met3', 'prec3',
                                          'mat2', 'mit2', 'met2', 'prec2',
                                          'mat1', 'mit1', 'met1', 'prec1']]
    # get a copy
    processed_features = weather_data_features.copy()
    return processed_features


def process_target(weather_data):
    """ Take weather_data DataFrame and return a DataFrame containing only the target to be used by our model

    :param weather_data: DataFrame containing all relevant weather data
    :return: DataFrame containing only the target
    """
    processed_target = pd.DataFrame()
    processed_target['met0'] = weather_data['met0']
    return processed_target
