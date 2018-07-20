"""This module will be where we set up our data, and train our model.
This uses a decision tree.
This has been inspired by the Kaggle machine learning tutorial."""
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import math


def get_rmse(max_leaf_nodes, training_features, training_targets,
             validation_features, validation_targets):
    """

    :param max_leaf_nodes: int representing max nodes for decision tree.
    :param training_features: DataFrame containing training features
    :param training_targets: DataFrame containing training targets
    :param validation_features: DataFrame containing validation features
    :param validation_targets: DataFrame containing validation targets
    :return: float representing root mean squared error, i.e. average error
    """
    decision_model = DecisionTreeRegressor(max_leaf_nodes= max_leaf_nodes)
    decision_model.fit(X=training_features, y=training_targets)
    validation_predictions = decision_model.predict(X=validation_features)
    rsme = math.sqrt(mean_squared_error(validation_targets, validation_predictions))
    return rsme


if __name__ == "__main__":
    # Just checking actual prediction vs actual values because 4 degrees error seems really low!
    from get_features_and_targets import ModelSets
    data_set = ModelSets('weather data/past 3 years weather data.csv')
    decision_model = DecisionTreeRegressor()
    decision_model.fit(X=data_set.training_features, y=data_set.training_targets)
    validation_predictions = decision_model.predict(X=data_set.validation_features)
    comparison_list = []
    for row_num in range(data_set.validation_features.shape[0]):
        comparison_list.append((data_set.validation_targets['met0'].iloc[row_num], validation_predictions[row_num]))
    print(comparison_list)