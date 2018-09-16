""" Trains a decision tree model, then uses it to predict future temperature data"""
from get_features_and_targets import ModelSets
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

if __name__ == '__main__':
    # get our data to train our model
    data_sets = ModelSets('weather data/past 3 years weather data.csv')
    # here is our model to train
    decision_tree_model = DecisionTreeRegressor(max_leaf_nodes=20)
    # here we train our model
    decision_tree_model.fit(X=data_sets.training_features, y=data_sets.training_targets)

    # now let's get data from our user to predict with!
    print("This will be a simple application of our decision tree model.")
    print("We'll predict the mean temperature for a day, given the mean " +
          "temperature and precipitation of the 3 days prior")
    user_features = {}
    for i in range(3, 0, -1):
        feature_temp = "met{}".format(i)
        feature_prec = "prec{}".format(i)
        user_features[feature_temp] = [input("Please input the mean temperature {} days prior: ".format(i))]
        user_features[feature_prec] = [input("Please input the precipitation {} days prior: ".format(i))]
    user_features = pd.DataFrame(user_features)
    print("we predict that the average temperature will be: {}".format(decision_tree_model.predict(X=user_features)[0]))
