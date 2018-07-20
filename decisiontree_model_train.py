"""trains a decision tree model"""
from decisiontree_model_function import get_rmse
from get_features_and_targets import ModelSets
from sklearn.tree import DecisionTreeRegressor

data_sets = ModelSets('weather data/past 3 years weather data.csv')

# this was to find optimal number of leaf nodes.
# for leafs in [10, 20, 40, 60, 80, 100, 120, 140]:
#     rmse = get_rmse(leafs, training_features=data_sets.training_features, training_targets=data_sets.training_targets,
#                    validation_features=data_sets.validation_features, validation_targets=data_sets.validation_targets)
#     print(rmse)
# rmse decreases from 10 nodes to 20, then from 20 onwards increases. 20 seems most optimal.

# set max_leaf_nodes to 20 to prohibit overfitting or underfitting
decision_tree_model = DecisionTreeRegressor(max_leaf_nodes=20)
# here is our trained model
decision_tree_model.fit(X=data_sets.training_features, y=data_sets.training_targets)
# I could put some more lines to train this particular model, but instead I'll just train a different model
# using my previously built function since it's redundant not to use it.
print(get_rmse(20, training_features=data_sets.training_features, training_targets=data_sets.training_targets,
               validation_features=data_sets.validation_features, validation_targets=data_sets.validation_targets))
# could use following lines to find out relevant data info:
print(data_sets.data.describe())
print(data_sets.validation_targets.describe())
# rmse is ~3.39! That's about 1/3 of a standard deviation! That's really really good!
