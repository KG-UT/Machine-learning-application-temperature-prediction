"""We get the data in csv form, turn it into a DataFrame, then return features and targets"""
import process_data
import numpy as np
import pandas as pd

# Initially this was just a function returning a dictionary, but pycharm wouldn't stop yelling at me for using the
# dictionary keys
class ModelSets:
    """gets data from path, turns it into a DataFrame, then segregates it into relevant data sets for our model
    """
    data: 'DataFrame'
    training_features: 'DataFrame'
    training_targets: 'DataFrame'
    validation_features: 'DataFrame'
    validation_targets: 'DataFrame'
    test_features: 'DataFrame'
    test_targets: 'DataFrame'

    def __init__(self, data_pathway):
        # get data
        data = pd.read_csv(data_pathway, sep=",")
        # get rid of na values
        data = data.dropna(0, 'any')
        # randomize order
        self.data = data.reindex(np.random.permutation(data.index))
        # to get info on data, use .describe()

        # want to separate data as follows: 2/3 for training, 2/9 for validation, and 1/9 for test
        total_data_amount = data.shape[0]
        training_amount = total_data_amount//3
        validation_amount = (total_data_amount*2)//9
        test_amount = total_data_amount - training_amount - validation_amount

        self.training_features = process_data.process_features(data.head(training_amount))
        self.training_targets = process_data.process_target(data.head(training_amount))

        self.validation_features = process_data.process_features(data[training_amount:test_amount])
        self.validation_targets = process_data.process_target(data[training_amount:test_amount])

        self.test_features = process_data.process_features(data.tail(test_amount))
        self.test_targets = process_data.process_target(data.tail(test_amount))
