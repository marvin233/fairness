import numpy as np
import pandas as pd
import tensorflow as tf
from fairness_utils.utils import getRandomIndex

# make outputs stable across runs
np.random.seed(0)
tf.random.set_seed(0)

# load german credit risk dataset
data_path = ('datasets/compas')
df = pd.read_csv(data_path)


# preprocess data
data = df.values.astype(np.int32)

# split data into training data and test data
X = data[:, :-1]
y = data[:, -1]
train_num = int(len(X) * 0.6)
train_test_index = getRandomIndex(len(X), len(X))
X_train = X[train_test_index[:train_num]]
y_train = y[train_test_index[:train_num]]
X_test = X[train_test_index[train_num:]]
y_test = y[train_test_index[train_num:]]


# set constraints for each attribute, 839808 data points in the input space
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T
protected_attribs = [0, 1, 2]