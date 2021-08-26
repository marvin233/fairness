import numpy as np
import pandas as pd
import tensorflow as tf
from fairness_utils.utils import getRandomIndex

# make outputs stable across runs
np.random.seed(0)
tf.random.set_seed(0)

# load adult dataset, and eliminate unneccessary features
data_path = ('datasets/census')
df = pd.read_csv(data_path, encoding='latin-1')

# encode categorical attributes to integers
data = df.values
data = data.astype(np.int32)

# split data into training data, validation data and test data
X = data[:, :-1]
y = data[:, -1]
train_num = int(len(X) * 0.6)
train_test_index = getRandomIndex(len(X), len(X))
X_train = X[train_test_index[:train_num]]
y_train = y[train_test_index[:train_num]]
X_test = X[train_test_index[train_num:]]
y_test = y[train_test_index[train_num:]]
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T

protected_attribs = [0, 7, 8]