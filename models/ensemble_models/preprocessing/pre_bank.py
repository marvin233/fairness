import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# make outputs stable across runs
np.random.seed(42)
tf.set_random_seed(42)

# load bank dataset
data_path = ('../../datasets/bank')
df = pd.read_csv(data_path, encoding='latin-1')

# encode categorical attributes to integers
data = df.values
data = data.astype(np.int32)

# split data into training data, validation data and test data
X = data[:, :-1]
y = data[:, -1]
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

# set constraints for each attribute, 349920 data points in the input space
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T
protected_attribs = [0]