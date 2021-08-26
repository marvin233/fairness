import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from tensorflow import keras
from fairness_data.census import census_data
from fairness_data.bank import bank_data
from fairness_data.compas import compas_data
from fairness_utils.utils import getRandomIndex
import numpy as np


data = {"census": census_data, "bank": bank_data, "compas": compas_data}
dataset = "bank"

X, Y, _, _ = data[dataset]()
Y = Y[:, 1]
train_num = int(len(X) * 0.6)
train_test_index = getRandomIndex(len(X), len(X))
X_train = X[train_test_index[:train_num]]
Y_train = Y[train_test_index[:train_num]]
X_test = X[train_test_index[train_num:]]
Y_test = Y[train_test_index[train_num:]]

# create and train a six-layer neural network for the binary classification task
model = keras.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(15, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(5, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss=keras.losses.binary_crossentropy, optimizer="nadam", metrics=["accuracy"])
print('compile end')

# training
history = model.fit(X_train, Y_train, epochs=30)
model.evaluate(X_test, Y_test)
model.save("./original_models/"+dataset+".h5")
print(dataset, 'end')