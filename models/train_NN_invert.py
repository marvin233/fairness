import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
import numpy as np
import random
import pandas as df
from tensorflow import keras
from fairness_utils.config import census, bank, compas
from fairness_data.census import census_data
from fairness_data.bank import bank_data
from fairness_data.compas import compas_data
from fairness_utils.utils import getRandomIndex


def generate_random(x, input_bounds):
    if x == input_bounds[0]:
        return random.randint(input_bounds[0]+1, input_bounds[1])
    elif x == input_bounds[1]:
        return random.randint(input_bounds[0], input_bounds[1]-1)
    else:
        if random.randint(0, 1) == 0:
            return random.randint(input_bounds[0], x-1)
        else:
            return random.randint(x+1, input_bounds[1])


def invert_sensitive(x, sens_index, dataset):
    sens_index = sens_index - 1
    x = df.DataFrame(x)
    input_bounds_config = {'census': census.input_bounds,
                           'bank': bank.input_bounds,
                           'compas': compas.input_bounds}
    input_bounds = input_bounds_config[dataset][sens_index]
    x.iloc[:, sens_index] = x.iloc[:, sens_index].astype('int')

    for i in range(len(x)):
        x.iloc[i, sens_index] = generate_random(x.iloc[i, sens_index], input_bounds)
    x = np.array(x)
    return x


data = {
    "census": census_data,
    "bank": bank_data,
    "compas": compas_data
}


for (dataset, sensitive_param) in [("census", 1), ("census", 8), ("census", 9), ("bank", 1), ("compas", 1), ("compas", 2), ("compas", 3)]:
    X, Y, _, _ = data[dataset]()
    Y = Y[:, 1]
    train_num = int(len(X) * 0.6)
    train_test_index = getRandomIndex(len(X), len(X))
    X_train = X[train_test_index[:train_num]]
    Y_train = Y[train_test_index[:train_num]]
    X_test = X[train_test_index[train_num:]]
    Y_test = Y[train_test_index[train_num:]]
    print('data end')
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
    invert_X = invert_sensitive(X_train, sensitive_param, dataset)
    history = model.fit(invert_X, Y_train, epochs=30)
    model.save("./invert_models/" + dataset+"_"+str(sensitive_param) + ".h5")
    print(dataset, 'end')