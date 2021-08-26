import numpy as np
import pandas as df 
import tensorflow as tf
import os,sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import copy
import shap
import random
from sklearn.svm import SVC
from tensorflow.python.platform import flags
from fairness_data.census import census_data
from fairness_data.bank import bank_data
from fairness_data.compas import compas_data
from fairness_utils.config import census, bank, compas
from fairness_utils.utils import getRandomIndex
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from tensorflow import keras
FLAGS = flags.FLAGS


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


def check_for_error_condition(model, conf, t, sens):
    t = t.astype('int')
    label = model.predict(np.array([t]))
    if FLAGS.model_name == "NN":
        label = np.array([int(x > 0.5) for [x] in label])
    else:
        label = np.array(label).astype('int')
    # check for all the possible values of sensitive feature
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != t[sens-1]:
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            label_new = model.predict(np.array([tnew]))
            if FLAGS.model_name == "NN":
                label_new = np.array([int(x > 0.5) for [x] in label_new])
            else:
                label_new = np.array(label_new).astype('int')
            if label_new != label:
                return True
    return False


def init(dataset, sensitive_param, max_iter, sample_limit, model_name):
    data = {
        "census":census_data,
        "bank":bank_data,
        "compas": compas_data
    }
    data_config = {
        "census":census,
        "bank":bank,
        "compas":compas
    }
    model_config = {
        "LogisticRegression": LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=max_iter),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "MLPClassifier": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(3,), max_iter=max_iter, learning_rate='invscaling', random_state=0)),
        "SVC": SVC(kernel='rbf', probability=True, max_iter=max_iter),
    }
    invert_model_config = {
        "LogisticRegression": LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=max_iter),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "MLPClassifier": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(3,), max_iter=max_iter, learning_rate='invscaling', random_state=0)),
        "SVC": SVC(kernel='rbf', probability=True, max_iter=max_iter),
    }
    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    Y = Y[:,1]
    train_test_index = getRandomIndex(len(X), len(X))
    X = X[train_test_index]
    Y = Y[train_test_index]
    invert_X = invert_sensitive(X, sensitive_param, dataset)

    if model_name == "NN":
        model = keras.models.load_model("../models/original_models/" + dataset + ".h5")
        invert_model = keras.models.load_model("../models/invert_models/" + dataset + "_" + str(sensitive_param) + ".h5")
    else:
        model = model_config[model_name]
        model.fit(X, Y)
        invert_model = invert_model_config[model_name]
        invert_model.fit(invert_X, Y)

    total_init = 0
    init_list = []
    init_index = []

    for i in range(len(X)):
        y_orig = model.predict(np.array([X[i]]))
        y_new = invert_model.predict(np.array([X[i]]))

        if model_name == "NN":
            y_orig = np.array([int(x > 0.5) for [x] in y_orig])
            y_new = np.array([int(x > 0.5) for [x] in y_new])
        else:
            y_orig = np.array(y_orig).astype('int')
            y_new = np.array(y_new).astype('int')

        if y_orig != y_new:
            total_init += 1
            if check_for_error_condition(model, data_config[dataset], X[i], sensitive_param):
                temp = X[i].astype('int').tolist()
                init_list.append(temp)
                init_index.append(i)

    init_list = np.array(init_list)
    init_list = init_list[getRandomIndex(len(init_list), len(init_list))]
    init_list = init_list.tolist()

    # shap
    shap.initjs()
    if model_name == 'LogisticRegression':
        explainer = shap.LinearExplainer(model, X[0:2000])
        idi = np.array(init_list)
        shap_values = explainer.shap_values(idi)
    elif model_name == 'DecisionTreeClassifier':
        explainer = shap.TreeExplainer(model, X[0:2000])
        idi = np.array(init_list)
        shap_values = explainer.shap_values(idi)[0]
    elif model_name == 'SVC':
        explainer = shap.KernelExplainer(model.predict, X[0:2000])
        idi = np.array(init_list)
        shap_values = explainer.shap_values(idi[0: min(len(idi), 2000)])
    elif model_name == 'NN':
        idi = np.array(init_list)
        e = shap.DeepExplainer(model, X[0:2000])
        shap_values = e.shap_values(idi[0: min(len(idi), 2000)])[0]

    # shap_values DBSCAN
    cluster_model = DBSCAN(eps=0.09, min_samples=10)
    cluster_labels = cluster_model.fit(shap_values).labels_

    index = []
    for label_ in range(np.min(cluster_labels), np.max(cluster_labels)+1):
        num = 0
        for i in range(len(cluster_labels)):
            if label_ == cluster_labels[i]:
                index.append(i)
                num+=1
                if num == int(sample_limit/(np.max(cluster_labels)-np.min(cluster_labels)+1)):
                    break
    for i in range(len(cluster_labels)):
        if len(index) >= sample_limit:
            break
        elif i not in index:
            index.append(i)

    idi = idi[index].tolist()
    for i in range(len(X)):
        if len(idi) >= sample_limit:
            break
        if i not in init_index:
            idi.append(X[i])
    for i in range(len(X)):
        if len(init_list) >= sample_limit:
            break
        if i not in init_index:
            init_list.append(X[i])
            init_index.append(i)

    # create the folder for storing the fairness testing result
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    if not os.path.exists('../results/'+ dataset + '/'+ str(sensitive_param) + '/'):
        os.makedirs('../results/' + dataset + '/'+ str(sensitive_param) + '/')

    # storing the fairness testing result
    np.save('../results/' + dataset + '/' + str(sensitive_param) + '/'+model_name+'_w_I_init_samples.npy', np.array(init_list))
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/'+model_name+'_w_I_D_init_samples.npy', np.array(idi))

    # print the overview information of result
    print("Total Inputs are " + str(total_init))
    print("Total discriminatory inputs- " + str(len(init_list)))
    print("Total init samples- " + str(len(idi)))


def main(argv=None):
    init(dataset=FLAGS.dataset,
         sensitive_param=FLAGS.sens_param,
         max_iter=FLAGS.max_iter,
         sample_limit=FLAGS.sample_limit,
         model_name=FLAGS.model_name)


if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_integer('sens_param', 1, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_integer('max_iter', 30, 'maximum iteration of global perturbation')
    flags.DEFINE_integer('sample_limit', 1000, 'number of idi sample')
    flags.DEFINE_string('model_name', 'LogisticRegression', 'ML Models')

    tf.app.run()
