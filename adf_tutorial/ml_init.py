import numpy as np
import pandas as df 
import tensorflow as tf
import os,sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import copy
import shap
from tensorflow.python.platform import flags
from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_data.execution import execution_data
from adf_data.compas import compas_data
from adf_utils.config import census, credit, bank, execution, compas
from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_argmax
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import random
from xgboost import XGBRegressor
from sklearn.cluster import DBSCAN
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
    # check for all the possible values of sensitive feature
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != t[sens-1]:
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            label_new = model.predict(np.array([tnew]))
            if label_new != label:
                return True
    return False


def dnn_init(dataset, sensitive_param, max_iter, model_name):
    """
    The implementation of ADF
    :param dataset: the name of testing dataset
    :param sensitive_param: the index of sensitive feature
    :param max_iter: the maximum iteration of global perturbation
    """
    data = {"census":census_data, "credit":credit_data, "bank":bank_data, "execution": execution_data, "compas": compas_data}
    data_config = {"census":census, "credit":credit, "bank":bank, "execution":execution, "compas":compas}
    model_config = {
        "LogisticRegression": LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=max_iter),
        "SVC": SVC(kernel='rbf', probability=True, max_iter=max_iter),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "MLPClassifier": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(3,), max_iter=max_iter, learning_rate='invscaling', random_state=0)),
        "XGBRegressor": XGBRegressor(),
        "MLPRegressor": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(3,), activation='logistic', max_iter=max_iter, learning_rate='invscaling', random_state=0))
    }
    invert_model_config = {
        "LogisticRegression": LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=max_iter),
        "SVC": SVC(kernel='rbf', probability=True, max_iter=max_iter),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "MLPClassifier": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(3,), max_iter=max_iter, learning_rate='invscaling', random_state=0)),
        "XGBRegressor": XGBRegressor(),
        "MLPRegressor": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(3,), activation='logistic', max_iter=max_iter, learning_rate='invscaling', random_state=0))
    }
    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    Y = Y[:,1]
    model = model_config[model_name]
    model.fit(X, Y)
    invert_X = invert_sensitive(X, sensitive_param, dataset)
    invert_model = invert_model_config[model_name]
    invert_model.fit(invert_X, Y)
    total_init = 0
    init = set()
    init_list = []
    init_index = []

    for i in range(len(X)):
        y_orig = model.predict(np.array([X[i]]))
        y_new = invert_model.predict(np.array([X[i]]))
        if y_orig != y_new:
            total_init += 1
            if check_for_error_condition(model, data_config[dataset], X[i], sensitive_param):
                temp = X[i].astype('int').tolist()
                #temp = temp[:sensitive_param - 1] + temp[sensitive_param:]
                if tuple(temp) not in init:
                    init.add(tuple(temp))
                    init_list.append(temp)
                    init_index.append(i)

    # shap
    shap.initjs()
    if model_name == 'LogisticRegression':
        explainer = shap.LinearExplainer(model, X)
        idi = np.array(init_list)
        shap_values = explainer.shap_values(idi)
    elif model_name == 'DecisionTreeClassifier':
        explainer = shap.TreeExplainer(model, X)
        idi = np.array(init_list)
        shap_values = explainer.shap_values(idi)[0]
    elif model_name == 'SVC':
        explainer = shap.KernelExplainer(model.predict_proba, X[0:100], link="logit")
        idi = np.array(init_list)
        shap_values = explainer.shap_values(idi[0: min(len(idi),200)])[0]
    elif model_name in ['MLPRegressor', 'MLPClassifier']:
        explainer = shap.KernelExplainer(model.predict, X[0:100])
        idi = np.array(init_list)
        shap_values = explainer.shap_values(idi[0: min(len(idi),200)])

    # shap_values DBSCAN 每个聚类簇挑选10个
    cluster_model = DBSCAN(eps=0.09, min_samples=10)
    cluster_labels = cluster_model.fit(shap_values).labels_
    print('num of labels', np.max(cluster_labels)-np.min(cluster_labels)+1)
    index = []
    for label_ in range(np.min(cluster_labels), np.max(cluster_labels)+1):
        num = 0
        for i in range(len(cluster_labels)):
            if label_ == cluster_labels[i]:
                index.append(i)
                num+=1
                if num == int(100/(np.max(cluster_labels)-np.min(cluster_labels)+1)):
                    break
    for i in range(len(cluster_labels)):
        if len(index) >= 100:
            break
        if i not in index:
            index.append(i)

    # storing the fairness testing result
    idi = idi[index].tolist()
    for i in range(len(X)):
        if len(idi) >= 100:
            break
        if i not in init_index:
            idi.append(X[i])
    for i in range(len(X)):
        if len(init_list) >= 100:
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
    np.save('../results/' + dataset + '/' + str(sensitive_param) + '/'+model_name+'_init_samples.npy', np.array(init_list))
    np.save('../results/' + dataset + '/' + str(sensitive_param) + '/'+model_name+'_init_index.npy', np.array(init_index))
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/'+model_name+'_cluster_init_samples.npy', np.array(idi))

    # print the overview information of result
    print("Total Inputs are " + str(total_init))
    print("Total discriminatory inputs- " + str(len(init_list)))
    print("Total init samples- " + str(len(idi)))


def main(argv=None):
    dnn_init(dataset=FLAGS.dataset,
             sensitive_param=FLAGS.sens_param,
             max_iter=FLAGS.max_iter,
             model_name=FLAGS.model_name)

# census: 1 age, 8 race, 9 sex
# bank: 1 age
# compas: 1 sex, 2 age, 3 race


if __name__ == '__main__':
    flags.DEFINE_string("dataset", "compas", "the name of dataset")
    flags.DEFINE_integer('sens_param', 3, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_integer('max_iter', 300, 'maximum iteration of global perturbation')
    flags.DEFINE_string('model_name', 'SVC', 'ML Models')
    # LogisticRegression, SVC, DecisionTreeClassifier, MLPClassifier

    tf.app.run()
