import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
from z3 import *
import os
import copy
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.platform import flags
if sys.version_info.major==2:
    from Queue import PriorityQueue
else:
    from queue import PriorityQueue
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from fairness_baseline.lime import lime_tabular
from fairness_data.census import census_data
from fairness_data.bank import bank_data
from fairness_data.compas import compas_data
from fairness_utils.config import census, bank, compas
from fairness_utils.utils import cluster
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fairness_utils.utils import getRandomIndex
from fairness_utils.utils import majority_voting
from fairness_model.model import NN


# global variable
FLAGS = flags.FLAGS
idi_label = -1


def seed_test_input(dataset, cluster_num, limit):
    # build the clustering model
    clf = cluster(dataset, cluster_num)
    clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]  # len(clusters[0][0])==32561

    i = 0
    rows = []
    max_size = max([len(c[0]) for c in clusters])
    while i < max_size:
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c[0]):
                continue
            row = c[0][i]
            rows.append(row)
        i += 1
    return np.array(rows)


def getPath(X, model, input, conf):
    # use the original implementation of LIME
    explainer = lime_tabular.LimeTabularExplainer(X,
                                                  feature_names=conf.feature_name, class_names=conf.class_name, categorical_features=conf.categorical_features,
                                                  discretize_continuous=True)
    g_data = explainer.generate_instance(input, num_samples=5000)
    g_labels = model.predict(g_data)
    # build the interpretable tree
    tree = DecisionTreeClassifier(random_state=2019) #min_samples_split=0.05, min_samples_leaf =0.01
    tree.fit(g_data, g_labels)

    # get the path for decision
    path_index = tree.decision_path(np.array([input])).indices
    path = []
    for i in range(len(path_index)):
        node = path_index[i]
        i = i + 1
        f = tree.tree_.feature[node]
        if f != -2:
            left_count = tree.tree_.n_node_samples[tree.tree_.children_left[node]]
            right_count = tree.tree_.n_node_samples[tree.tree_.children_right[node]]
            left_confidence = 1.0 * left_count / (left_count + right_count)
            right_confidence = 1.0 - left_confidence
            if tree.tree_.children_left[node] == path_index[i]:
                path.append([f, "<=", tree.tree_.threshold[node], left_confidence])
            else:
                path.append([f, ">", tree.tree_.threshold[node], right_confidence])
    return path


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
                return val
    return t[sens - 1]


def global_solve(path_constraint, arguments, t, conf):
    s = Solver()
    for c in path_constraint:
        s.add(arguments[c[0]] >= conf.input_bounds[c[0]][0])
        s.add(arguments[c[0]] <= conf.input_bounds[c[0]][1])
        if c[1] == "<=":
            s.add(arguments[c[0]] <= c[2])
        else:
            s.add(arguments[c[0]] > c[2])

    if s.check() == sat:
        m = s.model()
    else:
        return None

    tnew = copy.deepcopy(t)
    for i in range(len(arguments)):
        if m[arguments[i]] == None:
            continue
        else:
            tnew[i] = int(m[arguments[i]].as_long())
    return tnew.astype('int').tolist()


def local_solve(path_constraint, arguments, t, index, conf):
    c = path_constraint[index]
    s = Solver()
    s.add(arguments[c[0]] >= conf.input_bounds[c[0]][0])
    s.add(arguments[c[0]] <= conf.input_bounds[c[0]][1])
    for i in range(len(path_constraint)):
        if path_constraint[i][0] == c[0]:
            if path_constraint[i][1] == "<=":
                s.add(arguments[path_constraint[i][0]] <= path_constraint[i][2])
            else:
                s.add(arguments[path_constraint[i][0]] > path_constraint[i][2])

    if s.check() == sat:
        m = s.model()
    else:
        return None

    tnew = copy.deepcopy(t)
    tnew[c[0]] = int(m[arguments[c[0]]].as_long())
    return tnew.astype('int').tolist()


def average_confidence(path_constraint):
    r = np.mean(np.array(path_constraint)[:,3].astype(float))
    return r


def gen_arguments(conf):
    arguments = []
    for i in range(conf.params):
        arguments.append(Int(conf.feature_name[i]))
    return arguments


def generate_data(w_I_D, dataset, sensitive_param, model_name, limit, rank1, w_I, cluster_num, X, model, data_config, arguments, rank2, T1, rank3, label_models):
    global idi_label
    # store the result of fairness testing
    global_disc_inputs = set()
    local_disc_inputs = set()
    tot_inputs = set()
    q = PriorityQueue()  # low push first

    if w_I_D:
        init_sample = np.load('../results/' + dataset + '/' + str(sensitive_param) + '/' + model_name + '_w_I_D_init_samples.npy')
        inputs = init_sample[:min(limit, len(init_sample))]
        for inp in inputs[::-1]:
            q.put((rank1, inp.tolist()))
    elif w_I:
        init_sample = np.load('../results/' + dataset + '/' + str(sensitive_param) + '/' + model_name + '_w_I_init_samples.npy')
        inputs = init_sample[:min(limit, len(init_sample))]
        for inp in inputs[::-1]:
            q.put((rank1, inp.tolist()))
    else:
        # select the seed input for fairness testing
        inputs = seed_test_input(dataset, cluster_num, limit)
        for inp in inputs[::-1]:
            q.put((rank1, X[inp].tolist()))

    visited_path = []
    l_count = 0
    g_count = 0
    while len(tot_inputs) < limit and q.qsize() != 0:
        t = q.get()
        t_rank = t[0]
        t = np.array(t[1])
        result = check_for_error_condition(model, data_config[dataset], t, sensitive_param)
        p = getPath(X, model, t, data_config[dataset])
        temp = copy.deepcopy(t.tolist())
        temp = temp[:sensitive_param - 1] + temp[sensitive_param:]
        tot_inputs.add(tuple(temp))
        if result != int(t[sensitive_param - 1]):
            temp = copy.deepcopy(t.tolist())
            temp[sensitive_param-1] = result
            idi_label = majority_voting(label_models, np.array([temp]))
            temp = np.array(list(temp) + list(idi_label)).astype('int')
            if t_rank > 2:
                global_disc_inputs.add(tuple(temp))
            else:
                local_disc_inputs.add(tuple(temp))
            if len(tot_inputs) == limit:
                break

            # local search
            for i in range(len(p)):
                path_constraint = copy.deepcopy(p)
                c = path_constraint[i]
                if c[0] == sensitive_param - 1:
                    continue
                if c[1] == "<=":
                    c[1] = ">"
                    c[3] = 1.0 - c[3]
                else:
                    c[1] = "<="
                    c[3] = 1.0 - c[3]

                if path_constraint not in visited_path:
                    visited_path.append(path_constraint)
                    input = local_solve(path_constraint, arguments, t, i, data_config[dataset])
                    l_count += 1
                    if input != None:
                        r = average_confidence(path_constraint)
                        q.put((rank2 + r, input))

        # global search
        prefix_pred = []
        for c in p:
            if c[0] == sensitive_param - 1:
                continue
            if c[3] < T1:
                break

            n_c = copy.deepcopy(c)
            if n_c[1] == "<=":
                n_c[1] = ">"
                n_c[3] = 1.0 - c[3]
            else:
                n_c[1] = "<="
                n_c[3] = 1.0 - c[3]
            path_constraint = prefix_pred + [n_c]

            # filter out the path_constraint already solved before
            if path_constraint not in visited_path:
                visited_path.append(path_constraint)
                input = global_solve(path_constraint, arguments, t, data_config[dataset])
                g_count += 1
                if input != None:
                    r = average_confidence(path_constraint)
                    q.put((rank3 - r, input))

            prefix_pred = prefix_pred + [c]

    disc_inputs = global_disc_inputs.union(local_disc_inputs)
    return global_disc_inputs, local_disc_inputs, disc_inputs


def symbolic_generation(dataset, sensitive_param, limit, max_iter, cluster_num, model_name):
    global idi_label
    data = {
        "census": census_data,
        "bank": bank_data,
        "compas": compas_data
    }
    data_config = {
        "census": census,
        "bank": bank,
        "compas": compas
    }
    model_config = {
        "SVC": SVC(kernel='rbf', probability=True, max_iter=max_iter),
        "LogisticRegression": LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=max_iter),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "MLPClassifier": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(3,), max_iter=max_iter, learning_rate='invscaling', random_state=0)),
    }
    # the rank for priority queue, rank1 is for seed inputs, rank2 for local, rank3 for global
    rank1 = 5
    rank2 = 1
    rank3 = 10
    T1 = 0.3
    arguments = gen_arguments(data_config[dataset])
    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    Y = Y[:, 1]
    train_test_index = getRandomIndex(len(X), len(X))
    X = X[train_test_index]
    Y = Y[train_test_index]
    if model_name == "NN":
        model = NN(keras.models.load_model("../models/original_models/" + dataset + ".h5"))
    else:
        model = model_config[model_name]
        model.fit(X, Y)

    # idi-label model
    label_models = joblib.load("../models/ensemble_models/" + dataset + "_ensemble.pkl")

    # generate
    for [w_I_D, w_I] in [[False, False], [False, True], [True, False]]:

        global_disc_inputs, local_disc_inputs, disc_inputs = generate_data(w_I_D, dataset, sensitive_param, model_name, limit, rank1, w_I, cluster_num, X, model, data_config, arguments, rank2, T1, rank3, label_models)
        if w_I_D:
            inputs_way = 'w_I_D'
        elif w_I:
            inputs_way = 'w_I'
        else:
            inputs_way = 'original'
        np.save('../results/SG/'+model_name + '_' + dataset + '_' + str(sensitive_param) + '_' + inputs_way + '_idi.npy', np.array(list(disc_inputs)))
        print('global', len(global_disc_inputs))
        print('local', len(local_disc_inputs))
        print('disc_inputs', len(disc_inputs))


def main(argv=None):
    symbolic_generation(dataset=FLAGS.dataset,
                        sensitive_param=FLAGS.sensitive_param,
                        limit=FLAGS.sample_limit,
                        max_iter=FLAGS.max_iter,
                        cluster_num=FLAGS.cluster_num,
                        model_name=FLAGS.model_name)


if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'census', 'the name of dataset')
    flags.DEFINE_integer('sensitive_param', 1, 'sensitive index, index start from 1, 9 for gender, 8 for race.')
    flags.DEFINE_integer('sample_limit', 100, 'number of samples to search')
    flags.DEFINE_integer('max_iter', 30, 'maximum iteration of global perturbation')
    flags.DEFINE_integer('cluster_num', 4, 'the number of clusters to form as well as the number of centroids to generate')
    flags.DEFINE_string('model_name', 'LogisticRegression', 'models')
    tf.app.run()