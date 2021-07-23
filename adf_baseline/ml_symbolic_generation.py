import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
from sklearn.tree import DecisionTreeClassifier
if sys.version_info.major==2:
    from Queue import PriorityQueue
else:
    from queue import PriorityQueue
from z3 import *
import os
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from adf_baseline.lime import lime_tabular
from adf_model.tutorial_models import dnn
from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_data.execution import execution_data
from adf_data.compas import compas_data
from adf_utils.config import census, credit, bank, execution, compas
from adf_utils.utils_tf import model_argmax
from adf_tutorial.utils import cluster
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from adf_utils.performance import accuracy, precision, recall, f1_score
FLAGS = flags.FLAGS


def seed_test_input(dataset, cluster_num, limit):
    """
    Select the seed inputs for fairness testing
    :param dataset: the name of dataset
    :param clusters: the results of K-means clustering
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
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
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param t: test case
    :param sens: the index of sensitive feature
    :return: whether it is an individual discriminatory instance
    """
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


def global_solve(path_constraint, arguments, t, conf):
    """
    Solve the constraint for global generation
    :param path_constraint: the constraint of path
    :param arguments: the name of features in path_constraint
    :param t: test case
    :param conf: the configuration of dataset
    :return: new instance through global generation
    """
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
    """
    Solve the constraint for local generation
    :param path_constraint: the constraint of path
    :param arguments: the name of features in path_constraint
    :param t: test case
    :param index: the index of constraint for local generation
    :param conf: the configuration of dataset
    :return: new instance through global generation
    """
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
    """
    The average confidence (probability) of path
    :param path_constraint: the constraint of path
    :return: the average confidence
    """
    r = np.mean(np.array(path_constraint)[:,3].astype(float))
    return r


def gen_arguments(conf):
    """
    Generate the argument for all the features
    :param conf: the configuration of dataset
    :return: a sequence of arguments
    """
    arguments = []
    for i in range(conf.params):
        arguments.append(Int(conf.feature_name[i]))
    return arguments


def symbolic_generation(dataset, sensitive_param, limit, max_iter, cluster_num, new_input, cluster_input, model_name):
    data = {"census":census_data, "credit":credit_data, "bank":bank_data, "execution":execution_data, "compas":compas_data}
    data_config = {"census":census, "credit":credit, "bank":bank, "execution":execution, "compas":compas}
    model_config = {
        "LogisticRegression": LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=max_iter),
        "SVC": SVC(kernel='rbf', probability=True, max_iter=max_iter),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "MLPRegressor": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(3,), activation='logistic', max_iter=max_iter, learning_rate='invscaling', random_state=0)),
        "MLPClassifier": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(3,), max_iter=max_iter, learning_rate='invscaling', random_state=0)),
    }
    # the rank for priority queue, rank1 is for seed inputs, rank2 for local, rank3 for global
    rank1 = 5
    rank2 = 1
    rank3 = 10
    T1 = 0.3

    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    Y = Y[:,1]
    arguments = gen_arguments(data_config[dataset])
    model = model_config[model_name]
    model.fit(X, Y)

    # store the result of fairness testing
    global_disc_inputs = set()
    global_disc_inputs_list = []
    global_init_list = []
    global_miss_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    tot_inputs = set()
    q = PriorityQueue()  # low push first

    if cluster_input:
        init_index = np.load('../results/' + dataset + '/' + str(sensitive_param) + '/'+model_name+'_cluster_init_samples.npy')
        inputs = init_index[:min(limit, len(init_index))]
        print(len(inputs))
        if len(inputs)<100:
            exit()
        for inp in inputs[::-1]:
            q.put((rank1, inp.tolist()))
    elif new_input:
        init_index = np.load('../results/'+dataset+'/'+ str(sensitive_param) + '/'+model_name+'_init_index.npy')
        inputs = init_index[:min(limit, len(init_index))]
        print(len(inputs))
        for inp in inputs[::-1]:
            q.put((rank1, X[inp].tolist()))
    else:
        # select the seed input for fairness testing
        inputs = seed_test_input(dataset, cluster_num, limit)
        for inp in inputs[::-1]:
            q.put((rank1, X[inp].tolist()))

    visited_path = []
    l_count = 0
    g_count = 0
    found_number = 0
    while len(tot_inputs) < limit and q.qsize() != 0:
        t = q.get()
        t_rank = t[0]
        t = np.array(t[1])
        found = check_for_error_condition(model, data_config[dataset], t, sensitive_param)
        if found:
            found_number += 1

        ###
        p = getPath(X, model, t, data_config[dataset])
        temp = copy.deepcopy(t.tolist())
        temp = temp[:sensitive_param - 1] + temp[sensitive_param:]

        tot_inputs.add(tuple(temp))
        
        ##marvin adds##
        if not found:
            global_miss_list.append(temp)
        ###############
        
        if found:
            if (tuple(temp) not in global_disc_inputs) and (tuple(temp) not in local_disc_inputs):
                if t_rank > 2:
                    global_disc_inputs.add(tuple(temp))
                    global_disc_inputs_list.append(temp)
                else:
                    local_disc_inputs.add(tuple(temp))
                    local_disc_inputs_list.append(temp)
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
                    q.put((rank3-r, input))

            prefix_pred = prefix_pred + [c]

    # create the folder for storing the fairness testing result
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    if not os.path.exists('../results/'+ dataset + '/'+ str(sensitive_param) + '/'):
        os.makedirs('../results/' + dataset + '/'+ str(sensitive_param) + '/')

    if cluster_input:
        inputs_way = 'cluster'
    elif new_input:
        inputs_way = 'new'
    else:
        inputs_way = 'ori'
    # storing the fairness testing result
    np.save('../results/' + dataset + '/' + str(sensitive_param) + '/sym_' + model_name + '_global_' + inputs_way + '.npy', np.array(global_disc_inputs))
    np.save('../results/' + dataset + '/' + str(sensitive_param) + '/sym_' + model_name + '_local_' + inputs_way + '.npy', np.array(local_disc_inputs))

    # print the overview information of result
    print("Total Inputs are " + str(len(tot_inputs)))
    print("Total discriminatory inputs of global search - " + str(len(global_disc_inputs)))
    print("Total discriminatory inputs of local search - " + str(len(local_disc_inputs)))
    print(dataset, sensitive_param)


def main(argv=None):
    symbolic_generation(dataset=FLAGS.dataset,
                        sensitive_param=FLAGS.sens_param,
                        limit=FLAGS.sample_limit,
                        max_iter=FLAGS.max_iter,
                        cluster_num=FLAGS.cluster_num,
                        new_input=FLAGS.new_input,
                        cluster_input=FLAGS.cluster_input,
                        model_name=FLAGS.model_name)

# census: 1 age, 8 race, 9 sex
# bank: 1 age
# compas: 1 sex, 2 age, 3 race
if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'census', 'the name of dataset')
    flags.DEFINE_integer('sens_param', 9, 'sensitive index, index start from 1, 9 for gender, 8 for race.')
    flags.DEFINE_integer('sample_limit', 100, 'number of samples to search')
    flags.DEFINE_integer('max_iter', 300, 'maximum iteration of global perturbation')
    flags.DEFINE_integer('cluster_num', 4, 'the number of clusters to form as well as the number of centroids to generate')
    flags.DEFINE_boolean('new_input', False, 'our new input approach')
    flags.DEFINE_boolean('cluster_input', False, 'shap & cluster')
    flags.DEFINE_string('model_name', 'MLPClassifier', 'ML Models')
    # LogisticRegression, SVC, DecisionTreeClassifier, MLPClassifier
    tf.app.run()