import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import copy
from fairness_data.census import census_data
from fairness_data.bank import bank_data
from fairness_data.compas import compas_data
from fairness_utils.config import census, bank, compas
from fairness_utils.performance import accuracy, precision, recall, f1_score
import joblib
import tensorflow as tf
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fairness_utils.utils import getRandomIndex


def check_for_error_condition(model, conf, t, sens):
    t = np.array(t).astype('int')
    label = model.predict(np.array([t]))
    # check for all the possible values of sensitive feature
    tnew = copy.deepcopy(t)
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != t[sens-1]:
            tnew[sens-1] = val
            label_new = model.predict(np.array([tnew]))
            if label_new != label:
                return val
    return t[sens - 1]


def retraining_testing(dataset, sensitive_param, max_iter, approach_name, model_name):
    model_config = {
        "LogisticRegression": LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=max_iter),
        "SVC": SVC(kernel='rbf', probability=True, max_iter=max_iter),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "MLPClassifier": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(3,), max_iter=max_iter, learning_rate='invscaling', random_state=0)),
    }
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

    X, Y, input_shape, nb_classes = data[dataset]()
    Y = Y[:, 1]
    train_num = int(len(X) * 0.6)
    train_test_index = getRandomIndex(len(X), len(X))
    X_train = X[train_test_index[:train_num]]
    Y_train = Y[train_test_index[:train_num]]
    X_test = X[train_test_index[train_num:]]
    Y_test = Y[train_test_index[train_num:]]
    original = np.load('../results/' + approach_name + '/' + model_name + '_' + dataset + '_' + str(sensitive_param) + '_original_idi.npy')
    w_I = np.load('../results/' + approach_name + '/' + model_name + '_' + dataset + '_' + str(sensitive_param) + '_w_I_idi.npy')
    w_I_D = np.load('../results/' + approach_name + '/' + model_name + '_' + dataset + '_' + str(sensitive_param) + '_w_I_D_idi.npy')
    all_disc_inputs = np.array([original, w_I_D])

    # performance
    model_copy = model_config[model_name]
    model_copy.fit(X_train, Y_train)
    y_pred = np.array(model_copy.predict(X_test))
    print('F1-Score: ' + str(f1_score(Y_test, y_pred)))
    print('==================================')
    # retraining & testing
    train_disc_input = all_disc_inputs
    test_disc_input = set()
    for i in range(len(all_disc_inputs)):
        for disc_input in all_disc_inputs[i]:
            test_disc_input.add(tuple(disc_input))
    test_disc_input = np.array([list(x) for x in list(test_disc_input)])

    for train_idi, inputs_way in zip(train_disc_input, ["original", "w_I_D"]):
        model_copy = model_config[model_name]
        X_train_copy = np.concatenate([X_train, train_idi[:, :-1]])
        Y_train_copy = np.concatenate([Y_train, train_idi[:, -1]])
        model_copy.fit(X_train_copy, Y_train_copy)
        # retraining performance
        y_pred = np.array(model_copy.predict(X_test))
        print(inputs_way)
        print('Retraining F1-Score: ' + str(f1_score(Y_test, y_pred)))
        total_idi_num = 0
        for inp in test_disc_input:
            inp = inp[:-1]
            result = check_for_error_condition(model_copy, data_config[dataset], inp, sensitive_param)
            if result != int(inp[sensitive_param - 1]):
                total_idi_num += 1
        print(str(total_idi_num) + '/' + str(len(test_disc_input)))
        print('==================================')


def main(argv=None):
    retraining_testing(dataset=FLAGS.dataset,
                       sensitive_param=FLAGS.sensitive_param,
                       max_iter=FLAGS.max_iter,
                       approach_name=FLAGS.approach_name,
                       model_name=FLAGS.model_name)


if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_integer('sensitive_param', 1, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_integer('max_iter', 10, 'maximum iteration of global perturbation')
    flags.DEFINE_string('approach_name', 'AEQ', 'approach')
    flags.DEFINE_string('model_name', 'LogisticRegression', 'models')
    tf.app.run()