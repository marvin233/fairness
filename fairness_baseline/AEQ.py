import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import copy
import random
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.platform import flags
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from scipy.optimize import basinhopping
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fairness_model.model import NN
from fairness_data.census import census_data
from fairness_data.compas import compas_data
from fairness_data.bank import bank_data
from fairness_utils.config import census, bank, compas
from fairness_utils.utils import getRandomIndex
from fairness_utils.utils import majority_voting


# global variable
FLAGS = flags.FLAGS
idi_label = -1


class Local_Perturbation(object):
    def __init__(self, model, conf, sensitive_param, param_probability, param_probability_change_size,
                 direction_probability, direction_probability_change_size, step_size):
        self.model = model
        self.conf = conf
        self.sensitive_param = sensitive_param
        self.param_probability = param_probability
        self.param_probability_change_size = param_probability_change_size
        self.direction_probability = direction_probability
        self.direction_probability_change_size = direction_probability_change_size
        self.step_size = step_size

    def __call__(self, x):
        # randomly choose the feature for perturbation
        param_choice = np.random.choice(range(self.conf.params) , p=self.param_probability)

        # randomly choose the direction for perturbation
        perturbation_options = [-1, 1]
        direction_choice = np.random.choice(perturbation_options, p=[self.direction_probability[param_choice],
                                                                     (1 - self.direction_probability[param_choice])])
        if (x[param_choice] == self.conf.input_bounds[param_choice][0]) or (x[param_choice] == self.conf.input_bounds[param_choice][1]):
            direction_choice = np.random.choice(perturbation_options)

        # perturbation
        x[param_choice] = x[param_choice] + (direction_choice * self.step_size)

        # clip the generating instance with each feature to make sure it is valid
        x[param_choice] = max(self.conf.input_bounds[param_choice][0], x[param_choice])
        x[param_choice] = min(self.conf.input_bounds[param_choice][1], x[param_choice])

        # check whether the test case is an individual discriminatory instance
        ei = check_for_error_condition(self.model, self.conf, x, self.sensitive_param)

        # update the probabilities of directions
        if (ei != int(x[self.sensitive_param - 1]) and direction_choice == -1) or (not (ei != int(x[self.sensitive_param - 1])) and direction_choice == 1):
            self.direction_probability[param_choice] = min(self.direction_probability[param_choice] +
                                                      (self.direction_probability_change_size * self.step_size), 1)
        elif (not (ei != int(x[self.sensitive_param - 1])) and direction_choice == -1) or (ei != int(x[self.sensitive_param - 1]) and direction_choice == 1):
            self.direction_probability[param_choice] = max(self.direction_probability[param_choice] -
                                                      (self.direction_probability_change_size * self.step_size), 0)

        # update the probabilities of features
        if ei != int(x[self.sensitive_param - 1]):
            self.param_probability[param_choice] = self.param_probability[param_choice] + self.param_probability_change_size
            self.normalise_probability()
        else:
            self.param_probability[param_choice] = max(self.param_probability[param_choice] - self.param_probability_change_size, 0)
            self.normalise_probability()

        return x

    def normalise_probability(self):
        """
        Normalize the probability
        :return: probability
        """
        probability_sum = 0.0
        for prob in self.param_probability:
            probability_sum = probability_sum + prob

        for i in range(self.conf.params):
            self.param_probability[i] = float(self.param_probability[i]) / float(probability_sum)


class Global_Discovery(object):
    def __init__(self, conf):
        """
        Initial function of global perturbation
        :param conf: the configuration of dataset
        """
        self.conf = conf

    def __call__(self, x):
        """
        Global perturbation
        :param x: input instance for local perturbation
        :return: new potential individual discriminatory instance
        """
        # clip the generating instance with each feature to make sure it is valid
        for i in range(self.conf.params):
            x[i] = random.randint(self.conf.input_bounds[i][0], self.conf.input_bounds[i][1])
        return x


def check_for_error_condition(model, conf, t, sens):
    t = np.array(t).astype('int')
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


def generate_data(model, dataset, data_config, sensitive_param, w_I_D, w_I, max_global, X, max_local, model_name, param_probability, param_probability_change_size, direction_probability, direction_probability_change_size, step_size, label_models, initial_input, minimizer):
    global idi_label
    # store the result of fairness testing
    global_disc_inputs = set()
    local_disc_inputs = set()

    def evaluate_local(inp):
        global idi_label
        result = check_for_error_condition(model, data_config[dataset], inp, sensitive_param)
        if result != int(inp[sensitive_param - 1]):
            temp = copy.deepcopy(inp.astype('int').tolist())
            temp[sensitive_param-1] = result
            temp = np.array(list(temp) + list(idi_label)).astype('int')
            local_disc_inputs.add(tuple(temp))
        return not result

    global_discovery = Global_Discovery(data_config[dataset])
    local_perturbation = Local_Perturbation(model, data_config[dataset], sensitive_param, param_probability, param_probability_change_size, direction_probability, direction_probability_change_size, step_size)

    if w_I_D:
        init_sample = np.load('../results/' + dataset + '/' + str(sensitive_param) + '/' + model_name + '_w_I_D_init_samples.npy')
        length = min(max_global, len(init_sample))
    elif w_I:
        init_sample = np.load('../results/' + dataset + '/' + str(sensitive_param) + '/' + model_name + '_w_I_init_samples.npy')
        length = min(max_global, len(init_sample))
    else:
        length = min(max_global, len(X))

    value_list = []
    for i in range(length):
        # global generation
        if w_I_D:
            inp = list(init_sample[i])
        elif w_I:
            inp = list(init_sample[i])
        else:
            inp = global_discovery.__call__(initial_input)

        result = check_for_error_condition(model, data_config[dataset], inp, sensitive_param)
        # if get an individual discriminatory instance
        if result != inp[sensitive_param - 1]:
            idi_label = majority_voting(label_models, np.array([inp]))
            temp = copy.deepcopy(inp)
            temp[sensitive_param-1] = result
            temp = np.array(list(temp) + list(idi_label)).astype('int')
            global_disc_inputs.add(tuple(temp))
            value_list.append([inp[sensitive_param - 1], result])
            # local generation
            basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer, niter=max_local)

    disc_inputs = global_disc_inputs.union(local_disc_inputs)
    return global_disc_inputs, local_disc_inputs, disc_inputs


def aequitas(dataset, sensitive_param, max_global, max_local, max_iter, step_size, model_name):
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
        "LogisticRegression": LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=max_iter),
        "SVC": SVC(kernel='rbf', probability=True, max_iter=max_iter),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "MLPClassifier": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(3,), max_iter=max_iter,learning_rate='invscaling', random_state=0)),
    }
    initial_input_dic = {
        'census': [4, 0, 9, 1, 0, 3, 2, 0, 1, 0, 0, 50, 0],
        'bank': [4, 3, 2, 3, 0, 0, 1, 0, 2, 23, 1, 22, 2, 0, 1, 2],
        'compas': [1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 4]
    }
    minimizer = {
        "method": "L-BFGS-B"
    }

    # initial
    initial_input = initial_input_dic[dataset]
    params = data_config[dataset].params
    # hyper-parameters for initial probabilities of directions
    init_prob = 0.5
    direction_probability = [init_prob] * params
    direction_probability_change_size = 0.001
    # hyper-parameters for features
    param_probability = [1.0 / params] * params
    param_probability_change_size = 0.001

    # prepare the training and testing data and model
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
        global_disc_inputs, local_disc_inputs, disc_inputs = generate_data(model, dataset, data_config, sensitive_param, w_I_D, w_I, max_global, X, max_local, model_name, param_probability, param_probability_change_size, direction_probability, direction_probability_change_size, step_size, label_models, initial_input, minimizer)

        if w_I_D:
            inputs_way = 'w_I_D'
        elif w_I:
            inputs_way = 'w_I'
        else:
            inputs_way = 'original'
        np.save('../results/AEQU/'+model_name+'_'+dataset+'_'+str(sensitive_param)+'_'+inputs_way+'_idi.npy', np.array(list(disc_inputs)))
        print(inputs_way)
        print('global', len(global_disc_inputs))
        print('local', len(local_disc_inputs))
        print('disc_inputs', len(disc_inputs))


def main(argv=None):
    aequitas(dataset=FLAGS.dataset,
             sensitive_param=FLAGS.sensitive_param,
             max_global=FLAGS.max_global,
             max_local=FLAGS.max_local,
             max_iter=FLAGS.max_iter,
             step_size=FLAGS.step_size,
             model_name=FLAGS.model_name)


if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_integer('sensitive_param', 1, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_integer('max_global', 100, 'number of maximum samples for global search')
    flags.DEFINE_integer('max_local', 100, 'number of maximum samples for local search')
    flags.DEFINE_integer('max_iter', 30, 'maximum iteration of global perturbation')
    flags.DEFINE_float('step_size', 1.0, 'step size for perturbation')
    flags.DEFINE_string('model_name', 'LogisticRegression', 'models config')
    tf.app.run()