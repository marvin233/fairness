from tensorflow import keras
import numpy as np
import tensorflow as tf
import sys, os
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import copy
import shap
from tensorflow.python.platform import flags
from scipy.optimize import basinhopping
from fairness_data.census import census_data
from fairness_data.bank import bank_data
from fairness_data.compas import compas_data
from fairness_model.model import dnn
from fairness_utils.utils_tf import model_prediction, model_argmax
from fairness_utils.config import census, bank, compas
from fairness_utils.utils import cluster
from fairness_utils.utils_tf import gradient_graph
from fairness_utils.utils_tf import model_train
from fairness_utils.utils import getRandomIndex
from fairness_utils.utils import majority_voting
import joblib


# global variable
FLAGS = flags.FLAGS
perturbation_size = 1
idi_label = -1


def check_for_error_condition(conf, sess, x, preds, t, sens):
    t = t.astype('int')
    # model_prediction
    label = model_argmax(sess, x, preds, np.array([t]))
    # check for all the possible values of sensitive feature
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != t[sens-1]:
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            label_new = model_argmax(sess, x, preds, np.array([tnew]))
            if label_new != label:
                return val
    return t[sens - 1]


def seed_test_input(clusters, limit):
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
            if len(rows) == limit:
                break
        i += 1
    return np.array(rows)


def clip(input, conf):
    for i in range(len(input)):
        input[i] = max(input[i], conf.input_bounds[i][0])
        input[i] = min(input[i], conf.input_bounds[i][1])
    return input


class Local_Perturbation(object):
    """
    The  implementation of local perturbation
    """

    def __init__(self, sess, grad, x, n_value, sens, input_shape, conf):
        """
        Initial function of local perturbation
        :param sess: TF session
        :param grad: the gradient graph
        :param x: input placeholder
        :param n_value: the discriminatory value of sensitive feature
        :param sens_param: the index of sensitive feature
        :param input_shape: the shape of dataset
        :param conf: the configuration of dataset
        """
        self.sess = sess
        self.grad = grad
        self.x = x
        self.n_value = n_value
        self.input_shape = input_shape
        self.sens = sens
        self.conf = conf

    def __call__(self, x):
        """
        Local perturbation
        :param x: input instance for local perturbation
        :return: new potential individual discriminatory instance
        """

        # perturbation
        s = np.random.choice([1.0, -1.0]) * perturbation_size

        n_x = x.copy()
        n_x[self.sens - 1] = self.n_value

        # compute the gradients of an individual discriminatory instance pairs
        ind_grad = self.sess.run(self.grad, feed_dict={self.x:np.array([x])})
        n_ind_grad = self.sess.run(self.grad, feed_dict={self.x:np.array([n_x])})

        ### patch 
        #if np.zeros(self.input_shape).tolist() == ind_grad[0].tolist() and np.zeros(self.input_shape).tolist() == \
        #        n_ind_grad[0].tolist():
        #    probs = 1.0 / (self.input_shape-1) * np.ones(self.input_shape)
        #    probs[self.sens - 1] = 0
        if 0 in (abs(ind_grad[0]) + abs(n_ind_grad[0])):
            grad_sum = 1 / (abs(ind_grad[0]) + abs(n_ind_grad[0]) + 0.000001)
        else:
            # nomalize the reciprocal of gradients (prefer the low impactful feature)
            grad_sum = 1.0 / (abs(ind_grad[0]) + abs(n_ind_grad[0]))
        grad_sum[self.sens - 1] = 0
        probs = grad_sum / np.sum(grad_sum)
        probs = probs/probs.sum()

        # randomly choose the feature for local perturbation
        index = np.random.choice(range(self.input_shape) , p=probs)
        local_cal_grad = np.zeros(self.input_shape)
        local_cal_grad[index] = 1.0

        x = clip(x + s * local_cal_grad, self.conf).astype("int")

        return x


def generate_data(sess, x, preds, dataset, clusters, data_config, sensitive_param, w_I_D, w_I, max_global, X, max_iter, max_local, input_shape, label_models):
    global idi_label
    # construct the gradient graph
    grad_0 = gradient_graph(x, preds)
    # store the result of fairness testing
    global_disc_inputs = set()
    local_disc_inputs = set()

    def evaluate_local(inp):
        global idi_label
        result = check_for_error_condition(data_config[dataset], sess, x, preds, inp, sensitive_param)
        if result != int(inp[sensitive_param - 1]):
            temp = copy.deepcopy(inp.astype('int').tolist())
            temp[sensitive_param - 1] = result
            temp = np.array(list(temp) + list(idi_label)).astype('int')
            local_disc_inputs.add(tuple(temp))
            return False
        return True

    if w_I_D:
        init_sample = np.load('../results/' + dataset + '/' + str(sensitive_param) + '/NN_w_I_D_init_samples.npy')
        inputs = init_sample[:min(max_global, len(init_sample))]
    elif w_I:
        init_sample = np.load('../results/' + dataset + '/' + str(sensitive_param) + '/NN_w_I_init_samples.npy')
        inputs = init_sample[:min(max_global, len(init_sample))]
    else:
        inputs = seed_test_input(clusters, min(max_global, len(X)))
    for num in range(len(inputs)):
        if w_I_D:
            sample = np.array([inputs[num]])
        elif w_I:
            sample = np.array([inputs[num]])
        else:
            index = inputs[num]
            sample = X[index:index + 1]
        # start global perturbation
        for iter in range(max_iter + 1):
            probs = model_prediction(sess, x, preds, sample)[0]
            label = model_argmax(sess, x, preds, np.array(sample))
            prob = probs[label]
            max_diff = 0
            n_value = -1
            # search the instance with maximum probability difference for global perturbation
            for i in range(data_config[dataset].input_bounds[sensitive_param - 1][0], data_config[dataset].input_bounds[sensitive_param - 1][1] + 1):
                if i != sample[0][sensitive_param - 1]:
                    n_sample = sample.copy()
                    n_sample[0][sensitive_param - 1] = i
                    n_probs = model_prediction(sess, x, preds, n_sample)[0]
                    n_label = model_argmax(sess, x, preds, np.array(n_sample))
                    n_prob = n_probs[n_label]
                    if label != n_label:
                        n_value = i
                        break
                    else:
                        prob_diff = abs(prob - n_prob)
                        if prob_diff > max_diff:
                            max_diff = prob_diff
                            n_value = i
            # if get an individual discriminatory instance
            if label != n_label:
                idi_label = np.array([label])
                temp = copy.deepcopy(sample[0].astype('int').tolist())
                temp = np.array(list(temp) + list(idi_label)).astype('int')
                global_disc_inputs.add(tuple(temp))
                # start local perturbation
                minimizer = {"method": "L-BFGS-B"}
                local_perturbation = Local_Perturbation(sess, grad_0, x, n_value, sensitive_param, input_shape[1], data_config[dataset])
                basinhopping(evaluate_local, sample, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer, niter=max_local)
                break
            n_sample[0][sensitive_param - 1] = n_value
            if iter == max_iter:
                break
            # global perturbation
            s_grad = sess.run(tf.sign(grad_0), feed_dict={x: sample})
            n_grad = sess.run(tf.sign(grad_0), feed_dict={x: n_sample})
            # find the feature with same impact
            if np.zeros(data_config[dataset].params).tolist() == s_grad[0].tolist():
                g_diff = n_grad[0]
            elif np.zeros(data_config[dataset].params).tolist() == n_grad[0].tolist():
                g_diff = s_grad[0]
            else:
                g_diff = np.array(s_grad[0] == n_grad[0], dtype=float)
            g_diff[sensitive_param - 1] = 0
            if np.zeros(input_shape[1]).tolist() == g_diff.tolist():
                index = np.random.randint(len(g_diff) - 1)
                if index > sensitive_param - 2:
                    index = index + 1
                g_diff[index] = 1.0
            cal_grad = s_grad * g_diff
            sample[0] = clip(sample[0] + perturbation_size * cal_grad[0], data_config[dataset]).astype("int")
    disc_inputs = global_disc_inputs.union(local_disc_inputs)
    return global_disc_inputs, local_disc_inputs, disc_inputs


def adf(dataset, sensitive_param, cluster_num, max_global, max_local, max_iter):
    global idi_label
    data = {
        "census": census_data,
        "bank": bank_data,
        'compas': compas_data
    }
    data_config = {
        "census": census,
        "bank": bank,
        'compas': compas
    }
    clf = cluster(dataset, cluster_num)
    clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]

    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    train_num = int(len(X) * 0.6)
    train_test_index = getRandomIndex(len(X), len(X))
    X_train = X[train_test_index[:train_num]]
    Y_train = Y[train_test_index[:train_num]]

    # model
    tf.set_random_seed(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    preds = dnn(input_shape, nb_classes)(x)
    # training parameters
    train_params = {
        'nb_epochs': 30,
        'batch_size': 128,
        'learning_rate': 0.01
    }
    # training procedure
    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2021, 8, 16])
    model_train(sess, x, y, preds, X_train, Y_train, args=train_params, rng=rng, save=False)

    # idi-label model
    label_models = joblib.load("../models/ensemble_models/" + dataset + "_ensemble.pkl")

    for [w_I_D, w_I] in [[False, False], [False, True], [True, False]]:

        global_disc_inputs, local_disc_inputs, disc_inputs = generate_data(sess, x, preds, dataset, clusters, data_config, sensitive_param, w_I_D, w_I, max_global, X, max_iter, max_local, input_shape, label_models)
        if w_I_D:
            inputs_way = 'w_I_D'
        elif w_I:
            inputs_way = 'w_I'
        else:
            inputs_way = 'original'
        np.save('../results/ADF/'+dataset+'_'+str(sensitive_param)+'_'+inputs_way+'_idi.npy', np.array(list(disc_inputs)))
        print('global', len(global_disc_inputs))
        print('local', len(local_disc_inputs))
        print('disc_inputs', len(disc_inputs))
        print('==================================')

def main(argv=None):
    adf(dataset=FLAGS.dataset,
        sensitive_param=FLAGS.sensitive_param,
        cluster_num=FLAGS.cluster_num,
        max_global=FLAGS.max_global,
        max_local=FLAGS.max_local,
        max_iter=FLAGS.max_iter)


if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_integer('sensitive_param', 1, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_string('model_path', '../models/', 'the path for testing model')
    flags.DEFINE_integer('cluster_num', 4, 'the number of clusters to form as well as the number of centroids to generate')
    flags.DEFINE_integer('max_global', 100, 'maximum number of samples for global search')
    flags.DEFINE_integer('max_local', 100, 'maximum number of samples for local search')
    flags.DEFINE_integer('max_iter', 10, 'maximum iteration of global perturbation')

    tf.app.run()
