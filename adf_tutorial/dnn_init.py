import numpy as np
import pandas as df 
import tensorflow as tf
import os,sys
sys.path.append("../")
import copy
import shap 

from tensorflow.python.platform import flags

from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_argmax
from adf_utils.config import census, credit, bank

FLAGS = flags.FLAGS

def check_for_error_condition(conf, sess, x, preds, t, sens):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param t: test case
    :param sens: the index of sensitive feature
    :return: whether it is an individual discriminatory instance
    """
    t = t.astype('int')
    label = model_argmax(sess, x, preds, np.array([t]))

    # check for all the possible values of sensitive feature
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != t[sens-1]:
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            label_new = model_argmax(sess, x, preds, np.array([tnew]))
            if label_new != label:
                return True
    return False


def dnn_init(dataset, sensitive_param, model_path, max_global, max_local, max_iter):
    """
    The implementation of ADF
    :param dataset: the name of testing dataset
    :param sensitive_param: the index of sensitive feature
    :param model_path: the path of testing model
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param max_global: the maximum number of samples for global search
    :param max_local: the maximum number of samples for local search
    :param max_iter: the maximum iteration of global perturbation
    """
    data = {"census":census_data, "credit":credit_data, "bank":bank_data}
    data_config = {"census":census, "credit":credit, "bank":bank}

    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)
    saver = tf.train.Saver()
    orgmodel_path = model_path + dataset + "/test.model"
    saver.restore(sess, orgmodel_path)
     
    invert_sess = tf.Session(config=config)
    invert_model = dnn(input_shape, nb_classes)
    invert_preds = invert_model(x)
    invert_saver = tf.train.Saver()
    invert_path = model_path + dataset + "/invert/test.model"
    invert_saver.restore(invert_sess, invert_path)
    
    total_init = 0
    init = set()
    init_list = []
    init_index = []

    for i in range(len(X)):
        y_orig = model_argmax(sess, x, preds, np.array([X[i]]))
        y_new = model_argmax(invert_sess, x, invert_preds, np.array([X[i]]))
        if y_orig != y_new:
            total_init += 1     
            if check_for_error_condition(data_config[dataset], sess, x, preds, X[i], sensitive_param):
                temp = X[i].astype('int').tolist()
                #temp = temp[:sensitive_param - 1] + temp[sensitive_param:]
                if tuple(temp) not in init:
                    init.add(tuple(temp))
                    init_list.append(temp)
                    init_index.append(i)

    # create the folder for storing the fairness testing result
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    if not os.path.exists('../results/'+ dataset + '/'+ str(sensitive_param) + '/'):
        os.makedirs('../results/' + dataset + '/'+ str(sensitive_param) + '/')

    # storing the fairness testing result
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/init_samples.npy', np.array(init_list))
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/init_index.npy', np.array(init_index))

    # print the overview information of result
    print("Total Inputs are " + str(total_init))
    print("Total discriminatory inputs- " + str(len(init_list)))
    

def main(argv=None):
    dnn_init(dataset = FLAGS.dataset,
             sensitive_param = FLAGS.sens_param,
             model_path = FLAGS.model_path,
             max_global=FLAGS.max_global,
             max_local=FLAGS.max_local,
             max_iter = FLAGS.max_iter)

if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_integer('sens_param', 9, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_string('model_path', '../models/', 'the path for testing model')
    flags.DEFINE_integer('max_global', 1000, 'maximum number of samples for global search')
    flags.DEFINE_integer('max_local', 1000, 'maximum number of samples for local search')
    flags.DEFINE_integer('max_iter', 10, 'maximum iteration of global perturbation')

    tf.app.run()
