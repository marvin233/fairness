'''
Author: Marvin
Date: 2020-11-16 21:06:06
Description: 
'''
import numpy as np
import pandas as df 
import sys
sys.path.append("../")

import tensorflow as tf
from tensorflow.python.platform import flags
from adf_data.census import census_data
from adf_data.bank import bank_data
from adf_data.credit import credit_data
from adf_data.execution import execution_data
from adf_data.compas import compas_data
from adf_utils.utils_tf import model_train, model_eval
from adf_model.tutorial_models import dnn
from adf_utils.config import census, credit, bank, execution, compas
import random

FLAGS = flags.FLAGS
random.seed(0)


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


def invert_sensitive(x, sens_index):
    sens_index = sens_index - 1
    x = df.DataFrame(x)
    input_bounds = None
    if FLAGS.dataset == 'census':
        input_bounds = census.input_bounds
    elif FLAGS.dataset == 'bank':
        input_bounds = bank.input_bounds
    elif FLAGS.dataset == 'credit':
        input_bounds = credit.input_bounds
    elif FLAGS.dataset == 'execution':
        input_bounds = execution.input_bounds
    elif FLAGS.dataset == 'compas':
        input_bounds = compas.input_bounds
    input_bounds = input_bounds[sens_index]
    x.iloc[:, sens_index] = x.iloc[:, sens_index].astype('int') # only for binary feature
    for i in range(len(x)):
        x.iloc[i, sens_index] = generate_random(x.iloc[i, sens_index], input_bounds)
    x = np.array(x)
    return x 


def training(dataset, sens_param, model_path, nb_epochs, batch_size, learning_rate, invert):
    """
    Train the model
    :param dataset: the name of testing dataset
    :param model_path: the path to save trained model
    """
    data = {"census": census_data, "credit": credit_data, "bank": bank_data, "execution": execution_data, "compas":compas_data}
    train_dir = model_path + dataset + "/" + str(FLAGS.sens_param) + "/"

    # prepare the data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    if invert:
        X = invert_sensitive(X, sens_param)
        train_dir = model_path + dataset +"/"+ str(FLAGS.sens_param) +"/invert/"

    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)
    
    # training parameters
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': train_dir,
        'filename': 'test.model'
    }

    # training procedure
    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2019, 7, 15])
    model_train(sess, x, y, preds, X, Y, args=train_params,
                rng=rng, save=True)

    # evaluate the accuracy of trained model
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x, y, preds, X, Y, args=eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))


def main(argv=None):
    training(dataset = FLAGS.dataset,
             sens_param = FLAGS.sens_param,
             model_path = FLAGS.model_path,
             nb_epochs = FLAGS.nb_epochs,
             batch_size = FLAGS.batch_size,
             learning_rate = FLAGS.learning_rate,
             invert = False)
    training(dataset = FLAGS.dataset,
             sens_param = FLAGS.sens_param,
             model_path = FLAGS.model_path,
             nb_epochs = FLAGS.nb_epochs,
             batch_size = FLAGS.batch_size,
             learning_rate = FLAGS.learning_rate,
             invert = True)


if __name__ == '__main__':
    flags.DEFINE_string("dataset", "bank", "the name of dataset")
    flags.DEFINE_integer('sens_param', 1, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_string("model_path", "../models/", "the name of path for saving model")
    flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')

    tf.app.run()