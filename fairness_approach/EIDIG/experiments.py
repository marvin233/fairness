import os
import tensorflow as tf
import numpy as np
import EIDIG
from preprocessing import pre_census
from preprocessing import pre_compas
from preprocessing import pre_bank
from tensorflow import keras
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
np.random.seed(0)
tf.random.set_seed(0)
import copy
import generation_utilities
from fairness_utils.config import census, bank, compas


class NN(object):
    def __init__(self, model):
        self.model = model
    def predict(self, x):
        y = self.model.predict(x)
        y = np.array([int(x>0.5) for [x] in y])
        return y
    def fit(self, x, y):
        self.model.fit(x, y, epochs=30)


def generate_data(data_config, MODEL, w_I_D, w_I, dataset, sensitive_param, model_name, g_num, X, c_num, fashion, protected_attribs, constraint, decay, l_num, max_iter, s_g, s_l, epsilon_l, model_INF):
    if w_I_D:
        seeds_INF = np.load('./results/' + dataset + '/' + str(sensitive_param) + '/NN_w_I_D_init_samples.npy')
        seeds_INF = seeds_INF[:min(g_num, len(seeds_INF))]
    elif w_I:
        seeds_INF = np.load('./results/' + dataset + '/' + str(sensitive_param) + '/NN_w_I_init_samples.npy')
        seeds_INF = seeds_INF[:min(g_num, len(seeds_INF))]
    else:
        g_num = min(len(X), g_num)
        clustered_data = generation_utilities.clustering(X, c_num)
        seeds = np.empty(shape=(0, len(X[0])))
        for i in range(g_num):
            new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i % c_num, fashion=fashion)
            seeds = np.append(seeds, [new_seed], axis=0)
        seeds_INF = copy.deepcopy(seeds)

    g_idi_EIDIG_INF, l_idi_EIDIG_INF, all_idi_EIDIG_INF = EIDIG.individual_discrimination_generation(dataset, sensitive_param, data_config, MODEL, X, seeds_INF, protected_attribs, constraint, model_INF, decay, l_num, l_num+1, max_iter, s_g, s_l, epsilon_l)
    return g_idi_EIDIG_INF, l_idi_EIDIG_INF, all_idi_EIDIG_INF


def comparison(benchmark, protected_attribs, constraint, g_num=100, l_num=100, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    # generate
    pre_config = {
        'compas': pre_compas,
        'census': pre_census,
        'bank': pre_bank
    }
    data_config = {
        "census": census,
        "bank": bank,
        "compas": compas
    }
    model_name = 'EIDIG'
    dataset = benchmark.split('_')[0]
    sensitive_param = int(benchmark.split('_')[1])
    pre = pre_config[dataset]

    X = pre.X
    y = pre.y
    model_INF = keras.models.load_model("./models/original_models/" + dataset + ".h5")
    MODEL = NN(model_INF)


    for [cluster_input, new_input] in [[False, False], [False, True], [True, False]]:
        if cluster_input:
            inputs_way = 'cluster'
        elif new_input:
            inputs_way = 'new'
        else:
            inputs_way = 'ori'
        g_idi_EIDIG_INF, l_idi_EIDIG_INF, disc_inputs = generate_data(data_config, MODEL, cluster_input, new_input, dataset, sensitive_param, model_name, g_num, X, c_num, fashion, protected_attribs, constraint, decay, l_num, max_iter, s_g, s_l, epsilon_l, model_INF)
        np.save('./results/EIDIG/'+dataset+'_'+str(sensitive_param)+'_'+inputs_way+'_idi.npy', np.array(list(disc_inputs)))
        print('g_idi_EIDIG_INF', len(g_idi_EIDIG_INF))
        print('l_idi_EIDIG_INF', len(l_idi_EIDIG_INF))
        print('disc_inputs', len(disc_inputs))
        print('==================================')


def main(argv=None):
    pre_config = {
        'compas': pre_compas,
        'census': pre_census,
        'bank': pre_bank
    }
    pre = pre_config[FLAGS.dataset]
    comparison(FLAGS.dataset+'_'+str(FLAGS.sens_param), [FLAGS.sens_param-1], pre.constraint, FLAGS.max_global, FLAGS.max_local, 0.5, 4, 10, 1.0, 1.0, 1e-6, 'RoundRobin')


if __name__ == '__main__':
    flags.DEFINE_string("dataset", "bank", "the name of dataset")
    flags.DEFINE_integer('sens_param', 1, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_integer('max_global', 100, 'maximum number of samples for global search')
    flags.DEFINE_integer('max_local', 100, 'maximum number of samples for local search')

    tf.compat.v1.app.run()