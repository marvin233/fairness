import numpy as np
import sys
sys.path.append("../")
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import os
from fairness_data.census import census_data
from fairness_data.bank import bank_data
from fairness_data.compas import compas_data


known_number_types = (int, float, np.float16, np.float32, np.float64,
                      np.int8, np.int16, np.int32, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64)


class _ArgsWrapper(object):

    """
    Wrapper that allows attribute access to dictionaries
    """

    def __init__(self, args):
        if not isinstance(args, dict):
            args = vars(args)
        self.args = args

    def __getattr__(self, name):
        return self.args.get(name)


def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end


def getRandomIndex(n, x):
    np.random.seed(0)
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index


def majority_voting(models, x):
    return models.predict(x)


datasets_dict = {
    'census':census_data,
    'bank': bank_data,
    'compas': compas_data
}


def cluster(dataset, cluster_num=4):
    if os.path.exists('../clusters/' + dataset + '.pkl'):
        clf = joblib.load('../clusters/' + dataset + '.pkl')
    else:
        X, Y, input_shape, nb_classes = datasets_dict[dataset]()
        clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(X)
        joblib.dump(clf, '../clusters/' + dataset + '.pkl')
    return clf
