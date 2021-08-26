from abc import ABCMeta
import numpy as np
from fairness_model.layer import *


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        """
        For compatibility with functions used as model definitions (taking
        an input tensor and returning the tensor giving the output
        of the model on that input).
        """
        return self.get_probs(*args, **kwargs)

    def get_layer(self, x, layer):
        """
        Expose the hidden features of a model given a layer name.
        :param x: A symbolic representation of the network input
        :param layer: The name of the hidden layer to return features at.
        :return: A symbolic representation of the hidden features
        :raise: NoSuchLayerError if `layer` is not in the model.
        """
        # Return the symbolic representation for this layer.
        output = self.fprop(x)
        try:
            requested = output[layer]
        except KeyError:
            raise NoSuchLayerError()
        return requested

    def get_logits(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output logits (i.e., the
                 values fed as inputs to the softmax layer).
        """
        return self.get_layer(x, 'logits')

    def get_probs(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output probabilities (i.e.,
                the output values produced by the softmax layer).
        """
        try:
            return self.get_layer(x, 'probs')
        except NoSuchLayerError:
            pass
        except NotImplementedError:
            pass
        import tensorflow as tf
        return tf.nn.softmax(self.get_logits(x))

    def get_layer_names(self):
        """
        :return: a list of names for the layers that can be exposed by this
        model abstraction.
        """

        if hasattr(self, 'layer_names'):
            return self.layer_names

        raise NotImplementedError('`get_layer_names` not implemented.')

    def fprop(self, x):
        """
        Exposes all the layers of the model returned by get_layer_names.
        :param x: A symbolic representation of the network input
        :return: A dictionary mapping layer names to the symbolic
                 representation of their output.
        """
        raise NotImplementedError('`fprop` not implemented.')


class CallableModelWrapper(Model):
    def __init__(self, callable_fn, output_layer):
        """
        Wrap a callable function that takes a tensor as input and returns
        a tensor as output with the given layer name.
        :param callable_fn: The callable function taking a tensor and
                            returning a given layer as output.
        :param output_layer: A string of the output layer returned by the
                             function. (Usually either "probs" or "logits".)
        """

        self.output_layer = output_layer
        self.callable_fn = callable_fn

    def get_layer_names(self):
        return [self.output_layer]

    def fprop(self, x):
        return {self.output_layer: self.callable_fn(x)}


class NoSuchLayerError(ValueError):

    """Raised when a layer that does not exist is requested."""


class NN(object):
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        y = self.model.predict(x)
        y = np.array([int(x>0.5) for [x] in y])
        return y

    def fit(self, x, y, epochs):
        self.model.fit(x, y, epochs=epochs)


def dnn(input_shape=(None, 13), nb_classes=2):
    activation = ReLU
    layers = [Linear(30),
              activation(),
              Linear(20),
              activation(),
              Linear(15),
              activation(),
              Linear(10),
              activation(),
              Linear(5),
              activation(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model


class MLP(Model):
    def __init__(self, layers, input_shape):
        """
        Construct a multilayer perceptron (MLP)
        :param layers: a sequence of layers
        :param input_shape: the shape of dataset
        """
        super(MLP, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
                layer.name = name
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False):
        """
        Get the symbolic output of each layer
        :param x: the input placeholder
        :param set_ref: whether set reference
        :return: a dictionary of layers' name and tensor
        """
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states