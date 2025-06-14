import numpy as np
import tensorflow as tf


class Layer(object):
    def get_output_shape(self):
        return self.output_shape


class Linear(Layer):
    def __init__(self, num_hid):
        self.num_hid = num_hid

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                 keep_dims=True))
        with tf.name_scope("linear"):
            self.W = tf.Variable(init, name='kernel')
            self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'), name='bias')

    def fprop(self, x):
        x = tf.cast(x, tf.float32)
        return tf.matmul(x, self.W) + self.b


class Conv2D(Layer):
    def __init__(self, output_channels, kernel_shape, strides, padding):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape

        if len(self.kernel_shape)==2:
            kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                    self.output_channels)
        else:
            kernel_shape = tuple(self.kernel_shape) + (self.output_channels,)

        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        init = tf.random_normal(kernel_shape, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                   axis=(0, 1, 2)))
        with tf.name_scope("conv2d"):
            self.kernels = tf.Variable(init, name='kernel')
            self.b = tf.Variable(
                np.zeros((self.output_channels,)).astype('float32'), name='bias')

        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b


class MaxPooling(Layer):
    def __init__(self, ksize, strides, padding):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        batch_size, rows, cols, input_channels = input_shape
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.max_pool(x, (1,) + tuple(self.ksize) + (1,), (1,) + tuple(self.strides) + (1,), self.padding)


class AvgPooling(Layer):
    def __init__(self, ksize, strides, padding):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        batch_size, rows, cols, input_channels = input_shape
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.avg_pool(x, (1,) + tuple(self.ksize) + (1,), (1,) + tuple(self.strides) + (1,), self.padding)


class ReLU(Layer):
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.relu(x)


class Tanh(Layer):
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.tanh(x)


class Sigmoid(Layer):
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.sigmoid(x)


class Softmax(Layer):
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)


class Dropout(Layer):
    def __init__(self, keep_prob):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.dropout(x, self.keep_prob)


class Flatten(Layer):
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [shape[0], output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])