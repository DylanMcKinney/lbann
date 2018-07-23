#!/usr/bin/env python
import sys
import os
import subprocess
import functools
#global lbann_pb2
import lbann_pb2
import google.protobuf.text_format as txtf
pb = lbann_pb2.LbannPB()
class _Model(object):
    #def __init__(self):
    #    pass
    def add(self, layer):
        pass
        #il = model.layer.add()
        #l.name = layer.name
        #l.parents = str_list(layer.parents)
        #l.device_allocation = device
        #exec('l.' + layer_type + '.SetInParent()')

    def compile(self, loss, optimizer, metrics):
        #obj = pb.model.objective_function.add()
        #exec('obj.' + loss + '.SetInParent()')
        #pb.model.metric = metrics
        layer = pb.model.layer.add()
        exec('layer.' + 'target' + '.SetInParent()')
        with open('keras_cnn.prototext', 'w') as f:
            f.write(txtf.MessageToString(pb))
        #, loss=objective_function.cross_entropy(),
                #optimizer=optimizer.sgd(),
                #metric=Metric.categorical_accuracy()):

class Sequential(_Model):
    def __init__(self, batch_size=64,epochs=10):
        pb.model.name = 'sequential_model'
        pb.model.data_layout = 'data_parallel'
        pb.model.mini_batch_size = batch_size
        pb.model.num_epochs = epochs
        pb.model.block_size = 256
        pb.model.disable_cuda = True
        layer = pb.model.layer.add()
        exec('layer.' + 'input' + '.SetInParent()')
        layer.input.io_buffer = "partitioned"

class _Layer(object):
    #def __init__(self, **kwargs):
    #    pass
    def add_activation(self, activation):
        layer = pb.model.layer.add()
        exec('layer.' + activation + '.SetInParent()')

    def handle_tuple(self, val):
        if isinstance(val, tuple):
            return ' '.join(str(i) for i in val)
        elif isinstance(val, str):
            return val

class _Learning(_Layer):
    def add_weights(name, initializer= 'constant_initializer'):
        weight = model.weights.add()
        weight.name = name
        exec('w.' + initializer + '.SetInParent()')


class _Conv(_Learning):
    def __init__(self, rank,
                 filters, kernel_size,
                 strides, padding,
                 data_format, dilation_rate,
                 activation, use_bias,
                 kernel_initializer, bias_initializer,
                 kernel_regularizer, bias_regularizer, activity_regularizer,
                 kernel_constraint, bias_constraint, **kwargs):
        super(_Conv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint



class Conv2D(_Conv):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        layer = pb.model.layer.add()
        exec('layer.convolution.SetInParent()')
        layer.convolution.num_dims = 2
        layer.convolution.num_output_channels = filters
        layer.convolution.conv_dims = self.handle_tuple(kernel_size)
        layer.convolution.conv_strides = self.handle_tuple(strides)
        layer.convolution.conv_pads =  "0 0"
        layer.convolution.has_bias = use_bias
        layer.convolution.has_vectors = True
        if activation:
            self.add_activation(activation)





class _Pooling2D(_Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                  **kwargs):
        super(_Pooling2D, self).__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

class MaxPooling2D(_Pooling2D):
    def __init__(self, pool_size=(2, 2),
                 strides=None,
                 padding='valid',
                 **kwargs):
        super(MaxPooling2D, self).__init__(
            pool_size,
            strides, padding,
            **kwargs)
        layer = pb.model.layer.add()
        exec('layer.pooling.SetInParent()')
        layer.pooling.num_dims = 2
        layer.pooling.has_vectors = True
        layer.pooling.pool_pads = "0 0"
        layer.pooling.pool_dims = self.handle_tuple(pool_size)
        if not strides:
            strides = pool_size
        layer.pooling.pool_strides = self.handle_tuple(strides)
        layer.pooling.pool_mode = 'max'


class Flatten(_Layer):
    def __init__(self, data_format=None, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        layer = pb.model.layer.add()
        exec('layer.reshape.SetInParent()')
        layer.reshape.flatten = True

class Dropout(_Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        layer = pb.model.layer.add()
        exec('layer.dropout.SetInParent()')
        layer.dropout.keep_prob = rate

class Dense(_Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense, self).__init__(**kwargs)
        layer = pb.model.layer.add()
        exec('layer.fully_connected.SetInParent()')
        layer.fully_connected.num_neurons = units
        layer.fully_connected.has_bias = use_bias
        if activation:
            self.add_activation(activation)
        #self.activation = activations.get(activation)
        #self.kernel_initializer = initializers.get(kernel_initializer)
        #self.bias_initializer = initializers.get(bias_initializer)
        #self.kernel_regularizer = regularizers.get(kernel_regularizer)
        #self.bias_regularizer = regularizers.get(bias_regularizer)
        #self.activity_regularizer = regularizers.get(activity_regularizer)
        #self.kernel_constraint = constraints.get(kernel_constraint)
        #self.bias_constraint = constraints.get(bias_constraint)
        #self.input_spec = InputSpec(min_ndim=2)
        #self.supports_masking = True





