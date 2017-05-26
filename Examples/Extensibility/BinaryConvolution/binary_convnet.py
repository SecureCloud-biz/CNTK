# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import cntk as C

from custom_convolution_ops import *

# the definition for a binary convolution layer
def BinConvolution(operand,
                   filter_shape,
                   num_filters=1,
                   channels = 1,
                   init=C.glorot_uniform(),
                   pad=False,
                   strides=1,
                   bias=True,
                   init_bias=0,
                   op_name='BinConvolution', name=''):
    """ arguments:
            operand: tensor to convolve
            filter_shape: tuple indicating filter size
            num_filters: number of filters to use 
            channels: number of incoming channels
            init: type of initialization to use for weights
    """
    kernel_shape = (num_filters, channels) + filter_shape
    W = C.parameter(shape=kernel_shape, init=init, name="filter")

    binary_convolve_operand_p = C.placeholder(operand.shape, operand.dynamic_axes)
    binary_convolve = C.convolution(CustomMultibit(W, 1), CustomMultibit(binary_convolve_operand_p, 1), auto_padding=[False, pad, pad], strides=[strides])
    r = C.as_block(binary_convolve, [(binary_convolve_operand_p, operand)], 'binary_convolve')

    bias_shape = (num_filters, 1, 1)
    b = C.parameter(shape=bias_shape, init=init_bias)
    r = r + b
    # apply learnable param relu
    P = C.parameter(shape=r.shape, init=init, name="prelu")
    r = C.param_relu(P, r)
    return r

# Create the network.
def M1A1W():

    # Input variables denoting the features and label data
    feature_var = C.input((num_channels, image_height, image_width))
    label_var = C.input((num_classes))

    # apply model to input
    scaled_input = C.element_times(C.constant(0.00390625), feature_var)

    # first layer is ok to be full precision
    z = C.layers.Convolution((3, 3), 32, pad=True, activation=C.relu)(scaled_input)
    z = C.layers.MaxPooling((3,3), strides=(2,2))(z)

    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = BinConvolution(z, (3,3), 128, channels=32, pad=True)
    z = C.layers.MaxPooling((3,3), strides=(2,2))(z)

    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = BinConvolution(z, (3,3), 128, channels=128, pad=True)
    z = C.layers.MaxPooling((3,3), strides=(2,2))(z)

    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = BinConvolution(z, (1,1), num_classes, channels=128, pad=True)
    z = C.layers.AveragePooling((z.shape[1], z.shape[2]))(z)
    z = C.reshape(z, (num_classes,))

    # add on a binary regularization a la gang hua
    weight_sum = C.constant(0)
    for p in z.parameters:
        if (p.name == "filter"):
            weight_sum = C.plus(weight_sum, C.reduce_sum(C.minus(1, C.square(p))))
    bin_reg = C.element_times(.000005, weight_sum)


    # after the last layer, we need to apply a learnable scale
    SP = C.parameter(shape=z.shape, init=0.001)
    z = C.element_times(z, SP)

    # loss and metric
    ce = C.cross_entropy_with_softmax(z, label_var)
    ce = C.plus(ce, bin_reg)
    pe = C.classification_error(z, label_var)

    C.logging.log_number_of_parameters(z) ; print()

    return {
        'feature': feature_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'output': z
    }

abs_path = os.path.dirname(os.path.abspath(__file__))
custom_convolution_ops_dir = os.path.join(abs_path, "..", "..", "Image", "Classification", "ConvNet", "Python")
sys.path.append(custom_convolution_ops_dir)

from ConvNet_CIFAR10_DataAug import *

############################# 
# main function boilerplate #
#############################

if __name__=='__main__':
    # create model
    model = M1A1W()

    # train using the 
    reader      = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
    train_model(reader, model['output'], C.combine([model['ce'], model['pe']]), max_epochs=80)

    # save and load (as an illustration)
    path = data_path + "/model.cmf"
    model.save(path)

    # test
    model = Function.load(path)
    # Replace all python binary convolution user-functions with the fast native binary convolution operator
    # Note that the native binary convolution Function currently only supports eval, and is thus not used for training
    

    reader = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)
    
