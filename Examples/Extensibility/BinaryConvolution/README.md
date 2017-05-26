# Binary Convolution: Examples/Extensibility/BinaryConvolution

## Overview
This folder contains an implementation of a custom function that performs a binary convolution. It calls into the halide function halide_convolve to achieve good speeds

## Explanation
Modern convolutional neural networks often require well over a billion floating point operations to classify an image. This high number of operations means that running classifiers on resource limited hardware such as Raspberry Pis or smartphones is quite cumbersome and often results in a very low framerate. There have been many efforts to reduce the number of operations required, however, one recent line of inquiry offers a particularly impressive speed up. It was shown in BinaryConnect by Courbariaus et Al (https://arxiv.org/abs/1511.00363) that it is possible to replace weights and activations with single bit values without sacrificing network functionality. By reducing to a single bit, floating point multiplies can be replaced with single bitwise xnor operations. This allows up to 64 operations to be performed in a single clock by packing the bits of the weights and activations appropriately, which ideally would provide a 64x speed up! It turns out actually getting the 64x speedup when comparing to hand optimized libraries like cblas is quite difficult, but a solid 10x is doable, and implemented here.

Single bit binarization essentially just takes the sign of each input, packs those inputs into chunks of 64, then does a series of xnor accumulates. Although this sounds quite simple, it contains many operations that simply don't work well in most deep learning frameworks. In other words, there aren't any popular frameworks that would allow creation of this function using only intrinsics. Fortunately, CNTK provides a custom function utility that lets us create our own functions and have them work quickly and seemlessly. This directory contains various implementations of network binarization using custom functions. Here's a quick tour of the files and what they contain.

|[BinaryConvolveOp.h](./BinaryConvolveOp.h)     |This file contains the CPP custom implementation of binary convolution. It calls into a halide function (halide_convolve) to compute the outputs
|:---------|:---
|[halide_convolve.cpp](./halide_convolve.cpp)   |The Halide definition of binarization and convolution. Allows good speed up in a representationally simple way, see (Halide-lang.org)
|[custom_functions.py](./custom_functions.py)   |Python definitions of custom functions that emulate binarization.  Although no speed up is achieved using these functions, they allow for binary networks to be trained in a very simple way. They also serve as good examples of how to define custom functions purely in python. 
|[customtest.py](./customtest.py)               |Registers the CPP binary convolve function and forward props it, also compares results versus a python binary convolution to confirm correctness.
|[cifar_models.py](./cifar_models.py)           |Contains python definitions of a binary convolutional layer along with many binary network definitions for the cifar dataset. Note that some networks defined in this file are binarized to bit widths besides 1. Although the CPP function only supports single bit binarization at this time, the other bit widths are fun to play with and will be supported soon.
|[convnet.py](./convnet.py)                     |A training driver script which instantiates a model from cifar_models and trains it on the CIFAR10 dataset

## Using this code

### Getting the data

CIFAR-10 dataset is not included in the CNTK distribution but can be easily downloaded and converted by following the instructions in [DataSets/CIFAR-10](../../Image/DataSets/CIFAR-10). We recommend you to keep the downloaded data in the respective folder while downloading, as the scripts in this folder assume that by default.

### Training and testing the binary convolution network

To use this code, see [customtest.py](./customtest.py), which registers the Binary Convolve operation and compares it to a pure python implemenation for correctness. Once registered, the custom function can be used in the same way as any other CNTK op. Because the CPP implementation does not currently support backprop, typical usage will require a model trained using the python approximators. The parameters of the trained model can be cloned into a model using the much faster CPP custom function.

## Editing the Halide Function
If you're interested in modifying the binarization performed in halide_convolve.cpp, go for it! Halide is open source and clonable from (https://github.com/halide/Halide/). Once you have halide set up you can simply run
>> g++ -std=c++11 -I <Halide_Dir>/include/ halide_convolve.cpp <Halide_Dir>/lib/libHalide.a -o halide_convolve -ldl -lpthread -ltinfo -lz
>> ./halide_convolve

to build a new library with your changes. Note that halide_convolve is currently set up to target the platform it's built on, but you can change it to target other things, even small ARM devices like the raspberry pi!

## Making Your Own Model
Exploring other models with binarization is pretty smooth using the functions provided here. Simply define a model using the syntax in cifar_models and whichever python custom function fits your needs. This model can be trained in the same way as any other CNTK model and will learn the proper binary weights. When the model is finished training, you can define another identical model that uses BinaryConvolve ops instead of the python binary convolution. Simply clone the parameters of your trained model into the new model and it should be ready to go. You should see significant forward prop performance speed increase when evaling the model using CPP halidelayers compared to a pure python model.
