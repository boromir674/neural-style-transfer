"""This module contains the high-level architecture design of our 'style model'.

As 'style model' we define a neural network (represented as a mathematical
graph) with several convolutional layers with weights extacted from a pretrained
image model (ie the vgg19 model trained for the task of image classification on
the imagenet dataset) and some average pooling layers with predefined weights.

All weigths of the style model stay constants during optimization of the
training objective (aka cost function).

Here we only take the convolution layer weights and define several new
AveragePooling. We opt for AveragePooling compared to MaxPooling, since it has
been shown to yield better results.
"""

# Each row instructs to build a Layer in a Computattional Graph which is used
# for the NST Algorithm: to pass images and get compute Tensors, to optimize
# weights, compute Costs, etc

### For each 'conv_' item:
# we build a Layer of with ReLU activation function and
# Conv2D Tensor Operation ( AX + b ).
# Other ReLU parameters: strides=[1, 1, 1, 1], padding='SAME'
# The weights, A, b matrices, of the Conv2D are extracted from a
# pretrained model (ie the vgg19 model trained for the task of image
# classification on the imagenet dataset).

### For each 'avgpool_' item:
# we build a Layer of with Average Pooling operation
# (ie a Tensor Operation that computes the average of a 2x2 window of the
# input Tensor). The weights of the Average Pooling are predefined.
# Avg Pool Parameters: ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'

LAYERS = (
    "conv1_1",  # ReLU A, b weight matrices loaded with pretrained model values
    "conv1_2",  # ReLU ...
    "avgpool1",  # AvgPool parameters: ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
    "conv2_1",
    "conv2_2",
    "avgpool2",
    "conv3_1",
    "conv3_2",
    "conv3_3",
    "conv3_4",
    "avgpool3",
    "conv4_1",
    "conv4_2",
    "conv4_3",
    "conv4_4",
    "avgpool4",
    "conv5_1",
    "conv5_2",
    "conv5_3",
    "conv5_4",
    "avgpool5",
)
