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

LAYERS = (
    'conv1_1' ,
    'conv1_2' ,
    'avgpool1',
    'conv2_1' ,
    'conv2_2' ,
    'avgpool2',
    'conv3_1' ,
    'conv3_2' ,
    'conv3_3' ,
    'conv3_4' ,
    'avgpool3',
    'conv4_1' ,
    'conv4_2' ,
    'conv4_3' ,
    'conv4_4' ,
    'avgpool4',
    'conv5_1' ,
    'conv5_2' ,
    'conv5_3' ,
    'conv5_4' ,
    'avgpool5',
)
