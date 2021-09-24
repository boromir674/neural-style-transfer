# coding: utf-8

# # Deep Learning & Art: Neural Style Transfer
# - Implement the neural style transfer algorithm
# - Generate novel artistic images using your algorithm
#
# Most of the algorithms you've studied optimize a cost function to get a set of parameter values. In Neural Style Transfer, you'll optimize a cost function to get pixel values!

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from style.vgg_utils import *
import numpy as np
import tensorflow as tf
import pprint

content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image);


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]



model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

# You should see something the image presented below on the right:
#
# <img src="images/louvre_generated.png" style="width:800px;height:300px;">
#

# Here are few other examples:
#
# - The beautiful ruins of the ancient city of Persepolis (Iran) with the style of Van Gogh (The Starry Night)
# <img src="images/perspolis_vangogh.png" style="width:750px;height:300px;">
#
# - The tomb of Cyrus the great in Pasargadae with the style of a Ceramic Kashi from Ispahan.
# <img src="images/pasargad_kashi.png" style="width:750px;height:300px;">
#
# - A scientific study of a turbulent fluid with the style of a abstract blue fluid painting.
# <img src="images/circle_abstract.png" style="width:750px;height:300px;">


#
# You can also tune your hyperparameters:
# - Which layers are responsible for representing the style? STYLE_LAYERS
# - How many iterations do you want to run the algorithm? num_iterations
# - What is the relative weighting between content and style? alpha/beta


# ## What you should remember
# - Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image
# - It uses representations (hidden layer activations) based on a pretrained ConvNet.
# - The content cost function is computed using one hidden layer's activations.
# - The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.
# - Optimizing the total cost function results in synthesizing new images.


# ### References:
#
# The Neural Style Transfer algorithm was due to Gatys et al. (2015). Harish Narayanan and Github user "log0" also have highly readable write-ups from which we drew inspiration. The pre-trained network used in this implementation is a VGG network, which is due to Simonyan and Zisserman (2015). Pre-trained weights were from the work of the MathConvNet team.
#
# - Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
# - Harish Narayanan, [Convolutional neural networks for artistic style transfer.](https://harishnarayanan.org/writing/artistic-style-transfer/)
# - Log0, [TensorFlow Implementation of "A Neural Algorithm of Artistic Style".](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
# - Karen Simonyan and Andrew Zisserman (2015). [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf)
# - [MatConvNet.](http://www.vlfeat.org/matconvnet/pretrained/)
#

# In[ ]:



