

import os
import attr

import tensorflow as tf
from images.image import ArtImage

from .neural_transfer import NeuralTransfer
from .vgg_utils import load_vgg_model


my_dir = os.path.dirname(os.path.realpath(__file__))


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    return tf.matmul(A, tf.transpose(A))

@attr.s
class StyleModel:
    pretrained = {'vgg19': 'imagenet-vgg-verydeep-19.mat'}

    content_layer = attr.ib(init=True)
    style_layers = attr.ib(init=True)
    pretrained_model = attr.ib(init=True)
    output_dir = attr.ib(init=True)

    @classmethod
    def create_style_model(cls, model, content_layer='conv4_2', style_layers=(('conv1_1', 0.2),
                                                                              ('conv2_1', 0.2),
                                                                              ('conv3_1', 0.2),
                                                                              ('conv4_1', 0.2),
                                                                              ('conv5_1', 0.2))):

        return VGGStyle(content_layer,
                        style_layers,
                        load_vgg_model(os.path.join(my_dir, '../{}'.format(cls.pretrained[model]))),
                        os.path.join(my_dir, '../art_output'))


@attr.s
class VGGStyle(StyleModel, NeuralTransfer):
    train_step = attr.ib(init=False)

    def content_cost(self, a_C, a_G):
        """Computes the content cost

        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

        Returns:
        J_content -- scalar that you compute using equation 1 above.
        """
        a_C, a_G = args[0], args[1]
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        # Reshape a_C and a_G
        a_C_unrolled = tf.reshape(a_C, [m, n_H * n_W, n_C])
        a_G_unrolled = tf.reshape(a_G, [m, n_H * n_W, n_C])

        return tf.reduce_sum(tf.square(a_C - a_G)) / (4 * n_H * n_W * n_C)

    def style_cost(self, a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

        Returns:
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
        a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
        a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

        # Computing gram_matrices for both images S and G (≈2 lines)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)

        return tf.reduce_sum(tf.square(GS - GG)) / (4 * n_C ** 2 * (n_H * n_W) ** 2)

    def total_style_cost(self, sess):
        J_style = 0

        for layer_name, coeff in self.style_layers:
            # Select the output tensor of the currently selected layer
            out = self.pretrained_model[layer_name]

            # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
            a_S = sess.run(out)

            # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
            # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
            # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
            a_G = out

            # Compute style_cost for the current layer
            J_style_layer = self.style_cost(a_S, a_G)

            # Add coeff * J_style_layer of this layer to overall style cost
            J_style += coeff * J_style_layer
        return J_style

    def total_cost(self, J_content, J_style):
        """Computes the total cost function\n
        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost

        Returns:
        J -- total cost as defined by the formula above.
        """
        return kwargs.get('alpha', 10) * J_content + kwargs.get('beta', 40) * J_style

    def fine_tune(self, content_image, style_image, iterations):
        """
        :param content_image:
        :param style_image:
        :param iterations:
        :return: generated_art
        """
        ci = ArtImage.from_file(content_image)
        si = ArtImage.from_file(style_image)
        generated_art = ArtImage.noisy(ci.matrix)
        session = self.gg(ci.matrix, si.matrix)
        return self.model_nn(session, generated_art, num_iterations=iterations)

    def gg(self, content_image, style_image):
        tf.reset_default_graph()
        sess = tf.InteractiveSession()  # Start interactive session

        # CONTENT
        # Assign the content image to be the input of the VGG model.
        sess.run(model['input'].assign(content_image))

        # Select the output tensor of layer conv4_2
        out = self.pretrained_model[self.content_layer]
        # out = model['conv4_2']

        # Set a_C to be the hidden layer activation from the layer we have selected
        a_C = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        J_content = self.content_cost(a_C, a_G)

        #STYLE
        # Assign the input of the model to be the "style" image
        sess.run(self.pretrained_model['input'].assign(style_image))
        J_style = self.total_style_cost(sess)

        J = total_cost(J_content, J_style, alpha=10, beta=40)

        # ### Optimizer
        # * Use the Adam optimizer to minimize the total cost `J` with learning rate 0.2.
        # * https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
        optimizer = tf.train.AdamOptimizer(2.0)
        self.train_step = optimizer.minimize(J)
        return sess

    def model_nn(self, sess, input_image, num_iterations=200):
        """
        :param sess:
        :param input_image: usually starts with a noisy version of th econtent_image
        :param num_iterations:
        :return:
        """
        # Initialize global variables (you need to run the session on the initializer)
        sess.run(tf.global_variables_initializer())

        # Run the noisy input image (initial generated image) through the model. Use assign().
        sess.run(model['input'].assign(input_image))
        i = 0
        generated_image = None
        for i in range(num_iterations):

            # Run the session on the train_step to minimize the total cost
            sess.run([self.train_step])

            # Compute the generated image by running the session on the current model['input']
            generated_image = sess.run(model['input'])

            if i % 20 == 0:  # Print every 20 iteration
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))
                ArtImage.save_image(os.path.join(self.output_dir, '{}.png'.format(i)), generated_image)

        # save last generated image
        generated_art = ArtImage.from_matrix(generated_image)
        generated_art.save(os.path.join(self.output_dir, 'generated_image_{}.png'.format(i)))
        return generated_image
