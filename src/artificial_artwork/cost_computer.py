import tensorflow as tf
from .nst_math import gram_matrix


class NSTCostComputer:
    @classmethod
    def compute(cls, J_content: float, J_style: float, alpha: float=10, beta: float=40) -> float:
        """Compute the total cost function.

        Computes the total cost (aka learning error) as a linear combination of
        the 'content cost' and 'style cost'.

        Total cost = alpha * J_content + beta * J_style
        
        Or mathematically expressed as:

        J(G) = alpha * J_content(C, G) + beta * J_style(S, G)

        where G: Generated Image, C: Content Image, S: Style Image
        and J, J_content, J_style are mathematical functions

        Args:
            J_content (float): content cost
            J_style (float): style cost
            alpha (float, optional): hyperparameter to weight content cost. Defaults to 10.
            beta (float, optional): hyperparameter to weight style cost. Defaults to 40.

        Returns:
            float: the total cost as defined by the formula above
        """
        return alpha * J_content + beta * J_style


class NSTContentCostComputer:
    @classmethod
    def compute(cls, a_C, a_G):
        """
        Computes the content cost
        
        Assumption 1: a layer l has been chosen from a (Deep) Neural Network
        trained on images, that should act as a style model.

        Then:
        1. a_C (3D volume) are the hidden layer activations in the chosen layer (l), when the C
        image is forward propagated (passed through) in the network.
        
        2. a_G (3D volume) are the hidden layer activations in the chosen layer (l), when the G
        image is forward propagated (passed through) in the network.

        3. The above activations are a n_H x n_W x n_C tensor
        OR Height x Width x Number_of_Channels

        Pseudo code for latex expression of the mathematical equation:

        J_content(C, G) = \frac{1}{4 * n_H * n_W * n_C} * \sum_{all entries} (a^(C) - a^(G))^2
        OR
        J_content(C, G) = sum_{for all entries} (a^(C) - a^(G))^2 / (4 * n_H * n_W * n_C)

        Note that n_H * n_W * n_C is part of the normalization term.

        Args:
            a_C (tensor): of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
            a_G (tensor): of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
        
        Returns: 
            (tensor): 1D with 1 scalar value computed using the equation above
        """
        # Retrieve dimensions from a_G
        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape a_C and a_G
        # a_C_unrolled = tf.reshape(a_C, [m, n_H * n_W, n_C])
        # a_G_unrolled = tf.reshape(a_G, [m, n_H * n_W, n_C])
        
        # compute the cost
        J_content = tf.reduce_sum(tf.square(a_C - a_G)) / (4 * n_H * n_W * n_C)
        return J_content


class GramMatrixComputer(type):
    def __new__(mcs, *args, **kwargs):
        class_object = super().__new__(mcs, *args, **kwargs)
        class_object.compute_gram = gram_matrix
        return class_object


class NSTLayerStyleCostComputer(metaclass=GramMatrixComputer):

    @classmethod
    def compute(cls, a_S, a_G):
        """
        Compute the Style Cost, using the activations of the l style layer.

        Mathematical equation written in Latex code:
        J^{[l]}_style (S, G) = \frac{1}{4 * n_c^2 * (n_H * n_W)^2} \sum^{n_C}_{i=1} \sum^{c_C}_{j=1} (G^{(S)}_{(gram)i,j} - G^{(G)}_{(gram)i,j})^2
        
        OR

        Cost(S, G) = \sum^{n_C}_{i=1} \sum^{c_C}_{j=1} (G^{(S)}_{(gram)i,j} - G^{(G)}_{(gram)i,j})^2 / ( 4 * n_c^2 * (n_H * n_W)^2 )
        
        Args:
            a_S (tensor): tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
            a_G (tensor): tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
        
        Returns: 
            (tensor): J_style_layer tensor representing a scalar value, style cost defined above by equation (2)
        """
        # Retrieve dimensions from a_G
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        
        # Reshape the images to have them of shape (n_C, n_H*n_W)
        a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
        a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

        # Computing gram_matrices for both images S and G
        GS = cls.compute_gram(a_S)
        GG = cls.compute_gram(a_G)

        # Computing the loss
        J_style_layer = tf.reduce_sum(tf.square(GS - GG)) / ( 4 * n_C**2 * (n_H * n_W)**2)
            
        return J_style_layer


class NSTStyleCostComputer:
    style_layer_cost = NSTLayerStyleCostComputer.compute

    @classmethod
    def compute(cls, tf_session, model_layers):
        """
        Computes the overall style cost from several chosen layers
        
        Args:
            tf_session (tf.compat.v1.INteractiveSession): the active interactive tf session
            model_layers () -- our image model (probably pretrained on large dataset)
            STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them
        
        Returns: 
            (tensor): J_style - tensor representing a scalar value, style cost defined above by equation (2)
        """
        # initialize the overall style cost
        J_style = 0

        # for layer_name, coeff in STYLE_LAYERS:
        for style_layer_id, nst_style_layer in model_layers:

            # Select the output tensor of the currently selected layer
            out = nst_style_layer.neurons
            # out = model[layer_name]

            # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
            a_S = tf_session.run(out)

            # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
            # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
            # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
            a_G = out
            
            # Compute style_cost for the current layer
            J_style_layer = cls.style_layer_cost(a_S, a_G)

            # Add coeff * J_style_layer of this layer to overall style cost
            J_style += nst_style_layer.coefficient * J_style_layer

        return J_style
