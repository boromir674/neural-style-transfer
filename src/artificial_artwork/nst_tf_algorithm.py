from time import time
from typing import Dict
import attr
import tensorflow as tf

from .tf_session_runner import TensorflowSessionRunner
from .style_model import graph_factory
from .cost_computer import NSTContentCostComputer, NSTStyleCostComputer
from .utils import Subject


@attr.s
class NSTAlgorithmRunner:
    session_runner = attr.ib()
    apply_noise = attr.ib()
    # model_design = attr.ib()
    optimization = attr.ib(default=attr.Factory(lambda: Optimization()))

    nst_algorithm = attr.ib(init=False, default=None)
    parameters = attr.ib(init=False, default=None)

    nn_builder = attr.ib(init=False, default=None)
    nn_cost_builder = attr.ib(init=False, default=None)

    # broadcast facilities to notify observers/listeners
    progress_subject = attr.ib(init=False, default=attr.Factory(Subject))
    persistance_subject = attr.ib(init=False, default=attr.Factory(Subject))

    # NETWORK_OUTPUT = 'conv4_2'

    @classmethod
    def default(cls, apply_noise):
        session_runner = TensorflowSessionRunner.with_default_graph_reset()
        return NSTAlgorithmRunner(session_runner, apply_noise)

    def run(self, nst_algorithm, model_design):
        ## Prepare ##
        self.nst_algorithm = nst_algorithm

        c_image = nst_algorithm.parameters.content_image
        s_image = nst_algorithm.parameters.style_image

        image_specs = type('ImageSpecs', (), {
            'height': c_image.matrix.shape[1],
            'width': c_image.matrix.shape[2],
            'color_channels': c_image.matrix.shape[3]
        })()

        print(' --- Loading CV Image Model ---')

        style_network = graph_factory.create(image_specs, model_design)

        noisy_content_image_matrix = self.apply_noise(self.nst_algorithm.parameters.content_image.matrix)

        print(' --- Building Computations ---')

        self.nn_builder = NeuralNetBuilder(
            style_network,
            self.session_runner.session
        )

        # indicate content_image and the output layer of the Neural Network
        self.nn_builder.build_activations(
            c_image.matrix, model_design.network_design.output_layer)

        self.nn_cost_builder = CostBuilder(
            NSTContentCostComputer.compute,
            NSTStyleCostComputer.compute,
        )

        self.nn_cost_builder.build_content_cost(
            self.nn_builder.a_C,
            self.nn_builder.a_G,
        )

        self.nn_builder.assign_input(s_image.matrix)

        # manually set the neurons attribute for each NSTStyleLayer
        # using the loaded cv model (which is a dict of layers)
        # the NSTStyleLayer ids attribute to query the dict
        for style_layer_id, nst_style_layer in model_design.network_design.style_layers:
            nst_style_layer.neurons = style_network[style_layer_id]
        # TODO obviously encapsulate the above code elsewhere

        self.nn_cost_builder.build_style_cost(
            self.session_runner.session,
            model_design.network_design.style_layers,
        )

        self.nn_cost_builder.build_cost(alpha=10, beta=40)

        self.optimization = Optimization()
        self.optimization.optimize_against(self.nn_cost_builder.cost)

        ## Run Iterative Learning Algorithm ##

        print(' --- Preparing Iterative Learning Algorithm ---')
        input_image = noisy_content_image_matrix

        # Initialize global variables (you need to run the session on the initializer)
        self.session_runner.run(tf.compat.v1.global_variables_initializer())

        # Run the noisy input image (initial generated image) through the model
        self.session_runner.run(style_network['input'].assign(input_image))
        self.perform_nst(style_network)

    def perform_nst(self, style_network):
        print(' --- Running Iterative Algorithm ---')

        i = 0
        self.time_started = time()

        while not self.nst_algorithm.parameters.termination_condition.satisfied:
            generated_image = self.iterate(style_network)
            progress = self._progress(generated_image, completed_iterations=i+1)
            if i % 20 == 0:
                self._notify_persistance(progress)
                Jt, Jc, Js = self._eval_cost()
                self._print_cost(type('C', (), {
                    'Jt': Jt,
                    'Jc': Jc,
                    'Js': Js,
                }), iteration_index=i)
                progress['metrics'].update({
                    'cost': Jt,
                    'content-cost': Jc,
                    'style-cost': Js,
                })
            progress['metrics']['duration'] = time() - self.time_started  # in seconds
            self._notify_progress(progress)
            i += 1

        try:
            self._notify_persistance(progress)
        except NameError as progress_not_evaluated_error:
            raise NoIterationsDoneError(
                'The algorithm did not iterate. Probably the '
                'f"{self.nst_algorithm.parameters.termination_condition}"'
                ' termination condition is too "strict."') \
                    from progress_not_evaluated_error

        print(' --- Finished Learning Algorithm :) ---')


    def iterate(self, image_model):
        # Run the session on the train_step to minimize the total cost
        self.session_runner.run([self.optimization.train_step])

        # Compute the generated image by running the session on the current model['input']
        generated_image = self.session_runner.run(image_model['input'])
        return generated_image

    def _progress(self, generated_image, completed_iterations: int) -> Dict:
        return {
            'metrics': {
                'iterations': completed_iterations,  # number of iterations completed
            },
            'content_image_path': self.nst_algorithm.parameters.content_image.file_path,
            'style_image_path': self.nst_algorithm.parameters.style_image.file_path,
            'output_path': self.nst_algorithm.parameters.output_path,
            'matrix': generated_image,
        }

    def _notify_persistance(self, progress):
        self.persistance_subject.state = type('SubjectState', (), progress)
        self.persistance_subject.notify()

    def _notify_progress(self, progress):
        # set subject with the appropriate state to broadcast
        self.progress_subject.state = type('SubjectState', (), progress)
        # notify all observers/listeners that have 'subscribed'
        self.progress_subject.notify()

    def _eval_cost(self):
        # pass cost objects in session to evaluate them
        Jt, Jc, Js = self.session_runner.run([
            self.nn_cost_builder.cost,
            self.nn_cost_builder.content_cost,
            self.nn_cost_builder.style_cost,
        ])
        return Jt, Jc, Js

    def _print_cost(self, costs, iteration_index):
        print(
            f' Iteration: {iteration_index}\n'
            f'  Total Cost    : {costs.Jt}\n'
            f'   Content cost : {costs.Jc}\n'
            f'   Style cost   : {costs.Js}\n'
        )


class NoIterationsDoneError(Exception): pass


@attr.s
class NeuralNetBuilder:
    """Configure a pretrained image model to facilitate nst algorithm."""
    model = attr.ib()
    session = attr.ib()
    output_neurons = attr.ib(init=False, default=None)
    a_C = attr.ib(init=False, default=None)
    a_G = attr.ib(init=False, default=None)

    def assign_input(self, image):
        # Assign the content image to be the input of the VGG model.
        self.session.run(self.model['input'].assign(image))

    def build_activations(self, content_image, model_layer_id: str):
        self.assign_input(content_image)
        self._set_output(model_layer_id)
        self._setup_activations()

    def _set_output(self, model_layer_id: str):
        # Select the output tensor of a Neural Network layer
        self.output_neurons = self.model[model_layer_id]

    def _setup_activations(self):
        # Set a_C to be the hidden layer activation from the layer we have selected
        self.a_C = self.session.run(self.output_neurons)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        self.a_G = self.output_neurons


@attr.s
class CostBuilder:
    compute_content_cost = attr.ib()
    compute_style_cost = attr.ib()

    cost = attr.ib(init=False, default=None)
    content_cost = attr.ib(init=False, default=None)
    style_cost = attr.ib(init=False, default=None)

    def build_content_cost(self, content_image_activations, generated_image_activations):
        # Compute the content cost
        self.content_cost = self.compute_content_cost(
            content_image_activations,
            generated_image_activations)

    def build_style_cost(self, tf_session, style_layers):
        # Compute the style cost
        self.style_cost = self.compute_style_cost(tf_session, style_layers)

    def build_cost(self, alpha=10, beta=40):
        """Build the function of the Total Cost (loss function).

        The Total Cost function J(G) (learning error) is the linear combination
        of the 'content cost' (J_content) and 'style cost' (J_style).

        After invoking this method the Cost Function is accessible via the
        'cost' attribute.

        Total cost = alpha * J_content + beta * J_style

        Or mathematically expressed as:

        J(G) = alpha * J_content(C, G) + beta * J_style(S, G)

        where G: Generated Image, C: Content Image, S: Style Image
        and J, J_content, J_style are mathematical functions

        Args:
            alpha (float, optional): hyperparameter to weight content cost. Defaults to 10.
            beta (float, optional): hyperparameter to weight style cost. Defaults to 40.
        """
        self.cost = alpha * self.content_cost + beta * self.style_cost


@attr.s
class Optimization:
    optimizer = attr.ib(default=attr.Factory(lambda: tf.compat.v1.train.AdamOptimizer(2.0)))
    _train_step = attr.ib(init=False, default=None)

    def optimize_against(self, cost):
        self._train_step = self.optimizer.minimize(cost)

    @property
    def train_step(self):
        return self._train_step
