import typing as t
from time import time
from typing import Dict

import attr
import tensorflow as tf
from software_patterns import Subject

from .cost_computer import NSTContentCostComputer, NSTStyleCostComputer
from .style_model import graph_factory
from .tf_session_runner import TensorflowSessionRunner

# Represents a Layer (cloned from a pretrained model network)
# of the non-constant network we optimize weights for

LayerID = str


class NSTStyleLayerType(t.Protocol):
    id: LayerID
    coefficient: float
    neurons: t.Any


class NSTLayerSelectionType(t.Protocol):
    def __iter__(self) -> t.Iterable[t.Tuple[LayerID, NSTStyleLayerType]]:
        ...


class NetworkDesignType(t.Protocol):
    network_layers: t.Tuple[LayerID]
    style_layers: NSTLayerSelectionType
    output_layer: LayerID


class ModelDesignType(t.Protocol):
    pretrained_model: t.Any
    network_design: NetworkDesignType


# define custom Layer type for type checking
Layer = t.Union[t.Any, tf.Tensor]


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

    # references to most recently Evaluated Cost values
    Jt = attr.ib(init=False, default=None)
    Jc = attr.ib(init=False, default=None)
    Js = attr.ib(init=False, default=None)

    # NETWORK_OUTPUT = 'conv4_2'

    @classmethod
    def default(cls, apply_noise):
        session_runner = TensorflowSessionRunner.with_default_graph_reset()
        return NSTAlgorithmRunner(session_runner, apply_noise)

    def run(
        self,
        nst_algorithm,
        model_design: ModelDesignType,
    ):
        ## Prepare ##
        self.nst_algorithm = nst_algorithm

        c_image = nst_algorithm.parameters.content_image
        s_image = nst_algorithm.parameters.style_image

        image_specs = type(
            "ImageSpecs",
            (),
            {
                "height": c_image.matrix.shape[1],
                "width": c_image.matrix.shape[2],
                "color_channels": c_image.matrix.shape[3],
            },
        )()

        print(" --- Loading CV Image Model ---")

        style_network = graph_factory.create(
            image_specs,  # Input tensor is designed to match images dimensions
            model_design,  #
        )
        # One-Time Operation: APPLY NOISE to Content Image, with a ratio
        from artificial_artwork.image.image_operations import ImageNoiseAdder

        noise_adder = ImageNoiseAdder(seed=1234)

        RATIO_VALUE_OF_ORIGINAL_PROTOTYPE = 0.6

        noisy_content_image_matrix = noise_adder(
            # image as numpy ndarry (numpy.typing.NDArray)
            self.nst_algorithm.parameters.content_image.matrix,
            # ratio in [0, 1], 0 means no noise, 1 means all noise
            RATIO_VALUE_OF_ORIGINAL_PROTOTYPE,
        )

        print(" --- Building Computations ---")

        self.nn_builder = NeuralNetBuilder(style_network, self.session_runner.session)

        ### Practically, we PASS the Content Image throught Graph
        # indicate content_image and the output layer of the Neural Network
        self.nn_builder.build_activations(
            c_image.matrix,
            model_design.network_design.output_layer,
        )

        # We have passed the User Content Image in the Network and we extracted the output Tensor
        # from the Output Layer (defined in NetworkDesign)
        # We will be leveraging this a_C tensor to measure Content Cost against
        # the Generated Image a_G

        self.nn_cost_builder = CostBuilder(
            NSTContentCostComputer.compute,
            NSTStyleCostComputer.compute,
        )

        # Content Image was passed through the graph, so we can get the activations
        self.nn_cost_builder.build_content_cost(
            self.nn_builder.a_C,  # we have the Content Image activations
            self.nn_builder.a_G,  # Generated Image activations
        )

        ### Practically, we PASS the Style Image throught Graph, and in
        # combination with Style Layers (and coefficients) we have defined, we
        # build the Computation Function for the Style Cost
        self.nn_builder.assign_input(s_image.matrix)

        # manually set the neurons attribute for each NSTStyleLayer
        # using the loaded cv model (which is a dict of layers)
        # the NSTStyleLayer ids attribute to query the dict
        for style_layer_id, nst_style_layer in model_design.network_design.style_layers:
            # for each selected Style Layer, simply copy reference of Layer from
            # Pretrained network Graph
            nst_style_layer.neurons = style_network[style_layer_id]
        # TODO obviously encapsulate the above code elsewhere

        # Build Style Cost Computation Function
        self.nn_cost_builder.build_style_cost(
            self.session_runner.session,
            model_design.network_design.style_layers,
        )

        # Since the nn_cost_builder has already built the Style and Content Cost
        # Computation Functions, now it is capable of building the 'Total Cost'
        # Total Cost can simply be: TC = alpha * Jc + beta * Js

        self.nn_cost_builder.build_cost(
            alpha=10,  # content cost weight (raw multiplier)
            beta=40,  # style cost weight (raw multiplier)
        )

        self.optimization = Optimization()
        self.optimization.optimize_against(self.nn_cost_builder.cost)

        ## Run Iterative Learning Algorithm ##

        print(" --- Preparing Iterative Learning Algorithm ---")

        # Take Input Content Image (as 3-color channel 4D tensor with means already subtracted)
        # and generate a random noise image as a starting for the Generated
        # Image (a_G)
        input_image = noisy_content_image_matrix

        # Initialize global variables (you need to run the session on the initializer)
        self.session_runner.run(tf.compat.v1.global_variables_initializer())

        ## Practically, PASS the Noisy Content Image through Graph
        # But out cost builder for example might not utilize the Graphs last Tensor output
        # since our Style Layers most probably are towards the middle layers of the pretrained network model

        # here basically we perform the first "iteration" outside the loop to
        # initialize the Generated Image (a_G) with the Noisy Content Image
        # and make the network produce the activations required for all the Cost Computing operations
        # and performing Weight Optimization (learning)
        self.session_runner.run(style_network["input"].assign(input_image))
        self.perform_nst(style_network)

    def perform_nst(self, style_network):
        print(" --- Running Iterative Algorithm ---")

        # Evaluation of Costs Frequency
        cost_eval_freq = 20
        i = 0
        self.time_started = time()

        while not self.nst_algorithm.parameters.termination_condition.satisfied:
            # We pass the Curernt Gen Image throguh the Graph and get the next iteration of gen Image
            generated_image = self.iterate(style_network)
            progress = self._progress(generated_image, completed_iterations=i + 1)
            # Evaluate Cost scalars every cost_eval_freq iters
            if i % cost_eval_freq == 0:
                self.Jt, self.Jc, self.Js = self._eval_cost()
                progress["metrics"].update(
                    {
                        "cost": self.Jt,
                        "content-cost": self.Jc,
                        "style-cost": self.Js,
                        "content-cost-weighted": self.nn_cost_builder.content_cost_weight
                        * self.Jc,
                        "style-cost-weighted": self.nn_cost_builder.style_cost_weight
                        * self.Js,
                    }
                )
                self._print_to_std(progress)
            if i % 20 == 0:
                self._notify_persistance(progress)
                self._print_to_std(progress)
            progress["metrics"]["duration"] = time() - self.time_started  # in seconds
            self._notify_progress(progress)
            i += 1

        try:
            self._notify_persistance(progress)
        except NameError as progress_not_evaluated_error:
            raise NoIterationsDoneError(
                "The algorithm did not iterate. Probably the "
                'f"{self.nst_algorithm.parameters.termination_condition}"'
                ' termination condition is too "strict."'
            ) from progress_not_evaluated_error

        print(" --- Finished Learning Algorithm :) ---")

    def iterate(self, image_model: t.Dict[str, Layer]):
        # Run the session on the train_step to minimize the total cost
        # This is our typical iterative learning loop / iteration, where the
        # weights are adjusted to minimize a cost / objective function
        self.session_runner.run([self.optimization.train_step])

        ## Practically, PASS the current version of the Generated Image through
        # the Graph
        # Compute the generated image by running the session on the current model['input']
        generated_image = self.session_runner.run(image_model["input"])
        return generated_image

    def _print_to_std(self, progress):
        weighted_Jc = self.nn_cost_builder.content_cost_weight * self.Jc
        weighted_Js = self.nn_cost_builder.style_cost_weight * self.Js
        iteration_index: int = progress["metrics"]["iterations"] - 1
        print(
            f" Iteration: {iteration_index}\n"
            f" Jc + Js = {self.Js + self.Jc}\n"
            f"  Total Cost     : {self.Jt}\n"
            f" a * Jc + b * Js = {weighted_Jc + weighted_Js}\n"
            f"  Weighted Content Cost : {weighted_Jc}\n"
            f"  Weighted Style Cost   : {weighted_Js}\n"
            f"   Content cost : {self.Jc}\n"
            f"   Style cost   : {self.Js}\n"
        )

    def _progress(self, generated_image, completed_iterations: int) -> Dict:
        return {
            "metrics": {
                "iterations": completed_iterations,  # number of iterations completed
            },
            "content_image_path": self.nst_algorithm.parameters.content_image.file_path,
            "style_image_path": self.nst_algorithm.parameters.style_image.file_path,
            "output_path": self.nst_algorithm.parameters.output_path,
            "matrix": generated_image,
        }

    def _notify_persistance(self, progress):
        self.persistance_subject.state = type("SubjectState", (), progress)
        self.persistance_subject.notify()

    def _notify_progress(self, progress):
        # set subject with the appropriate state to broadcast
        self.progress_subject.state = type("SubjectState", (), progress)
        # notify all observers/listeners that have 'subscribed'
        self.progress_subject.notify()

    def _eval_cost(self):
        """Evaluate Total (Style + Constent) Cost"""
        # pass cost objects in session to evaluate them
        Jt, Jc, Js = self.session_runner.run(
            [
                self.nn_cost_builder.cost,
                self.nn_cost_builder.content_cost,
                self.nn_cost_builder.style_cost,
            ]
        )
        return Jt, Jc, Js

    def _print_cost(self, iteration_index):
        weighted_Jc = self.nn_cost_builder.content_cost_weight * self.Jc
        weighted_Js = self.nn_cost_builder.style_cost_weight * self.Js
        print(
            f" Iteration: {iteration_index}\n"
            f" Jc + Js = {self.Js + self.Jc}\n"
            f"  Total Cost    : {self.Jt}\n"
            f" a * Jc + b * Js = {weighted_Jc + weighted_Js}\n"
            f"  Weighted Content Cost : {weighted_Jc}\n"
            f"  Weighted Style Cost   : {weighted_Js}\n"
            f"   Content cost : {self.Jc}\n"
            f"   Style cost   : {self.Js}\n"
        )


class NoIterationsDoneError(Exception):
    pass


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
        self.session.run(self.model["input"].assign(image))

    def build_activations(self, content_image, model_layer_id: str):
        self.assign_input(content_image)
        self._set_output(model_layer_id)
        self._setup_activations()

    def _set_output(self, model_layer_id: str):
        # Select the output tensor of a Neural Network layer
        self.output_neurons = self.model[model_layer_id]

    def _setup_activations(self):
        # Set a_C to be the hidden layer activation from the layer we have selected
        # Remember we have re-constructed the pretrained model Network by now.
        # And we have selected in the NetworkDesign a layer that will bee the
        # 'Output Layer'. Here we say that the Image Model will be modeling the
        # Content of the G Image as the output of that selected Output Layer.
        self.a_C = self.session.run(self.output_neurons)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer,
        # with G as input.
        self.a_G = self.output_neurons


@attr.s
class CostBuilder:
    compute_content_cost = attr.ib()
    compute_style_cost = attr.ib()

    # Total cost = alpha * J_content + beta * J_style
    cost = attr.ib(init=False, default=None)  # Total Cost
    content_cost = attr.ib(init=False, default=None)
    style_cost = attr.ib(init=False, default=None)

    # beta parameter (unormalized: does not ad to 1 with alpha)
    # normaly the style weight should sth like 4 factors bigger than the content weight
    content_cost_weight = attr.ib(init=False, default=10)
    style_cost_weight = attr.ib(init=False, default=40)

    def build_content_cost(self, content_image_activations, generated_image_activations):
        # Compute the content cost
        self.content_cost = self.compute_content_cost(
            content_image_activations, generated_image_activations
        )

    def build_style_cost(self, tf_session, style_layers):
        # Compute the style cost
        self.style_cost = self.compute_style_cost(tf_session, style_layers)

    def build_cost(self, **kwargs):  # alpha=10, beta=40):
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
        alpha = kwargs.get("alpha", self.content_cost)
        beta = kwargs.get("beta", self.style_cost)

        self.content_cost_weight = alpha
        self.style_cost_weight = beta

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
