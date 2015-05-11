import logging
import numpy

from blocks.algorithms import GradientDescent, Scale, StepRule
from blocks.bricks import Rectifier, MLP, Softmax
from blocks.dump import load_parameter_values
from blocks.dump import MainLoopDumpManager
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from fuel.transformers import Batch
from theano import tensor

from datastream import prepare_data

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def construct_model(activation_function, hidden_dims, out_dim):
    # Construct the model
    x = tensor.lmatrix('features')
    y = tensor.lvector('targets')

    mlp = MLP(activations=activation_function + [None],
                 dims=[x.shape[1]] + hidden_dims +
                 [out_dim])

    activations = mlp.apply(x)

    cost = Softmax().categorical_cross_entropy(y, activations)

    # Initialize parameters
    mlp.weights_init = IsotropicGaussian(0.01)
    mlp.biases_init = Constant(0.001)
    mlp.initialize()

    return cost


def train_model(cost, train_stream, valid_stream, load_location=None, save_location=None):

    # Define the model
    model = Model(cost)

    # Load the parameters from a dumped model
    if load_location is not None:
        logger.info('Loading parameters...')
        model.set_param_values(load_parameter_values(load_location))

    cg = ComputationGraph(cost)
    step_rule = Scale(learning_rate=0.01)
    algorithm = GradientDescent(cost=cost, step_rule=step_rule,
                                params=cg.parameters)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            DataStreamMonitoring([cost], valid_stream,
                                 prefix='valid_all', every_n_batches=1000),
            Printing(every_n_batches=1000)
        ]
    )
    main_loop.run()

    # Save the main loop
    if save_location is not None:
        logger.info('Saving the main loop...')
        dump_manager = MainLoopDumpManager(save_location)
        dump_manager.dump(main_loop)
        logger.info('Saved')


if __name__ == "__main__":

    # Build model
    cost = construct_model([Rectifier()], [100], 2)

    # TODO Prepare data
    train_stream = prepare_data("train")
    valid_stream= prepare_data("valid")

    # Train the model
    train_model(cost, train_stream, valid_stream,
                load_location=None, save_location=None)
