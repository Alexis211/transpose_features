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
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from theano import tensor


from datastream import prepare_data

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def construct_model(activation_function, input_dim, hidden_dims, out_dim):
    # Construct the model
    x = tensor.fmatrix('features')
    y = tensor.imatrix('targets')
    y = y[:,0]

    mlp = MLP(activations=activation_function + [None],
              dims=[input_dim] + hidden_dims +
              [out_dim])

    activations = mlp.apply(x)

    cost = Softmax().categorical_cross_entropy(y, activations)

    # Initialize parameters
    mlp.weights_init = IsotropicGaussian(0.01)
    mlp.biases_init = Constant(0.001)
    mlp.initialize()

    return cost


def train_model(cost, train_stream, load_location=None, save_location=None):

    cost.name = "Cross_entropy"

    # Define the model
    model = Model(cost)

    # Load the parameters from a dumped model
    if load_location is not None:
        logger.info('Loading parameters...')
        model.set_param_values(load_parameter_values(load_location))

    cg = ComputationGraph(cost)
    step_rule = Scale(learning_rate=0.00001)
    algorithm = GradientDescent(cost=cost, step_rule=step_rule,
                                params=cg.parameters)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            DataStreamMonitoring([cost], train_stream,
                                 prefix='train', after_epoch=False, every_n_epochs=10),
            Printing(after_epoch=False, every_n_epochs=10)
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
    feat = 7129
    train_ex = 38

    # Build model
    cost = construct_model([Rectifier()], train_ex + 1, [30], 2)

    # Build datastream
    train_stream = prepare_data("train")

    # Train the model
    train_model(cost, train_stream, load_location=None, save_location=None)
