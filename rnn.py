import logging
import numpy as np

from blocks.algorithms import GradientDescent, Momentum
from blocks.bricks import Tanh, Softmax, Linear
from blocks.bricks.recurrent import LSTM
from blocks.dump import load_parameter_values
from blocks.dump import MainLoopDumpManager
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from theano import tensor

from datastream import prepare_data, RandomTransposeIt, LogregOrderTransposeIt

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def construct_model(activation_function, input_dim, hidden_dim, out_dim):
    # Construct the model
    r = tensor.fmatrix('r')
    x = tensor.fmatrix('x')
    y = tensor.ivector('y')

    nx = x.shape[0]
    nj = x.shape[1]  # also is r.shape[0]
    nr = r.shape[1]

    # r is nj x nr
    # x is nx x nj
    # y is nx

    # r_rep is nx x nj x nr
    r_rep = r[None, :, :].repeat(axis=0, repeats=nx)
    # x3 is nx x nj x 1
    x3 = x[:, :, None]

    # concat is nx x nj x (nr + 1)
    concat = tensor.concatenate([r_rep, x3], axis=2)

    # Change concat from Batch x Time x Features to T X B x F
    rnn_input = concat.dimshuffle(1, 0, 2)

    linear = Linear(input_dim=input_dim, output_dim=4 * hidden_dim,
                    name="input_linear")
    lstm = LSTM(dim=hidden_dim, activation=activation_function,
                name="hidden_recurrent")
    top_linear = Linear(input_dim=hidden_dim, output_dim=out_dim,
                        name="out_linear")

    pre_rnn = linear.apply(rnn_input)
    states = lstm.apply(pre_rnn)[0]
    activations = top_linear.apply(states)
    activations = tensor.mean(activations, axis=0)

    cost = Softmax().categorical_cross_entropy(y, activations)

    pred = activations.argmax(axis=1)
    error_rate = tensor.neq(y, pred).mean()

    # Initialize parameters

    for brick in (linear, lstm, top_linear):
        brick.weights_init = IsotropicGaussian(0.01)
        brick.biases_init = Constant(0.)
        brick.initialize()

    return cost, error_rate


def train_model(cost, error_rate, train_stream, load_location=None, save_location=None):

    cost.name = "Cross_entropy"
    error_rate.name = 'Error_rate'

    # Define the model
    model = Model(cost)

    # Load the parameters from a dumped model
    if load_location is not None:
        logger.info('Loading parameters...')
        model.set_param_values(load_parameter_values(load_location))

    cg = ComputationGraph(cost)
    step_rule = Momentum(learning_rate=0.1, momentum=0.9)
    algorithm = GradientDescent(cost=cost, step_rule=step_rule,
                                params=cg.parameters)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            # DataStreamMonitoring([cost], test_stream, prefix='test',
            #                      after_epoch=False, every_n_epochs=10),
            DataStreamMonitoring([cost, error_rate], train_stream, prefix='train',
                                 after_epoch=False, every_n_epochs=10),
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
    train_ex = 100

    # Build model
    cost, error_rate = construct_model(Tanh(), train_ex + 1, 30, 2)

    # Build datastream
    train_stream = prepare_data("ARCENE", "train",
                                LogregOrderTransposeIt(10, True, 'model_param/logreg_param.pkl', 500))

    # Train the model
    train_model(cost, error_rate, train_stream, load_location=None, save_location=None)
