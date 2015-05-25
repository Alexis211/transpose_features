from theano import tensor

from blocks.algorithms import Momentum, AdaDelta
from blocks.bricks import Tanh, Softmax, Linear
from blocks.bricks.recurrent import LSTM
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise

from datastream import LogregOrderTransposeIt, RandomTransposeIt

activation_function = Tanh()

hidden_dim = 10

noise_std = 0.01


# step_rule = Momentum(learning_rate=0.01, momentum=0.9)
step_rule = AdaDelta()

# iter_scheme = LogregOrderTransposeIt(10, True, 'model_param/logreg_param.pkl', 500)
iter_scheme = RandomTransposeIt(10, True, 100, True)
valid_iter_scheme = iter_scheme

def construct_model(input_dim, out_dim):
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

    # apply noise
    cg = ComputationGraph([cost, error_rate])
    noise_vars = VariableFilter(roles=[WEIGHT])(cg)
    apply_noise(cg, noise_vars, noise_std)
    [cost, error_rate] = cg.outputs

    return cost, error_rate


