from theano import tensor

from blocks.algorithms import Momentum, AdaDelta
from blocks.bricks import Tanh, Softmax, Linear, MLP
from blocks.bricks.recurrent import LSTM
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise, apply_dropout

from datastream import LogregOrderTransposeIt, RandomTransposeIt


activation_function = Tanh()

mlp_hidden_dims = [10]
lstm_hidden_dim = 10
noise_std = 0.01
dropout = 0

num_feats = 100
use_ensembling = False

# step_rule = Momentum(learning_rate=0.01, momentum=0.9)
step_rule = AdaDelta()

# iter_scheme = LogregOrderTransposeIt(10, True, 'model_param/logreg_param.pkl', 500)
iter_scheme = RandomTransposeIt(10, True, num_feats, True)
valid_iter_scheme = RandomTransposeIt(10, True, None if use_ensembling else num_feats, True)

param_desc = '%s-%d-%s-%s-%s' % (repr(mlp_hidden_dims),
                              lstm_hidden_dim,
                              repr(noise_std),
                              repr(dropout),
                              'E' if use_ensembling else 'i')


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
    mlp_input = concat.dimshuffle(1, 0, 2)

    if use_ensembling:
        # Split time dimension into batches of size num_feats
        # Join that dimension with the B dimension
        ens_shape = (num_feats,
                     mlp_input.shape[0]/num_feats,
                     mlp_input.shape[1])
        mlp_input = mlp_input.reshape(ens_shape + (input_dim+1,))
        mlp_input = mlp_input.reshape((ens_shape[0], ens_shape[1] * ens_shape[2], input_dim+1))

    mlp = MLP(dims=[input_dim+1] + mlp_hidden_dims,
              activations=[activation_function for _ in mlp_hidden_dims],
              name='mlp')

    lstm_bot_linear = Linear(input_dim=mlp_hidden_dims[-1], output_dim=4 * lstm_hidden_dim,
                    name="lstm_input_linear")
    lstm = LSTM(dim=lstm_hidden_dim, activation=activation_function,
                name="hidden_recurrent")
    lstm_top_linear = Linear(input_dim=lstm_hidden_dim, output_dim=out_dim,
                        name="out_linear")

    rnn_input = mlp.apply(mlp_input)

    pre_rnn = lstm_bot_linear.apply(rnn_input)
    states = lstm.apply(pre_rnn)[0]
    activations = lstm_top_linear.apply(states)

    if use_ensembling:
        activations = activations.reshape(ens_shape + (out_dim,))
        # Unsplit batches (ensembling)
        activations = tensor.mean(activations, axis=1)

    # Mean over time
    activations = tensor.mean(activations, axis=0)

    cost = Softmax().categorical_cross_entropy(y, activations)

    pred = activations.argmax(axis=1)
    error_rate = tensor.neq(y, pred).mean()

    # Initialize parameters
    for brick in (mlp, lstm_bot_linear, lstm, lstm_top_linear):
        brick.weights_init = IsotropicGaussian(0.01)
        brick.biases_init = Constant(0.)
        brick.initialize()

    # apply noise
    cg = ComputationGraph([cost, error_rate])
    noise_vars = VariableFilter(roles=[WEIGHT])(cg)
    apply_noise(cg, noise_vars, noise_std)
    apply_dropout(cg, [mlp_input, rnn_input], dropout)
    [cost, error_rate] = cg.outputs

    return cost, error_rate


