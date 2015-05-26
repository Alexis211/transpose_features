from theano import tensor

from blocks.algorithms import Momentum, AdaDelta
from blocks.bricks import Rectifier, MLP, Softmax, Tanh
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise

from datastream import RandomTransposeIt


# step_rule = Momentum(learning_rate=0.01, momentum=0.9)
step_rule = AdaDelta()

iter_scheme = RandomTransposeIt(None, True, None, True)
valid_iter_scheme = iter_scheme

noise_std = 0.1

randomize_feats = True

hidden_dims = [101]
activation_functions = [Tanh() for _ in hidden_dims] + [None]

param_desc = '%s-%s-%s' % (repr(hidden_dims), repr(noise_std), 'R' if randomize_feats else '')

pt_freq = 11

def construct_model(input_dim, output_dim):
    # Construct the model
    r = tensor.fmatrix('r')
    x = tensor.fmatrix('x')
    y = tensor.ivector('y')

    nx = x.shape[0]
    nj = x.shape[1]  # also is r.shape[0]
    nr = r.shape[1]

    # r is nj x nr
    # x is nx x nj
    # y is nx x 1

    # r_rep is nx x nj x nr
    r_rep = r[None, :, :].repeat(axis=0, repeats=nx)
    # x3 is nx x nj x 1
    x3 = x[:, :, None]

    # concat is nx x nj x (nr + 1)
    concat = tensor.concatenate([r_rep, x3], axis=2)
    mlp_input = concat.reshape((nx * nj, nr + 1))

    # input_dim must be nr
    mlp = MLP(activations=activation_functions,
              dims=[input_dim+1] + hidden_dims + [output_dim])

    activations = mlp.apply(mlp_input)

    act_sh = activations.reshape((nx, nj, output_dim))
    final = act_sh.mean(axis=1)

    cost = Softmax().categorical_cross_entropy(y, final).mean()

    pred = final.argmax(axis=1)
    error_rate = tensor.neq(y, pred).mean()

    # Initialize parameters
    for brick in [mlp]:
        brick.weights_init = IsotropicGaussian(0.01)
        brick.biases_init = Constant(0.001)
        brick.initialize()

    # apply noise
    cg = ComputationGraph([cost, error_rate])
    noise_vars = VariableFilter(roles=[WEIGHT])(cg)
    apply_noise(cg, noise_vars, noise_std)
    [cost, error_rate] = cg.outputs

    return cost, error_rate

