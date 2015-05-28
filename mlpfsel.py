from theano import tensor

from blocks.algorithms import Momentum, AdaDelta
from blocks.bricks import Rectifier, MLP, Softmax, Tanh
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise

from datastream import RandomTransposeIt


learning_rate = 0.00001
momentum = 0.9
# step_rule = Momentum(learning_rate=learning_rate, momentum=momentum)
step_rule = AdaDelta()

ibatchsize = 100
iter_scheme = RandomTransposeIt(ibatchsize, True, None, True)
valid_iter_scheme = iter_scheme

noise_std = 0.1

randomize_feats = False

hidden_dims = [2, 2]
activation_functions = [Tanh() for _ in hidden_dims] + [Tanh()]

param_desc = '%s-%s-%s-mom%s,%s-i%d' % (repr(hidden_dims),
                                    repr(noise_std),
                                    'R' if randomize_feats else '',
                                    repr(learning_rate),
                                    repr(momentum),
                                    ibatchsize)

pt_freq = 11

def construct_model(input_dim, output_dim):
    # Construct the model
    r = tensor.fmatrix('r')
    x = tensor.fmatrix('x')
    y = tensor.ivector('y')

    # input_dim must be nr
    mlp = MLP(activations=activation_functions,
              dims=[input_dim] + hidden_dims + [2])

    weights = mlp.apply(r)

    final = tensor.dot(x, weights)

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

