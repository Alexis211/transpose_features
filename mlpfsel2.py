from theano import tensor

from blocks.algorithms import Momentum, AdaDelta, RMSProp
from blocks.bricks import Rectifier, MLP, Softmax, Tanh
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise, apply_dropout

from datastream import RandomTransposeIt

step_rule_name = 'adadelta'

# step_rule = Momentum(learning_rate=learning_rate, momentum=momentum)
if step_rule_name == 'adadelta':
    step_rule = AdaDelta()
if step_rule_name == 'rmsprop':
    step_rule = RMSProp()

ibatchsize = 100
iter_scheme = RandomTransposeIt(ibatchsize, True, None, True)
valid_iter_scheme = iter_scheme

w_noise_std = 1.5
r_noise_std = 0.4
r_dropout = 0.2
i_dropout = 0.8

randomize_feats = False

hidden_dims = []
activation_functions = [Tanh() for _ in hidden_dims] + [Tanh()]

n_inter = 10
inter_bias = -9
inter_act_fun = Tanh()

param_desc = '%s-%d-%s-n%s,%s-d%s,%s-%s-%s-i%d' % (repr(hidden_dims),
                                    n_inter,
                                    repr(inter_bias),
                                    repr(w_noise_std),
                                    repr(r_noise_std),
                                    repr(r_dropout),
                                    repr(i_dropout),
                                    'R' if randomize_feats else '',
                                    step_rule_name,
                                    ibatchsize)

pt_freq = 11

def construct_model(input_dim, output_dim):
    # Construct the model
    r = tensor.fmatrix('r')
    x = tensor.fmatrix('x')
    y = tensor.ivector('y')

    # input_dim must be nr
    mlp = MLP(activations=activation_functions,
              dims=[input_dim] + hidden_dims + [n_inter], name='inter_gen')
    mlp2 = MLP(activations=[None], dims=[n_inter, 2], name='end_mlp')

    inter_weights = mlp.apply(r)

    inter = inter_act_fun.apply(tensor.dot(x, inter_weights) - inter_bias)

    final = mlp2.apply(inter)

    cost = Softmax().categorical_cross_entropy(y, final).mean()

    pred = final.argmax(axis=1)
    error_rate = tensor.neq(y, pred).mean()

    # Initialize parameters
    for brick in [mlp, mlp2]:
        brick.weights_init = IsotropicGaussian(0.01)
        brick.biases_init = Constant(0.001)
        brick.initialize()

    # apply noise
    cg = ComputationGraph([cost, error_rate])
    weight_vars = VariableFilter(roles=[WEIGHT])(cg)
    apply_noise(cg, weight_vars, w_noise_std)
    apply_noise(cg, [r], r_noise_std)
    apply_dropout(cg, [inter_weights], r_dropout)
    apply_dropout(cg, [inter], i_dropout)
    [cost_reg, error_rate_reg] = cg.outputs

    return cost_reg, error_rate_reg, cost, error_rate

