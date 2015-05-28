from theano import tensor

from blocks.algorithms import Momentum, AdaDelta, RMSProp
from blocks.bricks import Rectifier, MLP, Softmax, Tanh, Bias
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

w_noise_std = 0.3
r_noise_std = 0.3
r_dropout = 0.2
s_dropout = 0.5
i_dropout = 0.8

randomize_feats = False

hidden_dims = [10]
activation_functions = [Tanh() for _ in hidden_dims] + [None]

n_inter = 16
inter_bias = None   # -5
inter_act_fun = Tanh()

input_cut_dim = None # 800
dataset = 'ARCENE'
pt_freq = 10

param_desc = '%s-%s-%d-%s-n%s,%s-d%s,%s,%s-%s-%s-i%d-cut%s' % (dataset,
                                    repr(hidden_dims),
                                    n_inter,
                                    repr(inter_bias),
                                    repr(w_noise_std),
                                    repr(r_noise_std),
                                    repr(r_dropout),
                                    repr(s_dropout),
                                    repr(i_dropout),
                                    'R' if randomize_feats else '',
                                    step_rule_name,
                                    ibatchsize,
                                    repr(input_cut_dim))



def construct_model(input_dim, output_dim):
    if input_cut_dim != None:
        input_dim = input_cut_dim

    # Construct the model
    r = tensor.fmatrix('r')[:, 0:input_dim]
    x = tensor.fmatrix('x')
    y = tensor.ivector('y')

    # input_dim must be nr
    mlp = MLP(activations=activation_functions,
              dims=[input_dim] + hidden_dims + [n_inter], name='inter_gen')
    mlp2 = MLP(activations=[None], dims=[n_inter, 2], name='end_mlp')
    to_init = [mlp, mlp2]

    inter_weights = mlp.apply(r)

    if inter_bias == None:
        ibias = Bias(n_inter)
        to_init.append(ibias)
        inter = inter_act_fun.apply(ibias.apply(tensor.dot(x, inter_weights)))
    else:
        inter = inter_act_fun.apply(tensor.dot(x, inter_weights) - inter_bias)

    final = mlp2.apply(inter)

    cost = Softmax().categorical_cross_entropy(y, final).mean()

    pred = final.argmax(axis=1)
    error_rate = tensor.neq(y, pred).mean()

    # Initialize parameters
    for brick in to_init:
        brick.weights_init = IsotropicGaussian(0.01)
        brick.biases_init = Constant(0.001)
        brick.initialize()

    # apply noise
    cg = ComputationGraph([cost, error_rate])

    weight_vars = VariableFilter(roles=[WEIGHT])(cg)
    apply_noise(cg, weight_vars, w_noise_std)

    apply_noise(cg, [r], r_noise_std)

    apply_dropout(cg, [inter_weights], r_dropout)

    tmp_output_vars = list(set(VariableFilter(bricks=[Tanh], name='output')(ComputationGraph([inter_weights])))
                         - set([inter_weights]))
    print 'dropout', s_dropout, 'on', tmp_output_vars
    apply_dropout(cg, tmp_output_vars, s_dropout)

    apply_dropout(cg, [inter], i_dropout)

    [cost_reg, error_rate_reg] = cg.outputs

    return cost_reg, error_rate_reg, cost, error_rate

