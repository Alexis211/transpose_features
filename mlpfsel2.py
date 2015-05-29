from theano import tensor
import theano
import numpy

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

w_noise_std = 1.0
r_noise_std = 1.0
r_dropout = 0.2
s_dropout = 0.9
i_dropout = 0.9

center_feats = True
normalize_feats = True
randomize_feats = False

hidden_dims = [70]
activation_functions = [Tanh() for _ in hidden_dims] + [None]

n_inter = 70
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
                                    ('C' if center_feats else'') +
                                    ('N' if normalize_feats else '') +
                                    ('R' if randomize_feats else ''),
                                    step_rule_name,
                                    ibatchsize,
                                    repr(input_cut_dim))



class Model(object):
    def __init__(self, ref_data, output_dim):
        input_dim = ref_data.shape[1]
        if input_cut_dim != None:
            input_dim = input_cut_dim

        ref_data_sh = theano.shared(numpy.array(ref_data, dtype=numpy.float32), name='ref_data')

        # Construct the model
        j = tensor.lvector('j')
        r = ref_data_sh[j, :][:, 0:input_dim]
        x = tensor.fmatrix('x')
        y = tensor.ivector('y')

        # input_dim must be nr
        mlp = MLP(activations=activation_functions,
                  dims=[input_dim] + hidden_dims + [n_inter], name='inter_gen')
        mlp2 = MLP(activations=[None], dims=[n_inter, 2], name='end_mlp')

        inter_weights = mlp.apply(r)

        if inter_bias == None:
            ibias = Bias(n_inter)
            ibias.biases_init = Constant(0)
            ibias.initialize()
            inter = ibias.apply(tensor.dot(x, inter_weights))
        else:
            inter = tensor.dot(x, inter_weights) - inter_bias
        inter = inter_act_fun.apply(inter)

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

        tmp_output_vars = list(set(VariableFilter(bricks=[Tanh], name='output')(ComputationGraph([inter_weights])))
                             - set([inter_weights]))
        print 'dropout', s_dropout, 'on', tmp_output_vars
        apply_dropout(cg, tmp_output_vars, s_dropout)

        apply_dropout(cg, [inter], i_dropout)

        [cost_reg, error_rate_reg] = cg.outputs

        self.cost = cost
        self.cost_reg = cost_reg
        self.error_rate = error_rate
        self.error_rate_reg = error_rate_reg
        self.pred = pred

