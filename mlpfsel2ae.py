from theano import tensor
import theano
import numpy

from theano.tensor.shared_randomstreams import RandomStreams

from blocks.algorithms import Momentum, AdaDelta, RMSProp
from blocks.bricks import Rectifier, MLP, Softmax, Tanh, Bias
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise, apply_dropout

from datastream import RandomTransposeIt

import ber as balanced_error_rate



step_rule_name = 'adadelta'
learning_rate = 0.1
momentum = 0.
decay_rate = 0.9

if step_rule_name == 'adadelta':
    step_rule = AdaDelta(decay_rate=decay_rate)
    step_rule_name = 'adadelta%s' % repr(decay_rate)
elif step_rule_name == 'rmsprop':
    step_rule = RMSProp()
elif step_rule_name == 'momentum':
    step_rule_name = "mom%s,%s" % (repr(learning_rate), repr(momentum))
    step_rule = Momentum(learning_rate=learning_rate, momentum=momentum)
else:
    raise ValueError("No such step rule: " + step_rule_name)

ibatchsize = None
iter_scheme = RandomTransposeIt(ibatchsize, False, None, False)
valid_iter_scheme = RandomTransposeIt(ibatchsize, False, None, False)

r_noise_std = 0.01
w_noise_std = 0.00
r_dropout = 0.0
x_dropout = 0.0
s_dropout = 0.0
i_dropout = 0.0
a_dropout = 0.0

s_l1pen = 0.02
i_l1pen = 0.00
a_l1pen = 0.000

ae_dims = [100, 100]
ae_f_noise_std = 0.02
ae_l1_pen = 0.01
ae_activations = [Tanh() for _ in ae_dims]

center_feats = True
normalize_feats = True
randomize_feats = False

train_on_valid = False

hidden_dims = []
activation_functions = [Tanh() for _ in hidden_dims] + [None]
hidden_dims_2 = []
activation_functions_2 = [Tanh() for _ in hidden_dims_2]

n_inter = 10
inter_bias = None   # -5
inter_act_fun = Tanh()

dataset = 'ARCENE'
pt_freq = 10

param_desc = '%s-%s,%d,%s-%s-n%s,%s-d%s,%s,%s,%s,%s-L1:%s,%s,%s-AE%sn%sp%s-%s-%s-i%s' % (dataset,
                                    repr(hidden_dims),
                                    n_inter,
                                    repr(hidden_dims_2),
                                    repr(inter_bias),
                                    repr(r_noise_std), repr(w_noise_std),
                                    repr(r_dropout),
                                    repr(x_dropout),
                                    repr(s_dropout),
                                    repr(i_dropout),
                                    repr(a_dropout),
                                    repr(s_l1pen), repr(i_l1pen), repr(a_l1pen),
                                    repr(ae_dims), repr(ae_f_noise_std), repr(ae_l1_pen),
                                    ('C' if center_feats else'') +
                                    ('N' if normalize_feats else '') +
                                    ('W' if train_on_valid else '') +
                                    ('R' if randomize_feats else ''),
                                    step_rule_name,
                                    repr(ibatchsize))



class Model(object):
    def __init__(self, ref_data, output_dim):
        input_dim = ref_data.shape[1]
        ref_data_sh = theano.shared(numpy.array(ref_data, dtype=numpy.float32), name='ref_data')

        rng = RandomStreams()

        ae_bricks = []
        ae_input = ref_data_sh
        ae_costs = []
        for i, (idim, odim) in enumerate(zip([input_dim] + ae_dims[:-1], ae_dims)):
            ae_mlp = MLP(activations=[ae_activations[i]],
                         dims=[idim, odim],
                         name='enc%i'%i)
            enc = ae_mlp.apply(ae_input)
            enc_n = ae_mlp.apply(ae_input + rng.normal(size=ae_input.shape, std=ae_f_noise_std))
            ae_mlp_dec = MLP(activations=[ae_activations[i]],
                             dims=[odim, idim],
                             name='dec%i'%i)
            dec = ae_mlp_dec.apply(enc_n)

            cost = tensor.sqrt(((ae_input - dec) ** 2).sum(axis=1)).mean() + \
                        ae_l1_pen * abs(enc).sum(axis=1).mean()
            ae_costs.append(cost)

            ae_input = enc
            ae_bricks = ae_bricks + [ae_mlp, ae_mlp_dec]

        self.ae_costs = ae_costs

        ref_data_enc = ae_input

        # Construct the model
        j = tensor.lvector('j')
        r = ref_data_enc[j, :]
        x = tensor.fmatrix('x')
        y = tensor.ivector('y')

        # input_dim must be nr
        mlp = MLP(activations=activation_functions,
                  dims=[ae_dims[-1]] + hidden_dims + [n_inter], name='inter_gen')
        mlp2 = MLP(activations=activation_functions_2 + [None],
                   dims=[n_inter] + hidden_dims_2 + [output_dim],
                   name='end_mlp')

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

        cost = Softmax().categorical_cross_entropy(y, final)
        confidence = Softmax().apply(final)

        pred = final.argmax(axis=1)
        # error_rate = tensor.neq(y, pred).mean()
        ber = balanced_error_rate.ber(y, pred)

        # Initialize parameters
        for brick in ae_bricks + [mlp, mlp2]:
            brick.weights_init = IsotropicGaussian(0.01)
            brick.biases_init = Constant(0.001)
            brick.initialize()

        # apply regularization
        cg = ComputationGraph([cost, ber])

        if r_dropout != 0:
            # - dropout on input vector r : r_dropout
            cg = apply_dropout(cg, [r], r_dropout)

        if x_dropout != 0:
            cg = apply_dropout(cg, [x], x_dropout)

        if s_dropout != 0:
            # - dropout on intermediate layers of first mlp : s_dropout
            s_dropout_vars = list(set(VariableFilter(bricks=[Tanh], name='output')
                                                     (ComputationGraph([inter_weights])))
                                 - set([inter_weights]))
            cg = apply_dropout(cg, s_dropout_vars, s_dropout)

        if i_dropout != 0:
            # - dropout on input to second mlp : i_dropout
            cg = apply_dropout(cg, [inter], i_dropout)

        if a_dropout != 0:
            # - dropout on hidden layers of second mlp : a_dropout
            a_dropout_vars = list(set(VariableFilter(bricks=[Tanh], name='output')
                                                     (ComputationGraph([final])))
                                 - set([inter_weights]) - set(s_dropout_vars))
            cg = apply_dropout(cg, a_dropout_vars, a_dropout)

        if r_noise_std != 0:
            cg = apply_noise(cg, [r], r_noise_std)

        if w_noise_std != 0:
            # - apply noise on weight variables
            weight_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, weight_vars, w_noise_std)

        [cost_reg, ber_reg] = cg.outputs
        
        if s_l1pen != 0:
            s_weights = VariableFilter(bricks=mlp.linear_transformations, roles=[WEIGHT])(cg)
            cost_reg = cost_reg + s_l1pen * sum(abs(w).sum() for w in s_weights)
        if i_l1pen != 0:
            cost_reg = cost_reg + i_l1pen * abs(inter).sum()
        if a_l1pen != 0:
            a_weights = VariableFilter(bricks=mlp2.linear_transformations, roles=[WEIGHT])(cg)
            cost_reg = cost_reg + a_l1pen * sum(abs(w).sum() for w in a_weights)


        self.cost = cost
        self.cost_reg = cost_reg
        self.ber = ber
        self.ber_reg = ber_reg
        self.pred = pred
        self.confidence = confidence

