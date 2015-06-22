from theano import tensor
import theano
import numpy

from blocks.algorithms import Momentum, AdaDelta, RMSProp, Adam
from blocks.bricks import Rectifier, MLP, Softmax, Tanh, Bias
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise, apply_dropout

from datastream import RandomTransposeIt

step_rule_name = 'rmsprop'
learning_rate = 0.1
momentum = 0.9

if step_rule_name == 'adadelta':
    step_rule = AdaDelta()
elif step_rule_name == 'rmsprop':
    step_rule = RMSProp()
elif step_rule_name == 'momentum':
    step_rule_name = "mom%s,%s" % (repr(learning_rate), repr(momentum))
    step_rule = Momentum(learning_rate=learning_rate, momentum=momentum)
elif step_rule_name == 'adam':
    step_rule = Adam()
else:
    raise ValueError("No such step rule: " + step_rule_name)

ibatchsize = None
iter_scheme = RandomTransposeIt(ibatchsize, False, None, False)
valid_iter_scheme = RandomTransposeIt(ibatchsize, False, None, False)

w_noise_std = 0.03

r_dropout = 0.0
s_dropout = 0.0
i_dropout = 0.5

nparts = 10
part_r_proba = 0.4

reconstruction_penalty = 1

center_feats = True
normalize_feats = True
randomize_feats = False

train_on_valid = False

hidden_dims_0 = [5]
activation_functions_0 = [Tanh() for _ in hidden_dims_0]
hidden_dims_1 = []
activation_functions_1 = [Tanh() for _ in hidden_dims_1] + [None]
hidden_dims_2 = []
activation_functions_2 = [Tanh() for _ in hidden_dims_2]

n_inter = 8
inter_act_fun = Tanh()

dataset = 'ARCENE'
pt_freq = 10

param_desc = '%s-%s,%s,%d,%s-%dp%s-n%s-d%s,%s,%s-p%s-%s-%s-i%s' % (dataset,
                                    repr(hidden_dims_0),
                                    repr(hidden_dims_1),
                                    n_inter,
                                    repr(hidden_dims_2),
                                    nparts,
                                    repr(part_r_proba),
                                    repr(w_noise_std),
                                    repr(r_dropout), repr(s_dropout), repr(i_dropout),
                                    repr(reconstruction_penalty),
                                    ('C' if center_feats else'') +
                                    ('N' if normalize_feats else '') +
                                    ('W' if train_on_valid else '') +
                                    ('R' if randomize_feats else ''),
                                    step_rule_name,
                                    repr(ibatchsize))



class Model(object):
    def __init__(self, ref_data, output_dim):
        ref_data_sh = theano.shared(numpy.array(ref_data, dtype=numpy.float32), name='ref_data')

        # Construct the model
        j = tensor.lvector('j')
        x = tensor.fmatrix('x')
        y = tensor.ivector('y')

        last_outputs = []
        s_dropout_vars = []
        r_dropout_vars = []
        i_dropout_vars = []
        penalties = []

        for i in range(nparts):
            fs = numpy.random.binomial(1, part_r_proba, size=(ref_data.shape[1],))
            input_dim = int(fs.sum())

            fs_sh = theano.shared(fs)
            r = ref_data_sh[j, :][:, fs_sh.nonzero()[0]]

            mlp0 = MLP(activations=activation_functions_0,
                      dims=[input_dim] + hidden_dims_0, name='enc%d'%i)
            mlp0r = MLP(activations=[None], dims=[hidden_dims_0[-1], input_dim], name='dec%d'%i)
            mlp1 = MLP(activations=activation_functions_1,
                      dims=[hidden_dims_0[-1]] + hidden_dims_1 + [n_inter], name='inter_gen_%d'%i)
            mlp2 = MLP(activations=activation_functions_2 + [None],
                       dims=[n_inter] + hidden_dims_2 + [output_dim],
                       name='end_mlp_%d'%i)

            encod = mlp0.apply(r)
            rprime = mlp0r.apply(encod)
            inter_weights = mlp1.apply(encod)

            ibias = Bias(n_inter, name='inter_bias_%d'%i)
            inter = ibias.apply(tensor.dot(x, inter_weights))
            inter = inter_act_fun.apply(inter)

            out = mlp2.apply(inter)

            penalties.append(tensor.sqrt(((rprime - r)**2).sum(axis=1)).mean()[None])

            last_outputs.append(out)

            r_dropout_vars.append(r)
            s_dropout_vars = s_dropout_vars + (
                                    VariableFilter(bricks=[Tanh], name='output')
                                                  (ComputationGraph([inter_weights]))
                            )
            i_dropout_vars.append(inter)

            # Initialize parameters
            for brick in [mlp0, mlp0r, mlp1, mlp2, ibias]:
                brick.weights_init = IsotropicGaussian(0.01)
                brick.biases_init = Constant(0.001)
                brick.initialize()

        final = tensor.concatenate([x[:, :, None] for x in last_outputs], axis=2).mean(axis=2)

        cost = Softmax().categorical_cross_entropy(y, final)
        confidence = Softmax().apply(final)

        pred = final.argmax(axis=1)
        error_rate = tensor.neq(y, pred).mean()

        # apply regularization
        cg = ComputationGraph([cost, error_rate])

        if w_noise_std != 0:
            # - apply noise on weight variables
            weight_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, weight_vars, w_noise_std)

        if s_dropout != 0:
            cg = apply_dropout(cg, s_dropout_vars, s_dropout)
        if r_dropout != 0:
            cg = apply_dropout(cg, r_dropout_vars, r_dropout)
        if i_dropout != 0:
            cg = apply_dropout(cg, i_dropout_vars, i_dropout)

        [cost_reg, error_rate_reg] = cg.outputs

        cost_reg = cost_reg + reconstruction_penalty * tensor.concatenate(penalties, axis=0).sum()

        self.cost = cost
        self.cost_reg = cost_reg
        self.error_rate = error_rate
        self.error_rate_reg = error_rate_reg
        self.pred = pred
        self.confidence = confidence

