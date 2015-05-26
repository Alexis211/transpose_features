from theano import tensor

from blocks.algorithms import Momentum
from blocks.bricks import Rectifier, MLP, Softmax, Tanh
from blocks.initialization import IsotropicGaussian, Constant

from datastream import RandomTransposeIt


step_rule = Momentum(learning_rate=0.01, momentum=0.9)

iter_scheme = RandomTransposeIt(10, True, 100, True)
valid_iter_scheme = iter_scheme

activation_function = [Tanh()]

hidden_dims = [30]


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

    # input_dim must be nr+1
    mlp = MLP(activations=activation_function + [None],
              dims=[input_dim] + hidden_dims + [output_dim])

    activations = mlp.apply(mlp_input)

    act_sh = activations.reshape((nx, nj, output_dim))
    final = act_sh.mean(axis=1)

    cost = Softmax().categorical_cross_entropy(y, final)

    pred = final.argmax(axis=1)
    error_rate = tensor.neq(y, pred).mean()

    # Initialize parameters
    mlp.weights_init = IsotropicGaussian(0.01)
    mlp.biases_init = Constant(0.001)
    mlp.initialize()

    return cost, error_rate

