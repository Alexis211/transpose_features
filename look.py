import numpy as np

from blocks.dump import load_parameter_values
from blocks.graph import ComputationGraph
from blocks.model import Model

from rnn import construct_model

cost, error = construct_model(101, 2)
model = Model(cost)


model.set_param_values(load_parameter_values("trained_rnn_classic/params.npz"))

print(ComputationGraph(cost).parameters)
# wh = ComputationGraph(cost).parameters[0].get_value()
# w = ComputationGraph(cost).parameters[1].get_value()
# w_lookup = ComputationGraph(cost).parameters[3].get_value()

# eig = np.zeros((unit, 0))
# for i in range(module):
#     whi = wh[unit * i:unit * (i + 1), unit * i:unit * (i + 1)]
#     eigi = np.linalg.eig(whi)[0]
#     eigi = np.sort(np.absolute(eigi.reshape((unit, 1))), axis=0)[::-1]
#     eig = np.concatenate((eig, eigi), axis=1)

# print(eig)
# matshow(eig, cmap=cm.gray)
# matshow(wh, cmap=cm.gray)
# show()
# return
