#!/usr/bin/env python

import logging
import numpy
import sys
import importlib

# from blocks.dump import load_parameter_values
# from blocks.dump import MainLoopDumpManager
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extras.extensions.plot import Plot
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent
from theano import tensor

from datastream import prepare_data, NoData
from apply_model import Apply

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    config = importlib.import_module('%s' % model_name)



def train_model(m, train_stream, valid_stream, load_location=None, save_location=None):

    # Define the model
    model = Model(m.cost_reg)

    ae_excl_vars = set()
    if hasattr(m, 'ae_costs'):
        for i, cost in enumerate(m.ae_costs):
            print "Trianing stacked AE layer", i+1
            # train autoencoder component separately
            cost.name = 'ae_cost%d'%i

            cg = ComputationGraph(cost)
            params = set(cg.parameters) - ae_excl_vars
            ae_excl_vars = ae_excl_vars | params

            algorithm = GradientDescent(cost=cost, step_rule=config.step_rule, params=list(params))
            main_loop = MainLoop(
                data_stream=NoData(train_stream),
                algorithm=algorithm,
                extensions=[
                    TrainingDataMonitoring([cost], prefix='train', every_n_epochs=1),
                    Printing(every_n_epochs=1),
                    FinishAfter(every_n_epochs=1000),
                ]
            )
            main_loop.run()

    cg = ComputationGraph(m.cost_reg)
    params = list(set(cg.parameters) - ae_excl_vars)
    algorithm = GradientDescent(cost=m.cost_reg, step_rule=config.step_rule,
                                params=params)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            TrainingDataMonitoring(
                [m.cost_reg, m.ber_reg, m.cost, m.ber],
                prefix='train', every_n_epochs=1*config.pt_freq),
            DataStreamMonitoring([m.cost, m.ber], valid_stream, prefix='valid',
                                 after_epoch=False, every_n_epochs=5*config.pt_freq),

            Printing(every_n_epochs=1*config.pt_freq, after_epoch=False),
            Plot(document='tr_'+model_name+'_'+config.param_desc,
                 channels=[['train_cost', 'train_cost_reg', 'valid_cost'],
                           ['train_ber', 'train_ber_reg', 'valid_ber']],
                 server_url='http://eos21:4201',
                 every_n_epochs=1*config.pt_freq, after_epoch=False),

            FinishAfter(every_n_epochs=10000)
        ]
    )
    main_loop.run()


if __name__ == "__main__":
    # Build datastream
    ref_data, train_stream, valid_stream, test_stream = prepare_data(config)

    # Build model
    m = config.Model(ref_data, 2)
    m.cost.name = 'cost'
    m.cost_reg.name = 'cost_reg'
    m.ber.name = 'ber'
    m.ber_reg.name = 'ber_reg'
    m.pred.name = 'pred'
    m.confidence.name = 'confidence'

    # Train the model
    saveloc = 'model_data/%s-%s' % (model_name, config.param_desc)
    train_model(m, train_stream, valid_stream,
                load_location=None, save_location=None)

    # Produce output on test file
    if test_stream != None:
        vec = numpy.zeros((0, 2))
        for v in Apply([m.pred, m.confidence], ['pred', 'confidence'], test_stream):
            vec = numpy.concatenate([vec, v['confidence']], axis=0)
        numpy.savetxt(saveloc+'.predict', vec[:, 1], fmt='%.6e')
