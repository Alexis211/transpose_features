#!/usr/bin/env python

import logging
import numpy
import sys
import importlib

from blocks.dump import load_parameter_values
from blocks.dump import MainLoopDumpManager
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent
from theano import tensor

from datastream import prepare_data
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
    model = Model(m.cost)

    # Load the parameters from a dumped model
    if load_location is not None:
        logger.info('Loading parameters...')
        model.set_param_values(load_parameter_values(load_location))

    cg = ComputationGraph(m.cost_reg)
    algorithm = GradientDescent(cost=m.cost_reg, step_rule=config.step_rule,
                                params=cg.parameters)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            TrainingDataMonitoring(
                [m.cost_reg, m.error_rate_reg], prefix='train', every_n_epochs=1*config.pt_freq),
            DataStreamMonitoring([m.cost, m.error_rate], valid_stream, prefix='valid',
                                 after_epoch=False, every_n_epochs=5*config.pt_freq),
            Printing(every_n_epochs=1*config.pt_freq, after_epoch=False),
            Plot(document='tr_'+model_name+'_'+config.param_desc,
                 channels=[['train_cost_reg', 'valid_cost'],
                           ['train_error_rate_reg', 'valid_error_rate']],
                 every_n_epochs=1*config.pt_freq, after_epoch=False)
        ]
    )
    main_loop.run()

    # Save the main loop
    if save_location is not None:
        logger.info('Saving the main loop...')
        dump_manager = MainLoopDumpManager(save_location)
        dump_manager.dump(main_loop)
        logger.info('Saved')


if __name__ == "__main__":
    # Build datastream
    ref_data, train_stream, valid_stream, test_stream = prepare_data(config)

    # Build model
    m = config.Model(ref_data, 2)
    m.cost.name = 'cost'
    m.cost_reg.name = 'cost_reg'
    m.error_rate.name = 'error_rate'
    m.error_rate_reg.name = 'error_rate_reg'
    m.pred.name = 'pred'
    m.confidence.name = 'confidence'

    # Train the model
    saveloc = 'model_data/%s-%s' % (model_name, config.param_desc)
    train_model(m, train_stream, valid_stream,
                load_location=None, save_location=saveloc)

    # Produce output on test file
    if test_stream != None:
        vec = numpy.zeros((0, 2))
        for v in Apply([m.pred, m.confidence], ['pred', 'confidence'], test_stream):
            vec = numpy.concatenate([vec, v['confidence']], axis=0)
        numpy.savetxt(saveloc+'.predict', vec[:, 1], fmt='%.6e')
