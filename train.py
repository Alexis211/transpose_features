#!/usr/bin/env python

import logging
import numpy as np
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

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    config = importlib.import_module('%s' % model_name)


def train_model(cost, error_rate, train_stream, valid_stream, load_location=None, save_location=None):

    cost.name = "cross_entropy"
    error_rate.name = 'error_rate'

    # Define the model
    model = Model(cost)

    # Load the parameters from a dumped model
    if load_location is not None:
        logger.info('Loading parameters...')
        model.set_param_values(load_parameter_values(load_location))

    cg = ComputationGraph(cost)
    algorithm = GradientDescent(cost=cost, step_rule=config.step_rule,
                                params=cg.parameters)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            TrainingDataMonitoring([cost, error_rate], prefix='train', every_n_epochs=1),
            DataStreamMonitoring([cost, error_rate], valid_stream, prefix='valid',
                                 after_epoch=False, every_n_epochs=5),
            Printing(every_n_epochs=1, after_epoch=False),
            Plot(document='tr_'+model_name+'_'+config.param_desc,
                 channels=[['train_cross_entropy', 'valid_cross_entropy'],
                           ['train_error_rate', 'valid_error_rate']],
                 every_n_epochs=1, after_epoch=False)
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
    train_stream = prepare_data("ARCENE", "train", config.iter_scheme)
    valid_stream = prepare_data("ARCENE", "valid", config.valid_iter_scheme)

    train_ex = train_stream.dataset.nitems

    # Build model
    cost, error_rate = config.construct_model(train_ex + 1, 2)

    # Train the model
    train_model(cost, error_rate, train_stream, valid_stream, load_location=None, save_location=None)
