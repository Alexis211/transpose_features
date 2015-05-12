import os
import logging
import numpy
from fuel import config

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme


logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def prepare_data(which_set):
    # Construct data stream
    logger.info('Building data stream...')

    dataset = H5PYDataset(os.path.join(
        config.data_path, 'Leukemia/leukemia_data.hdf5'), which_set=which_set, load_in_memory=True)
    return dataset


if __name__ == "__main__":
        # Test
    dataset = prepare_data("train")
    stream = DataStream(dataset, iteration_scheme=SequentialScheme(
        dataset.num_examples, 1))
    print next(stream.get_epoch_iterator())
