import os
import logging
import numpy
from fuel import config

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Mapping


logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def _concat(sample):
    feat = sample[0]
    target = sample[1]
    new_feat = numpy.dot(numpy.ones((feat.shape[1], 1)), feat)
    new_feat = numpy.concatenate((new_feat, feat.T), axis=1).astype(numpy.float32)
    return (new_feat, target.T)


def prepare_data(which_set):
    # Construct data stream
    logger.info('Building data stream...')

    dataset = H5PYDataset(os.path.join(
        config.data_path, 'Leukemia/leukemia_transpose.hdf5'), which_set=which_set, load_in_memory=True)
    stream = DataStream(dataset, iteration_scheme=ShuffledScheme(
        dataset.num_examples, 1))
    stream = Mapping(stream, _concat)

    return stream


if __name__ == "__main__":
        # Test
    stream = prepare_data("train")
    print next(stream.get_epoch_iterator())
