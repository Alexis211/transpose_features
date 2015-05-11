import logging
import numpy
from fuel import config
from fuel.datasets import BinarizedMNIST
from fuel.transformers import Mapping, Batch, Filter, Transformer

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def prepare_data():
    # Construct data stream
    logger.info('Constructing data stream')

    dataset = [numpy.ones((20, 1000)), numpy.ones((20,))]
    features = dataset[0].T

    # Convert each sample x_i into p samples (p is the number of features)
    # where each new sample = feature_j concat x_i_j
    new_dataset = [numpy.zeros((20000, 21)), numpy.zeros((20000))]

    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            new_sample = numpy.concatenate(
                (features[i, :], features[i, j:j + 1]), axis=1)
            new_target = dataset[1][j]
            new_dataset[0][i * features.shape[1] + j, :] = new_sample
            new_dataset[1][i * features.shape[1] + j] = new_target
            print i * features.shape[1] + j

    return new_dataset

if __name__ == "__main__":
    data_set = prepare_data()
    print data_set
    print data_set[0].shape
    print data_set[1].shape
    print data_set[0][-1, :]
