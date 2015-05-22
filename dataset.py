import numpy
import cPickle
import theano
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def fulldata_process(fulldata, train_set_size):
    fulldata_x = fulldata[:, :-1]
    fulldata_y = numpy.array(fulldata[:, -1], dtype='int32')

    # normalize
    fulldata_x = fulldata_x - fulldata_x.mean(axis=0, keepdims=True)
    fulldata_x_norms = numpy.sqrt((fulldata_x ** 2).sum(axis=0, keepdims=True))
    fulldata_x = fulldata_x / (fulldata_x_norms +
                               numpy.equal(fulldata_x_norms, 0))

    logger.info("Full dataset shape:", fulldata.shape)

    # valid_set_size = fulldata.shape[0] - train_set_size

    train_set_x = fulldata_x[0:train_set_size, :]
    train_set_y = fulldata_y[0:train_set_size]
    valid_set_x = fulldata_x[train_set_size:, :]
    valid_set_y = fulldata_y[train_set_size:]

    return train_set_x, train_set_y, valid_set_x, valid_set_y


def load_AMLALL(train_set_size=55):
    with open('data/AMLALL_full.pkl') as fh:
        fulldata = cPickle.load(fh)

    fulldata = numpy.array(fulldata, dtype=theano.config.floatX)
    numpy.random.shuffle(fulldata)

    return fulldata_process(fulldata, train_set_size)


def load_ARCENE():
    with open('data/arcene_full.pkl') as fh:
        fulldata = cPickle.load(fh)

    fulldata = numpy.array(fulldata, dtype=theano.config.floatX)
    return fulldata_process(fulldata, train_set_size=100)
