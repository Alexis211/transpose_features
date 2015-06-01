import numpy
import cPickle
import theano
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)



def load_AMLALL(train_set_size=55):
    with open('data/AMLALL_full.pkl') as fh:
        fulldata = cPickle.load(fh)

    fulldata = numpy.array(fulldata, dtype=theano.config.floatX)
    numpy.random.shuffle(fulldata)

    fulldata_x = fulldata[:, :-1]
    fulldata_y = numpy.array(fulldata[:, -1], dtype='int32')

    train_set_x = fulldata_x[0:train_set_size, :]
    train_set_y = fulldata_y[0:train_set_size]
    valid_set_x = fulldata_x[train_set_size:, :]
    valid_set_y = fulldata_y[train_set_size:]

    return train_set_x, train_set_y, valid_set_x, valid_set_y, None


def load_ARCENE():
    train_x = numpy.loadtxt('data/nips03/ARCENE/arcene_train.data')
    train_y = numpy.equal(numpy.loadtxt('data/nips03/ARCENE/arcene_train.labels'), 1)

    valid_x = numpy.loadtxt('data/nips03/ARCENE/arcene_valid.data')
    valid_y = numpy.equal(numpy.loadtxt('data/nips03/arcene_valid.labels'), 1)

    test_x = numpy.loadtxt('data/nips03/ARCENE/arcene_test.data')

    return train_x, train_y, valid_x, valid_y, test_x

def load_DOROTHEA():
    ntrain = 800
    nvalid = 350
    ntest = 800

    dim = 100000

    def do_x(fn, n):
        mtx = numpy.zeros((n, dim), dtype=numpy.float32)
        x = 0
        for x, l in enumerate(open(fn)):
            for y in l.rstrip().split(' '):
                mtx[x, int(y)-1] = 1
        return mtx

    train_x = do_x('data/nips03/DOROTHEA/dorothea_train.data', ntrain)
    train_y = numpy.equal(numpy.loadtxt('data/nips03/DOROTHEA/dorothea_train.labels'), 1)

    valid_x = do_x('data/nips03/DOROTHEA/dorothea_valid.data', nvalid)
    valid_y = numpy.equal(numpy.loadtxt('data/nips03/dorothea_valid.labels'), 1)

    test_x = do_x('data/nips03/DOROTHEA/dorothea_test.data', ntest)

    return train_x, train_y, valid_x, valid_y, test_x

def load_GISETTE():
    train_x = numpy.loadtxt('data/nips03/GISETTE/gisette_train.data')
    train_y = numpy.equal(numpy.loadtxt('data/nips03/GISETTE/gisette_train.labels'), 1)

    valid_x = numpy.loadtxt('data/nips03/GISETTE/gisette_valid.data')
    valid_y = numpy.equal(numpy.loadtxt('data/nips03/gisette_valid.labels'), 1)

    test_x = numpy.loadtxt('data/nips03/GISETTE/gisette_test.data')

    return train_x, train_y, valid_x, valid_y, test_x

