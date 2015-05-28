import logging
import random
import numpy

import cPickle

from picklable_itertools import iter_

from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme

import dataset

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def batch(items, batch_size):
    if batch_size == None:
        return [items]
    return [items[i: min(i + batch_size, len(items))]
            for i in xrange(0, len(items), batch_size)]


class NPYTransposeDataset(Dataset):

    def __init__(self, ref_data_x, data, **kwargs):
        self.ref_data_x = ref_data_x
        self.data_x, self.data_y = data

        self.ref_data_x = self.ref_data_x.astype(numpy.float32)
        self.data_x = self.data_x.astype(numpy.float32)

        self.nitems, self.nfeats = self.data_x.shape

        self.provides_sources = ('r', 'x', 'y')

        super(NPYTransposeDataset, self).__init__(**kwargs)

    def get_data(self, state=None, request=None):
        if request is None:
            raise ValueError("Expected request: i vector and j vector")
        i_list, j_list = request
        return (self.ref_data_x[:, j_list].T,
                self.data_x[:, j_list][i_list, :],
                self.data_y[i_list])


class TransposeIt(IterationScheme):

    def set_dims(self, nitems, nfeats):
        self.nitems = nitems
        self.nfeats = nfeats


class RandomTransposeIt(TransposeIt):

    def __init__(self, ibatchsize, irandom, jbatchsize, jrandom):
        self.ibatchsize = ibatchsize
        self.jbatchsize = jbatchsize

        self.irandom = irandom
        self.jrandom = jrandom

    def get_request_iterator(self):
        i = range(self.nitems)
        j = range(self.nfeats)

        if self.irandom:
            random.shuffle(i)
        if self.jrandom:
            random.shuffle(j)

        ib = batch(i, self.ibatchsize)
        jb = batch(j, self.jbatchsize)

        return iter_([(ii, jj) for ii in ib for jj in jb])


class LogregOrderTransposeIt(TransposeIt):

    def __init__(self, ibatchsize, irandom, jlogregparamfile, jcount):
        self.ibatchsize = ibatchsize
        self.irandom = irandom

        with open(jlogregparamfile) as f:
            w = cPickle.load(f)

        jsort = (-(w ** 2)).flatten().argsort(axis=0)

        self.js = list(jsort[:jcount])
        random.shuffle(self.js)

    def get_request_iterator(self):
        i = range(self.nitems)

        if self.irandom:
            random.shuffle(i)

        ib = batch(i, self.ibatchsize)

        return iter_([(ii, self.js) for ii in ib])


def prepare_data(name, iteration_scheme, valid_iteration_scheme, randomize_feats=False):
    if name == 'ARCENE':
        ds = dataset.load_ARCENE()
    elif name == 'DOROTHEA':
        ds = dataset.load_DOROTHEA()
    elif name == 'AMLALL':
        ds = dataset.load_AMLALL()
    else:
        raise ValueError("No such dataset " + name)

    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x = ds

    # Normalize all features according to coefficients given by train_x
    train_x_norms = numpy.sqrt((train_set_x ** 2).sum(axis=0, keepdims=True))
    div = train_x_norms + numpy.equal(train_x_norms, 0)
    train_set_x = train_set_x / div
    valid_set_x = valid_set_x / div
    test_set_x = test_set_x / div

    feats = train_set_x
    if randomize_feats:
        transform = numpy.random.randn(train_set_x.shape[0], train_set_x.shape[0])
        feats = numpy.dot(transform, feats)

    data_train = NPYTransposeDataset(feats, (train_set_x, train_set_y))
    data_valid = NPYTransposeDataset(feats, (valid_set_x, valid_set_y))

    iteration_scheme.set_dims(data_train.nitems, data_train.nfeats)
    valid_iteration_scheme.set_dims(data_valid.nitems, data_valid.nfeats)

    stream_train = DataStream(data_train, iteration_scheme=iteration_scheme)
    stream_valid = DataStream(
        data_valid, iteration_scheme=valid_iteration_scheme)

    return stream_train, stream_valid


if __name__ == "__main__":
    # Test
    stream = prepare_data(
        "ARCENE", "train", RandomTransposeIt(10, True, 100, True))
    print(next(stream.get_epoch_iterator()))
