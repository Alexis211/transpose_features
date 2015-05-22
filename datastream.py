import logging
import random

import cPickle

from picklable_itertools import iter_

from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme

import dataset

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def batch(items, batch_size):
    return [items[i: min(i + batch_size, len(items))]
            for i in xrange(0, len(items), batch_size)]


class NPYTransposeDataset(Dataset):

    def __init__(self, ref_data, data, **kwargs):
        self.ref_data_x, self.ref_data_y = ref_data
        self.data_x, self.data_y = data

        self.nitems, self.nfeats = self.data_x.shape

        self.provides_sources = ('r', 'x', 'y')

        super(NPYTransposeDataset, self).__init__(**kwargs)

    def get_data(self, state=None, request=None):
        if request is None:
            raise ValueError("Expected request: i vector and j vector")
        i_list, j_list = request
        return (self.ref_data_x[:, j_list].T,
                self.data_x[i_list, :][:, j_list],
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

        jsort = (-(w**2)).flatten().argsort(axis=0)
        js = jsort[:jcount]
        self.js = list(js)
        random.shuffle(self.js)

    def get_request_iterator(self):
        i = range(self.nitems)

        if self.irandom:
            random.shuffle(i)

        ib = batch(i, self.ibatchsize)

        return iter_([(ii, self.js) for ii in ib])


def prepare_data(name, part, iteration_scheme):
    if name == 'ARCENE':
        train_set_x, train_set_y, valid_set_x, valid_set_y = dataset.load_ARCENE()
    elif name == 'AMLALL':
        train_set_x, train_set_y, valid_set_x, valid_set_y = dataset.load_AMLALL()
    else:
        raise ValueError("No such dataset " + name)

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    if part == "train":
        data = NPYTransposeDataset(train_set, train_set)
    elif part == "valid":
        data = NPYTransposeDataset(train_set, valid_set)
    else:
        raise ValueError("No such part " + part)

    iteration_scheme.set_dims(data.nitems, data.nfeats)

    stream = DataStream(data, iteration_scheme=iteration_scheme)

    return stream


if __name__ == "__main__":
    # Test
    stream = prepare_data(
        "ARCENE", "train", RandomTransposeIt(10, True, 100, True))
    print(next(stream.get_epoch_iterator()))
