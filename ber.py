from theano import tensor
import theano
import numpy

def ber(y, pred):
    a = (tensor.neq(y, 1) * tensor.neq(pred, 1)).sum()
    b = (tensor.neq(y, 1) * tensor.eq(pred, 1)).sum()
    c = (tensor.eq(y, 1) * tensor.neq(pred, 1)).sum()
    d = (tensor.eq(y, 1) * tensor.eq(pred, 1)).sum()
    [a, b, c, d] = [tensor.cast(x, dtype=theano.config.floatX) for x in [a, b, c, d]]
    return (b / (a + b) + c / (c + d)) / numpy.float32(2)

