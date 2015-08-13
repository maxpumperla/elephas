from __future__ import absolute_import

def to_simple_rdd(sc, features, labels):
    pairs = [(x,y) for x,y in zip(features, labels)]
    return sc.parallelize(pairs)
