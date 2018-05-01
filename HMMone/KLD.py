import numpy as np
from scipy import *


def asymmetricKL(P, Q):
    print(P * log(P / Q))
    return sum(P * log(P / Q))  # calculate the kl divergence between P and Q


def symmetricalKL(P, Q):
    return (asymmetricKL(P, Q) + asymmetricKL(Q, P)) / 2.00

from functools import reduce
import operator
import math

def kl(p, q):
    return reduce(operator.add, map(lambda x, y: x*math.log(x/y), p, q))