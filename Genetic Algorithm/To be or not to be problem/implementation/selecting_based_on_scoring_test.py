import numpy as np

import matplotlib.pyplot as plt

import pdb


##
scores = np.array([0.4, 0.2, 0.1, 0.3])
scores_normalized = scores/np.sum(scores)
scores_accum = accumulator(scores_normalized)
##
items = []
for i in range(1000):
    rand = np.random.random()
    items.append(score_based_selector(rand, scores_accum))
##



##
def score_based_selector(rand, scores_accum):
    return np.argmax(scores_accum>rand)


def accumulator(x):
    x_accum = x.copy()
    for i in range(len(x)):
        x_accum[i] = np.sum(x[:i+1])

    return x_accum
##
