import numpy as np

def argmax(logits):
    return np.argmax(logits)

def argsort(seq, reverse=False):
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)