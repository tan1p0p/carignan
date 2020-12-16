import logging

import numpy as np

def show_logs():
    logging.basicConfig(
        format="%(relativeCreated)12d [%(filename)s:%(funcName)10s():%(lineno)s] [%(process)d] %(message)s",
        level=logging.DEBUG
    )

def zscore(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore