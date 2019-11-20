import logging, logging.config
import numpy as np
import pandas as pd


logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        },
    }
})


def make_logger(name):
    return logging.getLogger(name)


logger = make_logger('utils.py')


def check_npy_allclose(clsname, a, b, rtol=1e-05, atol=1e-08, equal_nan=False, logger=logger):
    allclose = np.allclose(a, b, rtol, atol, equal_nan)
    if not allclose:
        err_msg = """{} not all close:
                     [left]: {}
                     [right]: {}"""
        err_msg = err_msg.format(clsname, a, b)
        logger.error(err_msg)
    return allclose


def arr_str_to_npy_arr(s):
    arr = [float(item) for item in s[1:-1].split()]
    return np.asarray(arr)
