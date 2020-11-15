import logging


def add(a, b):
    logging.debug('{}+{}={}'.format(a, b, a+b))
    return a + b
