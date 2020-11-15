import numpy as np
from bisect import bisect_right
from itertools import chain, combinations
import logging

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))



def one_hot_vector(agent_type, T):
    v = np.zeros(T)
    v[agent_type-1] = 1
    return v


def eval_coalition(C, tasks, ret_reward=True):
    '''
    C is a list of agent weight!
    '''
    W = sum(C)
    thresholds = sorted([t.threshold for t in tasks])
    insertion_pt = bisect_right(thresholds, W)
    if insertion_pt == 0:
        res = None
    else:
        res = tasks[insertion_pt-1]

    logging.debug('eval_coalition({}, {})={}'.format(C, tasks, res))
    if ret_reward:
        if not res: # None
            return 0
        return res.reward
    return res

