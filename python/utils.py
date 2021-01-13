import numpy as np
from bisect import bisect_right
from itertools import chain, combinations, product
import logging
from simplexUtils import *
from functools import reduce
from operator import mul
from State import *
from collections import Counter, defaultdict

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))



def one_hot_vector(agent_type, T):
    v = np.zeros(T)
    v[agent_type-1] = 1
    return v

def find_higher_threshold_task(tasks, task):
    # assuming that tasks are already sorted,
    # return the next task that is higher in threshold that this current task.
    # If this task is the largest, return None
    i = tasks.index(task)
    if i == len(tasks)-1:
        return None
    return tasks[i+1]

def eval_coalition(C, tasks, ret_reward=True):
    '''
    C is a list of agent weight!
    '''
    for w in C:
        if w <= 0:
            logging.critical('w={} is non-positive'.format(w))
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
    return res, res.reward


def scale(counter, scalar):
    for x in counter.keys():
        counter[x] *= scalar
    return counter


def get_divison_vec(V_C, payoff_C):
    '''
    :param V_C:
    :param payoff_C:
    :return: normalize the payoff to fraction [0, 1]
    '''
    return {p:payoff_C[p]/V_C for p in payoff_C.keys()}

def nature_pick_proposer(state):
    logging.debug('state.active_agents: {}'.format(state.active_agents))
    return np.random.choice(list(state.active_agents))

def generate_proposals(state, proposer,bandit=False):
    # generate all proposal and division pairs from the currently active players
    other_active_agents = [player for player in state.active_agents if player != proposer]
    proposals = []
    for C in powerset(other_active_agents):
       coalition = list(C) + [proposer]
       coalition = sorted(coalition, key=lambda player:player.name)
       n_agents = len(coalition)
       # demand in increment of 10%
       divs = simplex_grid(n_agents)
       if bandit:
           coalition = tuple(coalition)
           proposals += [(coalition, tuple(0.1 * np.array(div))) for div in divs]
           continue
       proposals += [(coalition, 0.1 * np.array(div)) for div in divs]
    return proposals

def expected_coalition_value(coalition, predictor, game):
    # expected value of coalition given predictor's belief
    expected_value = 0
    for agent_types in product(range(1, game.T + 1), repeat=len(coalition)):
        prob = reduce(mul, \
                          [predictor.belief[agent][other_agent_type - 1] for \
                           agent, other_agent_type in zip(coalition, agent_types)])
        expected_value += prob * eval_coalition(list(agent_types), game.tasks)
    return expected_value

def expected_continuation_payoff(observer, game):
    state = game.state
    next_state_no_C = State(state.active_agents, state.t + 1, state.horizon)
    continuation_payoff = Counter()
    for agent_types in product(range(1, game.T + 1), repeat=len(state.active_agents)):
        prob = reduce(mul, \
                      [observer.belief[agent][other_agent_type - 1] for \
                       agent, other_agent_type in zip(state.active_agents, agent_types)])
        fixed_types = {agent: agent_type for agent, agent_type in \
                       zip(state.active_agents, agent_types)}
        continuation_payoff_fixed_belief = game_state_payoff(fixed_types, game, next_state_no_C)
        continuation_payoff += scale(continuation_payoff_fixed_belief, prob)
    return continuation_payoff

def predict_responses(predictor, proposal, game, ret_extra=False):
    coalition, div = proposal
    predicted_reward = expected_coalition_value(coalition, predictor, game)
    continuation_payoff = expected_continuation_payoff(predictor, game)
    responses = {}
    for player in coalition:
        player_share = div[coalition.index(player)]
        if player_share * predicted_reward >= continuation_payoff[player]:
            #print('predicted_reward:', predicted_reward, 'share:', player_share, 'cont:', continuation_payoff[player],
            #      'responding yes!')
            responses[player] = 'yes'
        else:
            responses[player] = 'no'

    if ret_extra:
        return responses, predicted_reward, continuation_payoff
    return responses


def game_state_payoff(fixed_types, game, state):
    '''
    :param fixed_types: dict of agents and types
    :param game:
    :param state:
    :return: a dictionary with key = agent, value = payoff
    '''

    if state.t >= state.horizon:
        # Base-case: termination! get the
        # exit values
        res = {agent:eval_coalition([fixed_types[agent]], game.tasks) \
                    for agent in state.active_agents}
        #logging.info('reservation value: {}'.format(res))

    # WON'T BE REACHED HERE IN THE ONE-STEP PROPOSER MODEL
    #else:
    #    num_agents = len(fixed_types)
    #    payoff = dict()

    #    # https://www.geeksforgeeks.org/python-combine-two-dictionary-adding-values-for-common-keys/
    #    for proposer in state.active_agents:
    #        proposal_payoff =  proposal_state_payoff(fixed_types, game, state, proposer)
    #        payoff = Counter(payoff) + Counter(proposal_payoff)

    #    # divided by the prob. of selecting a proposer
    #    res = {p: payoff[p]/num_agents for p in payoff.keys()}

    #logging.debug('game_state_payoff({}, t={}, active={})={}'.format(fixed_types, state.t, \
    #                                                               state.active_agents, res))
    return res
