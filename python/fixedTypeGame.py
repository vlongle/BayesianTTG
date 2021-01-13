from utils import eval_coalition, powerset
from State import State
from collections import Counter, defaultdict
import logging
import numpy as np
from utils import *



def greedy_division(proposer, proposal, V_C, tot_continuation, continuation_payoff, eps = 0.0):
    '''
    :param proposer:
    :param proposal:
    :param V_C:
    :param continuation_payoff:
    :return: optimize division for the proposer!
    '''
    diff = V_C - tot_continuation
    payoff = defaultdict(float)
    for other_player in proposal:
        if other_player == proposer:
            continue
        # pay the other players just enough to sway them to our team
        payoff[other_player] = continuation_payoff[other_player] +  diff * eps

    # Pay the rest to the proposer!
    payoff[proposer] = V_C - sum(payoff.values())
    return payoff


def proposal_state_payoff(fixed_types, game, state, proposer, ret_proposal=False):
    '''
    :param fixed_types:
    :param game:
    :param state:
    :param proposer:
    :return: a dictionary with key = agent, value = payoff

    - if some proposals are equally good, the proposer RANDOMLY picks one!
    - If agent believe that a coalition C is more available than the continuation,
    she will propose the greedy division for that coalition
    - Otherwise, she would propose a proportional weighting division
    '''

    # next state without any coalition formed
    next_state_no_C = State(state.active_agents, state.t+1, state.horizon)
    continuation_payoff =  game_state_payoff(fixed_types, game, next_state_no_C)

    other_players = [player for player in state.active_agents if player != proposer]


    best_proposals = [(None, float('-inf'), None, None)] # [C, proposer_payoff, payoff, V_C]

    # look at all the coalitions involving proposer
    for s in powerset(other_players):
        proposal = [proposer] + list(s)
        logging.debug('entertaining proposal={}'.format(proposal))
        W = [fixed_types[p] for p in proposal]
        V_C = eval_coalition(W, game.tasks)
        tot_continuation = sum([continuation_payoff[p] for p in s])

        # if feasible, then greedy
        if V_C >= tot_continuation:
            payoff = greedy_division(proposer, proposal, V_C, \
                                     tot_continuation, continuation_payoff)
        # else, proportional weight division
        else:
            # there should always exist at least one feasible proposal:
            # i.e. the singleton coalition!
            # if not feasible, agent should just propose the singleton coalition!!
            logging.debug('proposal={} NOT feasible'.format(proposal))
            #payoff = {agent:fixed_types[agent]/sum(W) for agent in proposal}
            #payoff[proposer] = 0
            continue

        proposer_payoff = payoff[proposer]

        logging.debug('other_players={}, V_C={}, cont={}, payoff={}'.format(s, V_C, tot_continuation, \
                                                                            proposer_payoff))
        # update best proposals
        if proposer_payoff == best_proposals[0][1]:
                best_proposals.append((proposal, proposer_payoff, payoff, V_C))
        elif proposer_payoff > best_proposals[0][1]:
            best_proposals = [(proposal, proposer_payoff, payoff, V_C)]



    logging.debug('best_proposals={}'.format(best_proposals))
    # randomly choose the optimal proposal
    best_proposal, _, best_payoff_C, best_V_C = best_proposals[np.random.choice(len(best_proposals))]

    active_agents = state.active_agents.difference(best_proposal)
    # next_state: increment time and update active_agents!
    next_state = State(active_agents, state.t+1, state.horizon)
    payoff = game_state_payoff(fixed_types, game, next_state)


    res = Counter(payoff) + Counter(best_payoff_C)
    logging.debug('proposal_state_payoff({}, t={}, active={}, proposer={})={} | proposal chosen:{}'.format(fixed_types, state.t, \
                                                                                      state.active_agents, proposer,
                                                                                      res, best_proposal))
    if ret_proposal:
        return best_proposal, best_V_C, best_payoff_C
    return res


