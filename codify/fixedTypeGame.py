from utils import eval_coalition, powerset
from State import State
from collections import Counter
import logging

def game_state_payoff(fixed_types, game, state):
    '''
    :param fixed_types: dict of agents and types
    :param game:
    :param state:
    :return: a dictionary with key = agent, value = payoff
    '''

    if state.t == state.horizon:
        # Base-case: termination! get the
        # exit values
        res = {agent:eval_coalition([agent.agent_type], game.tasks) \
                    for agent in state.active_agents}

    else:
        num_agents = len(fixed_types)
        payoff = dict()

        # https://www.geeksforgeeks.org/python-combine-two-dictionary-adding-values-for-common-keys/
        for proposer in fixed_types:
            proposal_payoff =  proposal_state_payoff(fixed_types, game, state, proposer)
            payoff = Counter(payoff) + Counter(proposal_payoff)

        # divided by the prob. of selecting a proposer
        res = {p: payoff[p]/num_agents for p in payoff.keys()}

    logging.info('game_state_payoff({}, t={}, active={})={}'.format(fixed_types, state.t, \
                                                                   state.active_agents, res))
    return res



def proposal_state_payoff(fixed_types, game, state, proposer):
    '''
    :param fixed_types:
    :param game:
    :param state:
    :param proposer:
    :return: a dictionary with key = agent, value = payoff

    TODO:
    - if some proposals are equally good, the proposer RANDOMLY picks one!
    - instead of division proportional to weight, the agent should run a linear program
    to pick a division such that it is agree-able to other agents while maximizing his own
    share!
    This can actually be solved explicitly!
    '''
    # next state without any coalition formed
    next_state_no_C = State(state.active_agents, state.t+1, state.horizon)
    continuation_payoff =  game_state_payoff(fixed_types, game, next_state_no_C)
    tot_continuation = sum(continuation_payoff.values())

    # look at all the coalitions involving proposer
    other_players = [player for player in state.active_agents if player != proposer]
    best_proposal_for_proposer = [None, 0] # [C, V_C]

    for s in powerset(other_players):
        proposal = [proposer] + list(s)
        W = [p.agent_type for p in proposal]
        V_C = eval_coalition(W, game.tasks)


        #  Division proportional to weight!
        #agree = True # all agents in C want to join the coalitions?
        #proposer_val = V_C * proposer.agent_type
        #for p in proposal:
        #    # revenue is divided proportional to the weights!
        #    if V_C * (p.agent_type)/sum(W) < continuation_payoff[p]:
        #        agree = False
        #        break
        #if not agree:
        #    proposer_val = continuation_payoff[proposer]

        #if proposer_val > best_proposal_for_proposer[1]:
        #    best_proposal_for_proposer = [proposal, V_C]

    proposal, V_C = best_proposal_for_proposer
    active_agents = state.active_agents.difference(proposal)
    next_state = State(active_agents, state.t+1, state.horizon)
    fixed_types2 = fixed_types.copy()

    map(fixed_types2.pop, proposal)
    payoff = game_state_payoff(fixed_types2, game, next_state)
    W_C = sum([p.agent_type for p in proposal])
    payoff_C = {p:V_C * p.agent_type/W_C for p in proposal}
    res = Counter(payoff) + Counter(payoff_C)
    logging.info('proposal_state_payoff({}, t={}, active={}, proposer={})={} | proposal chosen:{}'.format(fixed_types, state.t, \
                                                                                      state.active_agents, proposer,
                                                                                      res, proposal))
    return res


