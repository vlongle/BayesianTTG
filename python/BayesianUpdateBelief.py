'''
TODO: rewrite belief update !!!
'''
from itertools import product
from operator import mul
from functools import reduce
from utils import *
from fixedTypeGame import game_state_payoff, greedy_division
from State import State
import logging
from collections import Counter

def expected_coalition_given_type(observer, proposer, proposer_type, other_players, game, state,
                                  ret_payoff=False,delta=0.001):
    '''
    :param observer:
    :param proposer_type: zero-indexed!
    :param C:
    :return:
    Calculate E[R(C | t_j)] = E(V_c) - E(Q)
    '''
    res = 0

    other_active_agents = [player for player in state.active_agents if player != proposer\
                           and player != observer]

    logging.debug("other_active_agents={}".format(other_active_agents))
    div = Counter()
    for agent_types in product(range(1, game.T + 1), repeat=len(other_active_agents)):
        if agent_types:
            prob = reduce(mul, \
                              [observer.belief[agent][other_agent_type - 1] for \
                               agent, other_agent_type in zip(other_active_agents, agent_types)])
        else:
            prob = 1


        # lure price
        fixed_types = {agent: agent_type for agent, agent_type in \
                       zip(other_active_agents, agent_types)}
        fixed_types[proposer] = proposer_type + 1
        fixed_types[observer] = observer.agent_type

        logging.debug('expected_coalition fixed_types={}'.format(fixed_types))
        next_state_no_C = State(state.active_agents, state.t + 1, state.horizon)
        continuation_payoff = game_state_payoff(fixed_types, game, next_state_no_C)

        W = [agent_w for agent, agent_w in fixed_types.items() if agent in other_players] + [proposer_type+1]
        V_C = eval_coalition(W, game.tasks)

        lure_price = sum([continuation_payoff[p] for p in other_players])
        logging.info('V_C={}, lure_price={}, prob={}'.format(V_C, lure_price, prob))
        #res += prob * (max(V_C - lure_price, delta * min([t.reward for t in game.tasks])))

        # payoff division when feasible
        if V_C > lure_price:
            payoff = {agent:p for agent, p in continuation_payoff.items() if agent in other_players}
            payoff[proposer] = V_C - lure_price
        else:
            # if not feasible: proposer get pay 0, and other players get paid proportional to continuation payoff

            #payoff = {agent:proposer.MLE(agent) for agent in other_players}
            #Z = sum(payoff.values())
            #payoff = {agent:(p/Z) *  V_C for agent, p in payoff.items()}


            payoff = {agent:continuation_payoff[agent] for agent in other_players}
            Z = sum(payoff.values())
            payoff = {agent:(p/Z) *  V_C for agent, p in payoff.items()}
            payoff[proposer] = 0


        res += prob * payoff[proposer]
        logging.info('proposal={}, type={}, prob={}, div={}'.format(list(other_players) + [proposer],
                                                                     fixed_types, prob, payoff))
        Z = sum(payoff.values())
        payoff = get_divison_vec(Z, payoff)
        div += scale(payoff, prob)


    logging.info('expected_coalition(observer={}, proposer={}, proposer_type={}, other_players={})={} & {}'.format(
        observer, proposer, proposer_type, other_players, res, div
    ))

    if ret_payoff:
        return res, div
    return res




def update_belief_propose(observer, proposer, proposal, game, state):
    '''
    :param observer:
    :param proposer:
    :param proposal:
    :param game:
    :param state:
    :return:
    Observer updates her belief about the proposer's type based on
    the proposal = (C, D) where C = coalition and D = division rule
    >> Assumption: one-wrong at the time. Observer believes that his belief
    about other agents except the proposer is correct.

    Punish the inconsistency with current belief!
    '''
    if observer == proposer:
        return
    other_players = proposal.copy()
    other_players.remove(proposer)

    other_active_agents = [player for player in state.active_agents if player != proposer]

    logging.info("\nupdate_belief_propose(observer={}, proposer={}, proposal={})".format(observer,
                                                                                        proposer,
                                                                                        proposal))
    delta = 0.9
    eps = 0.1 # "trembling-hand" in the update
    #logging.debug("num_agents={}".format(num_agents))
    for proposer_type in range(game.T):
        # calculate Pr(C | t_j) = R(C|t_j)/ sum R(C' | t_j)
        normalizing = R_c = 0
        Rs = []
        for Cp in powerset(other_active_agents): # for C'
            R_Cp = expected_coalition_given_type(observer, proposer, proposer_type, Cp, game, state)
            normalizing += R_Cp
            if set(Cp) == set(other_players):
                R_c = R_Cp
            else:
                Rs.append((Cp, R_Cp))

        inconsistency = [R_c < R-eps for Cp, R in Rs]
        if any(inconsistency):
            logging.info("inconsistency!! proposer={} | observer={} | proposer_type={} | Rs={} | R_c = {}".format(proposer,
                                                                                        observer,
                                                                                        proposer_type,
                                                                                            Rs,
                                                                                            R_c))
            PrC_given_type = delta
        else:
            PrC_given_type = 1


        logging.info("given type={} | their choice = {} | other alternatives= {} | Pr={}".format(proposer_type+1, R_c, Rs,
                                                                  PrC_given_type))
        observer.belief[proposer][proposer_type] *= PrC_given_type




    observer.normalize_to_prob(proposer)

#def update_belief_propose(observer, proposer, proposal, game, state):
#    '''
#    :param observer:
#    :param proposer:
#    :param proposal:
#    :param game:
#    :param state:
#    :return:
#    Observer updates her belief about the proposer's type based on
#    the proposal = (C, D) where C = coalition and D = division rule
#    >> Assumption: one-wrong at the time. Observer believes that his belief
#    about other agents except the proposer is correct.
#
#    We'll ignore the division rule D comes up by the proposer, and only consider
#    the proposal C. The observer runs his own greedy calculation and find the expected
#    reward for the proposer given the proposer's type. The likelihood is proportional
#    to the expected reward!
#    '''
#    if observer == proposer:
#        return
#    other_players = proposal.copy()
#    other_players.remove(proposer)
#
#    other_active_agents = [player for player in state.active_agents if player != proposer]
#
#    logging.debug("update_belief_propose(observer={}, proposer={}, proposal={})".format(observer,
#                                                                                        proposer,
#                                                                                        proposal))
#    #logging.debug("num_agents={}".format(num_agents))
#    for proposer_type in range(game.T):
#        # calculate Pr(C | t_j) = R(C|t_j)/ sum R(C' | t_j)
#        normalizing = R_c = 0
#        Rs = []
#        for Cp in powerset(other_active_agents): # for C'
#            R_Cp = expected_coalition_given_type(observer, proposer, proposer_type, Cp, game, state)
#            normalizing += R_Cp
#            if set(Cp) == set(other_players):
#                R_c = R_Cp
#            else:
#                Rs.append(R_Cp)
#
#
#        deviations = [max(R_c - R, 0.01) for R in Rs]
#        PrC_given_type = sum(deviations)/len(deviations)
#
#        logging.info("PrC_given_type={} | type={}".format(PrC_given_type, proposer_type))
#        observer.belief[proposer][proposer_type] *= PrC_given_type




#    observer.normalize_to_prob(proposer)

#def update_belief_propose(observer, proposer, proposal, game, state):
#    '''
#    :param observer:
#    :param proposer:
#    :param proposal:
#    :param game:
#    :param state:
#    :return:
#    Observer updates her belief about the proposer's type based on
#    the proposal = (C, D) where C = coalition and D = division rule
#    >> Assumption: one-wrong at the time. Observer believes that his belief
#    about other agents except the proposer is correct.
#
#    We'll ignore the division rule D comes up by the proposer, and only consider
#    the proposal C. The observer runs his own greedy calculation and find the expected
#    reward for the proposer given the proposer's type. The likelihood is proportional
#    to the expected reward!
#    '''
#    if observer == proposer:
#        return
#    other_players = proposal.copy()
#    other_players.remove(proposer)
#
#    other_active_agents = [player for player in state.active_agents if player != proposer]
#
#    logging.debug("update_belief_propose(observer={}, proposer={}, proposal={})".format(observer,
#                                                                                        proposer,
#                                                                                        proposal))
#    #logging.debug("num_agents={}".format(num_agents))
#    for proposer_type in range(game.T):
#        # calculate Pr(C | t_j) = R(C|t_j)/ sum R(C' | t_j)
#        normalizing = R_c = 0
#        for Cp in powerset(other_active_agents): # for C'
#            R_Cp = expected_coalition_given_type(observer, proposer, proposer_type, Cp, game, state)
#            normalizing += R_Cp
#            if set(Cp) == set(other_players):
#                R_c = R_Cp
#

#        PrC_given_type = R_c/normalizing
#        logging.info("PrC_given_type={} | type={}".format(PrC_given_type, proposer_type))
#        observer.belief[proposer][proposer_type] *= PrC_given_type
#
#    observer.normalize_to_prob(proposer)

def update_belief_respond(observer, responder, response, proposal, share, game,
                          state):
    '''
    :return:
    observer updates her belief about the responder
    The likelihood is calculated as follows.
    P(Yes | type) ~  expected_share
    P(No | type) ~ continuation_payoff
    >> Assumption: one-wrong at the time. Observer believes that his belief
    about other agents except the responder is correct.
    '''
    if observer == responder:
        return

    other_players = proposal.copy()
    other_players.remove(responder)

    other_active_agents = [player for player in state.active_agents if player != responder\
                           and player != observer]

    num_agents = len(other_active_agents)
    logging.info("\nupdate_belief_responder(observer={}, responder={}, response={}, proposal={})".format(observer,
                                                                                                       responder,
                                                                                                       response,
                                                                                                       proposal))
    delta = 0.9
    for responder_type in range(game.T):
        # calculate belief(responder agent_type | action)
        expected_reward = continuation = 0
        # this probabilistic calculation can probably also be substituted
        # by sampling as we do with the policy maximization!
        for agent_types in product(range(1, game.T+1), repeat=num_agents):
            # probability of this type vector according to the observer's
            # belief
            if agent_types:
                prob = reduce(mul,\
                              [observer.belief[agent][other_agent_type-1] for \
                                agent, other_agent_type in zip(other_active_agents, agent_types)])
            else:
                logging.info('not agent_types! why?? {} {}'.format(agent_types, bool(agent_types)))
                prob = 1

            fixed_types = {agent: agent_type for agent, agent_type in \
                           zip(other_active_agents, agent_types)}
            fixed_types[responder] = responder_type + 1
            fixed_types[observer] = observer.agent_type

            W = [agent_w for agent, agent_w in fixed_types.items() if agent in other_players] + [responder_type+1]\

            expected_reward += share * prob * eval_coalition(W, \
                                                     game.tasks)

            next_state_no_C = State(state.active_agents, state.t + 1, state.horizon)
            continuation += prob * game_state_payoff(fixed_types, game, next_state_no_C)[responder]
            logging.info('agent_types={} | responder_type={} | fixed_type={} | expected_reward={} | continuation = {} | prob = {} | W={}'.format(
                agent_types, responder_type, fixed_types, expected_reward, continuation, prob, W
            ))

        eps = 0.1 # trembling hand
        # check for inconsistency!
        likelihood = 1
        if response == 'yes' and continuation > expected_reward + eps:
            likelihood = delta
        elif response == 'no' and expected_reward > continuation + eps:
            likelihood = delta
        # posterior update!
        observer.belief[responder][responder_type] *= likelihood
        observer.normalize_to_prob(responder) # normalize [0, 1]
        logging.info("likelihood for weight {} is {} | acceptance reward = {} | rejection = {}".format(
            responder_type, likelihood, expected_reward, continuation))



#def update_belief_respond(observer, responder, response, proposal, game,
#                          state):
#    '''
#    :return:
#    observer updates her belief about the responder
#    The likelihood is calculated as follows.
#    P(Yes | type) ~  expected_share
#    P(No | type) ~ continuation_payoff
#    >> Assumption: one-wrong at the time. Observer believes that his belief
#    about other agents except the responder is correct.
#    '''
#    if observer == responder:
#        return
#
#    other_players = proposal.copy()
#    other_players.remove(responder)
#
#    other_active_agents = [player for player in state.active_agents if player != responder\
#                           and player != observer]
#
#    num_agents = len(other_active_agents)
#    logging.debug("update_belief_responder(observer={}, responder={}, response={}, proposal={})".format(observer,
#                                                                                                       responder,
#                                                                                                       response,
#                                                                                                       proposal))
#    for responder_type in range(game.T):
#        # calculate belief(responder agent_type | action)
#        expected_reward = continuation = 0
#        # this probabilistic calculation can probably also be substituted
#        # by sampling as we do with the policy maximization!
#        for agent_types in product(range(1, game.T+1), repeat=num_agents):
#            # probability of this type vector according to the observer's
#            # belief
#            if agent_types:
#                prob = reduce(mul,\
#                              [observer.belief[agent][other_agent_type-1] for \
#                                agent, other_agent_type in zip(other_players, agent_types)])
#            else:
#                prob = 1
#
#            fixed_types = {agent: agent_type for agent, agent_type in \
#                           zip(other_active_agents, agent_types)}
#            fixed_types[responder] = responder_type + 1
#            fixed_types[observer] = observer.agent_type
#
#            W = [agent_w for agent, agent_w in fixed_types.items() if agent in other_players] + [responder_type+1]
#
#            expected_reward += prob * eval_coalition(W, \
#                                                     game.tasks)
#            continuation += prob * game_state_payoff(fixed_types, game, state)[responder]
#
#        if response == 'yes':
#            likelihood = expected_reward/(expected_reward + continuation)
#        else:
#            likelihood = continuation/(expected_reward + continuation)
#        # posterior update!
#        observer.belief[responder][responder_type] *= likelihood
#        observer.normalize_to_prob(responder) # normalize [0, 1]
#        logging.debug("likelihood for weight {} is {}".format(responder_type, likelihood))