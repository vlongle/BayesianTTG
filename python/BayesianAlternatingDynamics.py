'''
Given a Game, we will run our Monte Carlo algorithm
1. Nature chooses a proposer.
2. Proposer runs her policy (which is based on her belief).
3. Responders responds.
4. All observers update their beliefs

TODO:
- Update for outcomes
- Multigames with waiting room and that thing ...
'''

from BayesianOptimize import *
from BayesianUpdateBelief import *
import logging
from utils import *


def belief_table(game):
    state = game.state
    ret = ''
    for agent in state.active_agents:
        ret += agent.name + '\t' + str(agent.belief) + '\n'
    return ret
def update_game_state(game, proposal, success):
    '''
    :param game:
    :param proposal:
    :param success:
        True or False, whether the all participants agree
        to join the proposal
    :return:
        Remove players in the case of success
    '''
    game.state.t += 1
    if success:
        game.state.active_agents = game.state.active_agents.difference(proposal)

def alternating_dynamics(game):
    logging.debug('\n\n >> INITIAL BELIEF TABLE')
    logging.debug('\n' + belief_table(game))
    # one simulation
    for t in range(game.horizon):
        if len(game.state.active_agents) == 0:
            logging.debug('Have no active agents left to play!')
            break
        proposer = nature_pick_proposer(game.state)
        logging.info('\n ===== ROUND {} =====\n'.format(t))
        logging.info('>> NATURE picks {} \n'.format(proposer))

        # run policy
        proposal, division_frac = agent_propose(proposer, game, game.state)

        logging.info('\n\n >> proposal = {}, div = {}\n'.format(proposal, division_frac))
        success = True
        responses = {}
        for responder in proposal:
            if responder == proposer:
                continue
            responses[responder] = agent_respond(responder, proposal, \
                          division_frac[responder], game, game.state)
            if responses[responder] == 'no':
                success = False


        logging.info('\n\n responses = {}\n'.format(responses))
        logging.info("Observers={} update now!".format(game.state.active_agents))
        # all agents update their beliefs!
        for observer in game.state.active_agents:
            update_belief_propose(observer, proposer, proposal, game, game.state)
            for responder, response in responses.items():
                update_belief_respond(observer, responder, response, proposal,
                                      division_frac[responder],
                                      game, game.state)


        logging.info('\n\n >> BELIEF TABLE')
        logging.info('\n' + belief_table(game))

        update_game_state(game, proposal, success)

