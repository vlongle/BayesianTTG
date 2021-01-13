from fixedTypeGame import *
from itertools import product
from operator import mul
from functools import reduce
from utils import *
from BayesianUpdateBelief import *
'''
TODO: 
Stabilize this proposal!
- Either by run more Monte Carlo
or
- Take the expected value directly!!




TODO:
- pick equally good actions RANDOMLY for exploration!!!
'''





## Monte Carlo way!
#def agent_propose(proposer, game, state):
#    fixed_types = proposer.draw_types()
#    proposal, V_C, payoff_C = proposal_state_payoff(fixed_types, game, state, \
#                          proposer, ret_proposal=True)
#
#    division_frac = get_divison_vec(V_C, payoff_C)
#    # here we should use inverse propensity weighting!!
#    return proposal, division_frac

def scale(counter, scalar):
    for x in counter.keys():
        counter[x] *= scalar
    return counter


# TODO: randomly select optimal action for exploration!!!
def agent_propose(proposer, game, state):
    other_active_agents = [player for player in state.active_agents if player != proposer]
    best_R_Cp, best_proposal, best_div = float('-inf'), [], []

    proposals = [] # (proposal, R_cp)
    for Cp in powerset(other_active_agents):
        R_Cp, div = expected_coalition_given_type(proposer, proposer, proposer.agent_type-1,
                                      Cp, game, state, True)

        Cp = set(Cp)
        Cp.add(proposer)
        logging.debug("agent {} proposes {} w/ R_cp {}".format(proposer, Cp, R_Cp))
        if R_Cp > best_R_Cp:
            best_R_Cp = R_Cp
            best_div = [div]
            best_proposal = [Cp]
        elif R_Cp == best_R_Cp:
            best_div.append(div)
            best_proposal.append(Cp)


        proposals.append((Cp, R_Cp))
    logging.info("Proposer mental table\n")
    logging.info(proposals)

    logging.info("best_proposals = {}".format(best_proposal))
    # choose randomly among optimal policies
    if len(best_proposal) == 0 or len(best_proposal) != len(best_div):
        raise Exception('Something wrong in agent_propose!!!!')
    i = np.random.randint(low=0, high=len(best_proposal))
    logging.info("i={}".format(i))
    return  best_proposal[i], best_div[i]
### Monte Carlo way!
#def agent_respond(agent, proposal, agent_share, game, state):
#    '''
#    :param agent:
#    :param proposal:
#    :param agent_share:
#    :param game:
#    :param state:
#    :return:
#    This agent draws a type vector and calculate his expected_reward
#    and continuation_payoff based on that. Responds yes if expected_reward >=
#    continuation_payoff!
#    '''
#    fixed_types = agent.draw_types()
#    # increment time
#    next_state = State(state.active_agents, state.t + 1, state.horizon)
#
#    continuation_payoff = game_state_payoff(fixed_types, game, next_state)[agent]
#W = [fixed_types[agent] for agent in proposal]
#    V_C = eval_coalition(W, game.tasks)
#    V_C = eval_coalition(fixed_types.values(), game.tasks)
#    expected_reward = agent_share * V_C
#    if expected_reward >= continuation_payoff:
#        return 'yes'
#    return 'no'


## exhaustive calculation way
def agent_respond(agent, proposal, agent_share, game, state):
    '''
    :param agent:
    :param proposal:
    :param agent_share:
    :param game:
    :param state:
    :return:
    This agent draws a type vector and calculate his expected_reward
    and continuation_payoff based on that. Responds yes if expected_reward >=
    continuation_payoff!
    '''
    other_active_agents = [player for player in state.active_agents if player != agent]
    acceptance = rejection = 0
    for agent_types in product(range(1, game.T + 1), repeat=len(other_active_agents)):

        prob = reduce(mul, \
                      [agent.belief[agent][other_agent_type - 1] for \
                       agent, other_agent_type in zip(other_active_agents, agent_types)])

        fixed_types = {agent: agent_type for agent, agent_type in \
                       zip(other_active_agents, agent_types)}
        fixed_types[agent] = agent.agent_type
        # increment time
        next_state = State(state.active_agents, state.t + 1, state.horizon)

        continuation_payoff = game_state_payoff(fixed_types, game, next_state)[agent]
        W = [fixed_types[agent] for agent in proposal]
        V_C = eval_coalition(W, game.tasks)
        expected_reward = agent_share * V_C

        acceptance += prob * expected_reward
        rejection += prob * continuation_payoff

    # proportional choice
    #Z = acceptance + rejection
    #return np.random.choice(['yes', 'no'], p=[acceptance/Z, rejection/Z])

    logging.info('Responder {} mental table:\n'.format(agent))
    logging.info('Acceptance = {};  Rejection = {}\n'.format(acceptance, rejection))

    # arg-max choice!
    if acceptance > rejection:
        return 'yes'
    elif acceptance < rejection:
        return 'no'
    return np.random.choice(['yes', 'no'])

