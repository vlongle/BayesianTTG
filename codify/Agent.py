from utils import one_hot_vector
import numpy as np
import logging

class Agent:
    def __init__(self, name, agent_type):
        self.agent_type = agent_type
        self.policy = []
        self.name = str(name) + '.' + str(self.agent_type)
        self.belief = {}  # key = name, value = prob. vector over finite type

    def init_belief(self, game):
        '''
        Initially, uniform prior over other agents
        '''
        self.belief[self.name] = one_hot_vector(self.agent_type, game.T)
        for player in game.agents:
            if player.name == self.name:
                continue
            self.belief[player.name] = np.array([1 / game.T for _ in range(game.T)])

        logging.debug('agent {} belief: {}'.format(self, self.belief))

    def __repr__(self):
        return self.name

    def draw_types(self):
        '''
        draw the types of other agents based on current belief
        '''
        self.belief_types = {}  # key = name, value = drawn type
        for player_name, belief_prob in self.belief.items():
            self.belief_types[player_name] = np.random.choice(range(len(belief_prob)), \
                                                            p=belief_prob)

    def proposer_eval(self, state):
        pass

    def responder_eval(self, state):
        pass