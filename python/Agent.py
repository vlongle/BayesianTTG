from utils import one_hot_vector
import numpy as np
import logging

class Agent:
    def __init__(self, name, agent_type):
        self.agent_type = agent_type
        self.policy = []
        self.name = str(name) + '.' + str(self.agent_type)
        self.maiden_name = str(name)
        self.belief = {}  # key = player object, value = prob. vector over finite type
        self.proposal_values = []

    def init_belief(self, game):
        '''
        Initially, uniform prior over other agents
        '''
        self.belief[self] = one_hot_vector(self.agent_type, game.T)
        for player in game.agents:
            if player == self:
                continue
            self.belief[player] = np.array([1 / game.T for _ in range(game.T)])

        logging.debug('agent {} belief: {}'.format(self, self.belief))

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def normalize_to_prob(self, agent):
        self.belief[agent] = np.array([t/sum(self.belief[agent]) for t in self.belief[agent]])

    def draw_types(self):
        '''
        draw the types of other agents based on current belief
        '''
        belief_types = {}  # key = name, value = drawn type
        for player_name, belief_prob in self.belief.items():
            # plus one correct for the fact that our weights >= 1 but Python is zero-indexed
            belief_types[player_name] = np.random.choice(range(len(belief_prob)), \
                                                            p=belief_prob) + 1

        logging.debug('Agent {} draws {}'.format(self, belief_types))
        return belief_types

    def MLE(self, agent):
        return np.argmax(self.belief[agent])+1

    def set_belief(self, player, belief, game=None):
        if isinstance(player, str): # pass in player's name instead
            # of the object
            if not game:
                return
            player = game.agent_lookup[player]

        # belief should be a numpy array
        self.belief[player] = belief
