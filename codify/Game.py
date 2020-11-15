import numpy as np
from collections import defaultdict
from Agent import Agent
import logging

class Game:
    def __init__(self, T, N, tasks, horizon):
        self.T = T
        self.N = N
        self.tasks = tasks
        self.horizon = horizon
        self.nodes = defaultdict(set)
        self.agents = {}
        self.init_game()
        logging.info('\n ==== game.horizon: {}'.format(self.horizon))
        logging.info('game.tasks: {}'.format(self.tasks))
        logging.info('game.agents: {} \n'.format(self.agents))

    def init_game(self,seed=0):
        np.random.seed(seed)

        # chr(65)='A', chr(66)='B' and so on
        self.agents = {Agent(chr(i + 65), np.random.randint(1, self.T+1)) \
                  for i in range(self.N)}
        self.init_beliefs()

    def init_beliefs(self):
        for player in self.agents:
            player.init_belief(self)