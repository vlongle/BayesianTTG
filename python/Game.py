import numpy as np
from collections import defaultdict
from Agent import Agent
import logging
from State import State

class Game:
    def __init__(self, T, N, tasks, horizon, W=None, min_type=1):
        self.T = T # no. of types
        self.min_type = min_type # smallest weight possible for the player
        self.max_type = self.min_type + T-1
        self.N = N # no. of players
        self.tasks = self.sort_tasks(tasks)
        self.horizon = horizon # how many proposal rounds before a game terminate
        self.nodes = defaultdict(set)
        self.agents = {}
        self.agent_lookup = {} # key = player's name, value = player's object
        self.W = W
        self.init_game()
        logging.info('\n ==== game.horizon: {}'.format(self.horizon))
        logging.info('game.tasks: {}'.format(self.tasks))
        logging.info('game.agents: {} \n'.format(self.agents))


        tmp = [t.reward for t in tasks]
        self.min_reward = min(tmp)
        self.max_reward = max(tmp)
        print('min, max reward:', self.min_reward, self.max_reward)

    def sort_tasks(self, tasks):
        # sort based on the threshold weight of the tasks
        # tasks are assumed to have values increasingly monotonic with threshold
        return sorted(tasks, key=lambda task: task.threshold)
    def reset_state(self):
        self.init_state()

    def reset_belief(self):
        for agent in self.agents:
            agent.init_belief(self)
    def init_game(self,seed=100):
        np.random.seed(seed)
        if not self.W:
            # chr(65)='A', chr(66)='B' and so on
            self.agents = {Agent(chr(i + 65), np.random.randint(self.min_type, self.T+1)) \
                      for i in range(self.N)}
        else:
            self.agents =  {Agent(chr(i + 65), self.W[i]) \
                      for i in range(self.N)}
        self.agent_lookup = {agent.maiden_name :agent for agent in self.agents}
        self.init_beliefs()
        self.init_state()

    def init_beliefs(self):
        for player in self.agents:
            player.init_belief(self)

    def init_state(self):
        self.state = State(self.agents, 0, self.horizon)