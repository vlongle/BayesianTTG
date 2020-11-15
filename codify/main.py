from fixedTypeGame import *
from Task import *
from Game import *
import logging


logging.basicConfig(level=logging.INFO)

T = 3 # no. of types
N = 3 # no. of players
horizon = 1 # how many proposal rounds before a game terminate
tasks = [Task(threshold=1, reward=1),
         Task(threshold=2, reward=3),
         Task(threshold=4, reward=5.9)]


g = Game(T, N, tasks, horizon)
state = State(g.agents, 0, g.horizon)
fixed_types = {p:p.agent_type for p in g.agents}

game_state_payoff(fixed_types, g, state)
