from fixedTypeGame import *
from Task import *
from Game import *
import logging
from BayesianAlternatingDynamics import *
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
plt.style.use('ggplot')
import time


# TODO: horizon = 2 logic is flawed!!
start = time.time()
np.random.seed(100)

logging.basicConfig(level=logging.INFO)

T = 2 # no. of types
N = 2 # no. of players
horizon = 1 # how many proposal rounds before a game terminate
tasks = [Task(threshold=1, reward=1),
         Task(threshold=2, reward=3),
         Task(threshold=4, reward=5.9)]


W = [1,1]
game = Game(T, N, tasks, horizon, W)


n_learnings = 100
agents = sorted(game.agents, key=lambda agent:agent.agent_type)
agent_plots = [[[] for _ in range(len(agents))] for _ in range(len(agents))] # list of plots!
# each agent plot is curves of prob. of the true type of other agent!
for i in range(n_learnings):
    if i % 20 == 0:
        print('completing {}/{}'.format(i+1, n_learnings))
    alternating_dynamics(game)
    game.reset()
    for i, agent in enumerate(agents):
        for j, other_agent in enumerate(agents):
            agent_plots[i][j].append(agent.belief[other_agent][other_agent.agent_type-1])


end = time.time()
print('take=', end-start)

fig, axs = plt.subplots(len(agents))
for i, agent_plot in enumerate(agent_plots):
    axs[i].set_title("Belief of agent {}".format(agents[i]))
    for j in range(len(agents)):
        axs[i].plot(agent_plot[j], label='agent {}'.format(agents[j]))
    axs[i].legend(loc='lower right')

plt.show()

