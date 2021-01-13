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
from OptimalBeliefAlgo import *
from softmaxSelectionStateElimination import *
from BanditAlgo import *
from VPIAlgo import *
from statsmodels.graphics.tsaplots import plot_acf
from pprint import pprint
from scipy import stats
import os

start = time.time()
SEED = 0
np.random.seed(SEED)

logging.basicConfig(level=logging.CRITICAL)

T = 4 # no. of types
N = 3 # no. of players
horizon = 1 # how many proposal rounds before a game terminate
tasks = [Task(threshold=1, reward=1),
         Task(threshold=2, reward=3),
         Task(threshold=4, reward=5.9)]


#W = [1,2,2,3,4]
W = [1,2,3]
game = Game(T, N, tasks, horizon, W)


# evaluate the rewards of different algorithms. We will plot
# the cummulative rewards at each time step for each algorithm!


n_steps = 100



start = time.time()
def run_algorithm(Algo, game, n_steps):
    # CS over time
    outcomes = []
    agents = sorted(game.agents, key=lambda agent: agent.name)
    # belief over time
    beliefs = [[[] for _ in range(len(agents))] for _ in range(len(agents))]  # list of plots!
    # Algo is an object
    algo = Algo(game)
    for t in range(n_steps):
        if t % 10 == 0:
            print('{}/{}'.format(t, n_steps))
        # softmax Q, optimalBelief, VPI, or bandit
        if Algo.__name__ == 'BanditAlgo':
            CS, proposer, proposal = algo.formation_process()
            algo.update_belief(CS, proposer, proposal) # exp3 update
        else:
            CS = algo.formation_process()
            algo.update_belief(CS) # essentially a state elimination process
        game.reset_state()
        outcomes.append(CS)
        for i, agent in enumerate(agents):
            for j, other_agent in enumerate(agents):
                beliefs[i][j].append(agent.belief[other_agent][other_agent.agent_type - game.min_type])
    return outcomes, beliefs


def plot_beliefs(agents, beliefs):
    fig, axs = plt.subplots(len(agents))
    for i, agent_plot in enumerate(beliefs):
        axs[i].set_title("Belief of agent {}".format(agents[i]))
        for j in range(len(agents)):
            print('agent', i, 'belief about', j, 'is', beliefs[i][j])
            axs[i].plot(beliefs[i][j], label='agent {}'.format(agents[j]))
        axs[i].legend(loc='lower right')
def score_individual_outcomes(outcomes):
    # for each time, store a dictionary where key=player, value=cumulative payoff to that player currently
    payoff = []
    for t, CS in enumerate(outcomes):
        payoff.append(defaultdict(float))
        for coalition, coalition_value, div, task in CS:
            for player in coalition:
                #if t == 0:
                #    prev_value = 0
                #else:
                #    prev_value = payoff[-2][player.name]
                #payoff[-1][player.name] = div * coalition_value + prev_value
                payoff[-1][str(player)] = div[coalition.index(player)] * coalition_value

    return payoff

def score_outcomes(outcomes):
    # return the running cumulative total payoff of coalition structure
    # outcomes = [CS_t]_t where each CS_t is a list of tuple of coalition and value
    cumulative_payoff = [] # index = time, value = cumulative payoff
    cum_payoff = 0
    for CS in outcomes:
        cum_payoff += sum([value for coalition, value, div, task in CS])
        cumulative_payoff.append(cum_payoff)
    return cumulative_payoff

def get_agent_payoff_series(agents, indiv_payoff):
    agents_payoff = defaultdict(list) # key=player, value = payoff time series
    for i, agent in enumerate(agents):
        agent_payoff = []
        for payoff in indiv_payoff:
            agent_payoff.append(payoff[str(agent)])
        agents_payoff[agent] = agent_payoff
    return agents_payoff

def plot_auto_correlation(agents, indiv_payoff):
    ig, axs = plt.subplots(len(agents), 2)
    for i, agent in enumerate(agents):
        agent_payoff = []
        for payoff in indiv_payoff:
            agent_payoff.append(payoff[str(agent)])

        #print('agent_payoff:', agent_payoff)
        axs[i][0].plot(agent_payoff,label='agent {}'.format(agent))
        axs[i][0].set_title('Payoff over time')
        axs[i][0].legend(loc='lower right')
        plot_acf(np.array(agent_payoff), ax=axs[i][1])


def read_outcomes(filename, sep='*'):
    outcomes = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            CS = []
            line_CS = line.split('|')[:-1] # remove the trailing \n
            #print('line_CS:', line_CS)
            for line_coalition in line_CS:
                coal_processed = line_coalition.split(sep)
                #print('coal_processed:', coal_processed)
                #print('last:', coal_processed[-1])
                len_coal = int(coal_processed[0])
                # coalition, value, div, task
                CS.append((coal_processed[1:len_coal+1], float(coal_processed[len_coal+1]),
                           np.array(coal_processed[len_coal+2:2*len_coal+2],dtype=np.float64), # division
                           coal_processed[-1]))
            #for entry in line_CS.split(','):
            #    print('entry:', entry)
            outcomes.append(CS)
    return outcomes

def write_outcomes(outcomes, filename, sep='*'):
    with open(filename, 'w') as f:
        for CS in outcomes:
            for coalition, coalition_value, div, task in CS:
                f.write(str(len(coalition)) + sep)
                f.write(sep.join(map(str, coalition)))
                f.write(sep + str(coalition_value) + sep)
                f.write(sep.join(map(str, div)))
                f.write(sep + str(task))
                f.write('|')
            f.write('\n')

def write_beliefs(beliefs, filename):
    with open(filename, 'w') as f:
        num_agents = len(beliefs[0])
        f.write(str(num_agents) + '\n')
        for belief in beliefs:
            for belief_traj in belief:
                f.write(','.join(map(str, belief_traj)))
                f.write('\n')

def read_beliefs(filename):
    print('reading beliefs from file ', filename)
    beliefs = [[]]
    if os.stat(filename).st_size == 0:
        # empty file
        return beliefs
    with open(filename, 'r') as f:
        num_agents = int(f.readline())
        i = 0
        for line in f.readlines():
            #print('line:', line)
            if i >= num_agents:
                beliefs.append([])
                i = 0
            beliefs[-1].append(list(map(float, line.split(','))))
            i += 1
    return beliefs

def get_agents_from_CS(CS):
    agents = []
    for coalition_str in CS:
        agents += coalition_str[0]
    return sorted(agents)
# TODO: write the other two algorithms!
# TODO: 1. write these stuff to csv.
# TODO: 3. migrate these code to C++ if needed
# TODO: 4. Mixing player: random subset of active players at each round!


def run_and_write_algo(Algo, game, n_steps):
    outcomes, beliefs = run_algorithm(Algo, game, n_steps)
    cum_payoff = score_outcomes(outcomes)[-1]
    print('Algo:', Algo.__name__, 'beliefs:', beliefs[0][1], cum_payoff)
    write_outcomes(outcomes, Algo.__name__ +'_outcomes.txt')
    write_beliefs(beliefs, Algo.__name__ + '_beliefs.txt')

def analyze_algo(Algo, plot=False):
    if isinstance(Algo, list):
        # data generated by C++ files
        outcome_file = Algo[0]
        belief_file = Algo[1]
    else:
        # data generated by Python code
        outcome_file = Algo.__name__ + '_outcomes.txt'
        belief_file = Algo.__name__ + '_beliefs.txt'

    outcomes = read_outcomes(outcome_file)
    beliefs = read_beliefs(belief_file)

    #print('beliefs:', beliefs[0][1])
    cum_payoff = score_outcomes(outcomes)
    print('cum_payoff:', cum_payoff[-1])
    indiv_payoff = score_individual_outcomes(outcomes)
    agents = get_agents_from_CS(outcomes[0])

    agents_payoff = get_agent_payoff_series(agents, indiv_payoff)
    for agent, series in agents_payoff.items():
        print('agent', agent, 'payoff summarize:')
        print(stats.describe(series))
    # describe the payoff vector of each agent
    if plot:
        plot_beliefs(agents, beliefs)
        plot_auto_correlation(agents, indiv_payoff)

        #plot cumulative payoff
        fig = plt.figure()
        plt.plot(cum_payoff)
        plt.title('Cumulative payoff')
        plt.show()

    return cum_payoff


def plot_cum_payoffs():
    pass



if __name__ == '__main__':
    start = time.time()

    ##TODO: to be safe, don't run all these guys at the same time.
    ## TODO: just run each of them one at the time (while commenting out the others)
    #run_and_write_algo(OptimalBeliefAlgo, game, n_steps)

    ## CPP
    #analyze_algo(['cpp_softmax_outcomes.txt', 'cpp_softmax_beliefs.txt'], True)
    #analyze_algo(['cpp_VPI_outcomes.txt', 'cpp_VPI_beliefs.txt'], True)
    #analyze_algo(['cpp_bandit_outcomes.txt', 'cpp_bandit_beliefs.txt'], True)



    optimal_payoff = analyze_algo(['cpp_opt_outcomes.txt', 'cpp_opt_beliefs.txt'])
    softmax_payoff = analyze_algo(['cpp_softmax_outcomes.txt', 'cpp_softmax_beliefs.txt'])
    VPI = analyze_algo(['cpp_VPI_outcomes.txt', 'cpp_VPI_beliefs.txt'])
    bandit = analyze_algo(['cpp_bandit_outcomes.txt', 'cpp_bandit_beliefs.txt'])

    end = time.time()

    print('takes:', end-start)
    #run_and_write_algo(softmaxSelectionStateElimination, game, n_steps)

    #soft = softmaxSelectionStateElimination(game)
    #soft.proposal_outcome_test()

    #run_and_write_algo(VPIAlgo, game, n_steps)
    #run_and_write_algo(BanditAlgo, game, n_steps)

    #softmax_payoff = analyze_algo(softmaxSelectionStateElimination)
    #optimal_payoff = analyze_algo(OptimalBeliefAlgo)
    #bandit = analyze_algo(BanditAlgo)
    #VPI = analyze_algo(VPIAlgo)

    #end = time.time()
    #print('takes:', end-start)

    #print('softmax:', softmax_payoff)
    #print('optimal:', optimal_payoff)
    #print('bandit:', bandit)
    #print('VPI:', VPI)

    #analyze_algo(BanditAlgo, True)
    #analyze_algo(softmaxSelectionStateElimination, True)
    #analyze_algo(VPIAlgo, True)
    #analyze_algo(OptimalBeliefAlgo, True)

    plt.plot(softmax_payoff, label='softmax')
    plt.plot(optimal_payoff, label='full info')
    plt.plot(bandit, label='bandit')
    plt.plot(VPI, label='VPI')
    plt.legend()

    plt.show()
