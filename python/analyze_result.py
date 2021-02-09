from evaluateAlgorithms import *
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5
rcParams["legend.loc"] = 'lower right'



def plot_trials(trials, label):
    trials = np.array(trials)
    trials_mean = np.mean(trials, axis=0) # take mean along the column
    std = np.std(trials, axis=0)
    #print('trials[:, 10]:', trials[:, 10])
    #print('trials_mean:', trials_mean[-1])

    #print('std:', std.shape)
    #print('std\n:', std)
    # plot 1 standard deviation
    #print('plotting')

    print('trials_mean:', trials_mean[-1])
    plt.plot(trials_mean, label=label)
    n_time_step = trials.shape[1]
    #plt.fill_between(range(n_time_step), (trials_mean-std), (trials_mean+std),alpha=0.1)
    plt.xticks(np.arange(0, len(trials_mean), 10))
    plt.xlabel('Time step')


def meta_analyze(filename_prefix, ntrials):
    outcome_files = [filename_prefix + '_outcomes_' + str(i) + '.txt' for i in range(ntrials)]
    belief_files = [filename_prefix + '_beliefs_' + str(i) + '.txt' for i in range(ntrials)]
    inversion_files = [filename_prefix + '_inversions_' + str(i) + '.txt' for i in range(ntrials)]
    outcome_trials = []
    for outcome_f in outcome_files:
        outcomes = read_outcomes(outcome_f)
        outcome_trials.append(score_outcomes(outcomes))


    belief_trials = []
    for belief_f in belief_files:
        beliefs = read_beliefs(belief_f)
        belief_trials.append(np.mean(beliefs, axis=(0, 1)))

    inversions_trials = []
    for inv_f in inversion_files:
        inv = np.loadtxt(inv_f)
        inversions_trials.append(inv)

    return outcome_trials, belief_trials, inversions_trials


def compare_algos(filename_prefix_list, labels, ntrials):
    if len(filename_prefix_list) != len(labels):
        print('len(filename_prefix_list) != len(labels)')
        return
    outcome_algos = []
    belief_algos = []
    inversions_algos = []

    for f in filename_prefix_list:
        out, belief, inv = meta_analyze(f, ntrials)
        outcome_algos.append(out)
        belief_algos.append(belief)
        inversions_algos.append(inv)

    plt.figure(1)
    print('outcomes\n')
    for outcome_trials, l in zip(outcome_algos, labels):
        plot_trials(outcome_trials, label=l)
        outcome_trials = np.array(outcome_trials)
        #print('outcome_trials:', outcome_trials[:, 19])

    plt.legend()
    plt.title('Cumulative social welfare of 5 agents (30 trials)')
    plt.savefig('payoff_5agents.png')

    plt.figure(2)
    print('\n \n Belief strength:\n')
    for belief_trials, l in zip(belief_algos, labels):
        if 'bandit' in l:
            continue
        plot_trials(belief_trials, l)
        #print('belief_trials at 0 avg:', np.mean(np.array(belief_trials), axis=0))
        #print('belief_trials at 0 sum:', np.sum(np.array(belief_trials), axis=0))
        #print('belief_trials:', np.argwhere(np.isnan(belief_trials)))

    plt.legend()
    plt.title('Belief strength of 5 agents (30 trials)')
    plt.savefig('belief_5agents.png')

    print('\n \n invCount:\n')
    plt.figure(3)
    for inversions_trials, l in zip(inversions_algos, labels):
        if 'bandit' in l:
            continue
        plot_trials(inversions_trials, l)

    plt.title('Inversion Count of 5 agents (30 trials)')
    plt.legend()
    plt.savefig('inv_5agents.png')


    # plt.figure(1)
    # plot_trials(outcome_trials, 'Softmax social welfare for 5 agents (30 trials)', 'Cumulative social welfare', 'r')
    #
    # plt.figure(2)
    # plot_trials(belief_trials, 'Softmax belief strength for 5 agents (30 trials)', 'Avg belief strength', 'b')
    #
    # plt.figure(3)
    # plot_trials(inversions_trials, 'Softmax belief inversion count for 5 agents (30 trials)', 'Avg inversion count',
    #             'g')

#n_trials = 30
#files = []
#for i in range(n_trials):
#    files.append('data/cpp_softmax_outcomes_' + str(i) + '.txt')

#n_trials = 10
#n_trust_levels = 6
#for i in range(n_trust_levels):
#    #files = ['data/cpp_softmax_outcomes_exploit_' + str(i) + '_' + str(j) + '.txt' for j
#    #         in range(n_trials)]
#    files = ['data/cpp_softmax_outcomes_regularizer_' + str(i) + '_' + str(j) + '.txt' for j
#             in range(n_trials)]
#    plot_trials(files)
#


# outcome_file = './caseStudyData/cpp_softmax_outcomes_small.txt'
# #outcome_file = './caseStudyData/cpp_softmax_exploit_outcomes_small.txt'
#
# files = [outcome_file]
# plot_trials(files)
#
# beliefs = read_beliefs('./caseStudyData/cpp_softmax_beliefs_small.txt')
# #beliefs = read_beliefs('./caseStudyData/cpp_softmax_exploit_beliefs_small.txt')
# beliefs = np.array(beliefs)
#
# belief_strength_mean = np.mean(beliefs, axis=(0, 1))
#
# outcomes = read_outcomes(outcome_file)
# agents = get_agents_from_CS(outcomes[0])
#
# plot_beliefs(agents, beliefs)
# plt.figure(3)
# plt.plot(belief_strength_mean)
# print('belief_strength_mean:', belief_strength_mean)
#
# cum_payoff = score_outcomes(outcomes)
# print('final payoff:', cum_payoff[-1])
# payoffs = [cum_payoff[0]] + [cum_payoff[i] - cum_payoff[i-1] for i in range(1, len(cum_payoff))]
# print('payoffs:\n', payoffs)
#
# proposers = np.loadtxt('./caseStudyData/cpp_softmax_proposer_small.txt')
# #proposers = np.loadtxt('./caseStudyData/cpp_softmax_exploit_proposer_small.txt')
#
# wealth = np.loadtxt('./caseStudyData/cpp_softmax_exploit_wealth_small.txt',
#                     delimiter='*')
#
# inversions = np.loadtxt('./caseStudyData/cpp_inform_inversions_small.txt')
#
# print('inversions:', inversions)
# print('proposers:\n', proposers)
# print('wealth:\n', wealth)
# plt.show()

if __name__ == '__main__':

    dirs = [
            # inform
            './data_5agents/informed/cpp_informed',

            # softmax
            './data_5agents/softmax/cpp_softmax',

            # injection
            # './data_5agents/0.1_injection/cpp_injection',
            # './data_5agents/0.5_injection/cpp_injection',
            # './data_5agents/1.0_injection/cpp_injection',
             './data_5agents/5.0_injection/cpp_injection',

            # softmax regularizer
            # './data_5agents/0.1_regularizer/cpp_regularizer',
            # './data_5agents/0.5_regularizer/cpp_regularizer',
            # './data_5agents/1.0_regularizer/cpp_regularizer',
            # './data_5agents/5.0_regularizer/cpp_regularizer',
            # './data_5agents/10.0_regularizer/cpp_regularizer',
             './data_5agents/20.0_regularizer/cpp_regularizer',

         # bandit
        './data_5agents/bandit/cpp_bandit',

         # bandit regularizer
         # './data_5agents/0.1_regularizedBandit/cpp_regularizedBandit',
         # './data_5agents/0.5_regularizedBandit/cpp_regularizedBandit',
         './data_5agents/1.0_regularizedBandit/cpp_regularizedBandit',
         # './data_5agents/5.0_regularizedBandit/cpp_regularizedBandit',
         # './data_5agents/10.0_regularizedBandit/cpp_regularizedBandit',
         # './data_5agents/20.0_regularizedBandit/cpp_regularizedBandit',

            # bigger c cause numerical instability
            # './data_5agents/10.0_injection/cpp_injection',
            # './data_5agents/20.0_injection/cpp_injection',
            ]

    labels = [
               'informed',
               'softmax',

               # injection
              # 'injection with $c=0.1$',
              # 'injection with $c=0.5$',
              # 'injection with $c=1.0$',
              'injection with $c=5.0$',
              # 'injection with $c=10.0$',
              # 'injection with $c=20.0$',

            # softmax regularizer
            #  'softmax regularizer with $c=0.1$',
            #  'softmax regularizer with $c=0.5$',
            #  'softmax regularizer with $c=1.0$',
            #  'softmax regularizer with $c=5.0$',
            #  'softmax regularizer with $c=10.0$',
             'softmax regularizer with $c=20.0$',

            # bandit
            'bandit',

            # bandit regularizer
            # 'bandit regularizer with $c=0.1$',
            # 'bandit regularizer with $c=0.5$',
            'bandit regularizer with $c=1.0$',
            # 'bandit regularizer with $c=5.0$',
            # 'bandit regularizer with $c=10.0$',
            # 'bandit regularizer with $c=20.0$',
              ]
    compare_algos(dirs, labels, 30)
    plt.show()

    #soft_out = read_outcomes('./caseStudyData/cpp_softmax_outcomes_small.txt')
    #exploit_out =read_outcomes('./caseStudyData/cpp_softmax_exploit_outcomes_small.txt')

    #exploit_out =read_outcomes('./data_5agents/injection/cpp_injection_outcomes_0.txt')
    #exploit_out =read_outcomes('./caseStudyData/again/cpp_injection_outcomes_0.txt')
    #exploit_out =read_outcomes('./caseStudyData/two_trial_outcomes_0.txt')
    #exploit_out =read_outcomes('./caseStudyData/thirty_trial_outcomes_0.txt')



    #
    # print('soft_out:', soft_out)
    # print('exploit_out:', exploit_out)
    #soft = score_outcomes(soft_out)
    #exploit = score_outcomes(exploit_out)
    #print('soft:', soft[-1])
    #print('exploit:', exploit[-1])
