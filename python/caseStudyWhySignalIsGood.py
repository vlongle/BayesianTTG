'''
Soft regularizer is defined as follows.
    proposalValue = QValue + lambda * justice
where
    justice = 1/(1 + invCount)


1. Look at acceptance rate first!
'''

from analyze_result import *


def plot_acceptance(file, label, plot=True):
    acceptance = np.loadtxt(file)
    accept_rate = np.cumsum(acceptance)
    accept_rate = [x/(i+1) for i, x in enumerate(accept_rate)]

    if plot:
        plt.plot(accept_rate, label=label)
        plt.xlabel('Time step')
    return accept_rate

def plot_wealth(file, label):
    # CRITICAL: for this VERY specific study!
    wealth = np.loadtxt(file, delimiter='*')
    wealth = wealth[:, 1] - wealth[:, 0]

    plt.plot(wealth, label=label)
    plt.xlabel('Time step')
    return wealth

def case_study():
    # !! NOTE: the wealth_.txt file should have no trailing *. We need to use
    # the script remove_last_character_from_each_line.sh from the ../C++/ folder
    # to remove the character * from the end of each line.
    folder = './caseStudyWhySignalIsGood/'
    soft_file = folder + 'softmax/soft_subadditive'
    signal_file = folder + 'signal/signal_subadditive'
    plt.figure(1)
    soft_acceptance = plot_acceptance(soft_file + '_acceptance_.txt',
                                      'softmax acceptance rate')
    signal_acceptance = plot_acceptance(signal_file + '_acceptance_.txt',
                                        'regularized softmax acceptance rate')

    plt.title('Case study: acceptance rate of subadditive game')
    plt.legend()
    plt.figure(2)

    soft_wealth = plot_wealth(soft_file + '_wealth_.txt', 'softmax wealth difference')
    signal_wealth = plot_wealth(signal_file + '_wealth_.txt', 'regularized wealth difference')

    plt.title('Case study: Wealth difference')
    plt.legend()
    plt.show()


def plot_acceptance_trial(prefix, ntrials, label):
    files = [prefix+ '_acceptance_' + str(i) + '.txt' for i in range(ntrials)]
    acceptances = []
    for f in files:
        acceptances.append(plot_acceptance(f, '', False))
    acceptances = np.array(acceptances)
    mean_path = np.mean(acceptances, axis=0)
    plt.plot(mean_path, label=label)

def compare_algos_accept_fidelity(filename_prefix_list, labels, ntrials):
    if len(filename_prefix_list) != len(labels):
        print('len(filename_prefix_list) != len(labels)')
        return

    fidelity_algos = []
    plt.figure(1)
    for f, l in zip(filename_prefix_list, labels):
        plot_acceptance_trial(f, ntrials, l)

        files = [f + '_signalFidelity_' + str(i) + '.txt' for i in range(ntrials)]

        fidelity_algos.append([np.loadtxt(file) for file in files])

    plt.legend()
    plt.title('Average Acceptance Rate of 5 agents setting over 30 trials')

    plt.figure(2)
    for fidelity, l in zip(fidelity_algos, labels):
        print(np.array(fidelity).shape)
        plot_trials(fidelity, l)

    plt.legend()
    plt.title('Average Inversion Count for Wealth of 5 agents setting over 30 trials')

def compare_algos_why_signal_is_good():
    ntrials = 30
    prefix_list = [
        './data_5agents/softmax/cpp_softmax',
        './data_5agents/5.0_injection/cpp_injection',
    './data_5agents/5.0_regularizer/cpp_regularizer',
    ]

    labels = [
        'softmax',
        'injection with $c=5.0$',
    'softmax regularizer with $c=5.0$',
    ]

    compare_algos_accept_fidelity(prefix_list, labels, ntrials)



    plt.show()


case_study()

## sanity check about coalition size!
#out = read_outcomes('./data_5agents/softmax/cpp_softmax_outcomes_1.txt')
#coalition_sizes = [len(outcome) for outcome in out]
#print('coalition sizes:', Counter(coalition_sizes))
