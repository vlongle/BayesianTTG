from evaluateAlgorithms import *




def plot_trials(files):
    trials = []
    for file in files:
        outcomes = read_outcomes(file)
        cum_payoff = score_outcomes(outcomes)
        trials.append(cum_payoff)

    trials = np.array(trials)
    trials_mean = np.mean(trials, axis=0) # take mean along the column
    std = np.std(trials, axis=0)
    print('trials_mean:', trials_mean[-1])

    # plot 2 standard deviation
    plt.plot(trials_mean)
    n_time_step = trials.shape[1]
    #print('std:', std)
    #print('trials[0]:', trials[0])
    #print('trials[1]:', trials[1])
    #print('trials[0]-trials[1]:', trials[0]-trials[1])
    #plt.fill_between(range(n_time_step), (trials_mean-2*std), (trials_mean+2*std), alpha=0.1)



#n_trials = 30
#files = []
#for i in range(n_trials):
#    files.append('data/cpp_softmax_outcomes_' + str(i) + '.txt')

n_trials = 30
n_trust_levels = 6
for i in range(n_trust_levels):
    files = ['data/cpp_softmax_outcomes_exploit_' + str(i) + '_' + str(j) + '.txt' for j
             in range(n_trials)]
    plot_trials(files)


plt.show()
