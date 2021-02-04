#pragma once
#include "Agent.hpp"
#include "Task.hpp"
#include <vector>
#include <random>
#include <limits>
#include <map>
#include "Utils.hpp"
using namespace std;

class Game
{
public:
    // task's threshold and reward should be proportional!
    // don't share tasks and agentWeights so that we don't have parallel access!
    Game(int numPlayers, int numberOfWeights, vector<Task> tasks, mt19937_64 &generator,
         const vector<int> agentWeights = vector<int>(), int minWeight = 1);
    int numPlayers;
    int minWeight;
    int numberOfWeights;
    int maxWeight;
    double minReward;
    double maxReward;

    vector<double> agentWeights; // vector of double for coding convenience
    vector<Agent> agents;
    vector<Task> tasks;
    map<int, divisionRule> divisionRules; // key = coalition size,
    // value = all the simplex points

    // key = agent, value = ("yes"->1, "no"->0)
    pair<double, map<int, int>> predictReponses(Agent &predictor, Proposal &proposal, set<int> predictees = {});
    // In Python, range(minWeight, maxWeight+1)
    vector<int> weightRange;
    double expectedCoalitionValue(Agent &predictor, Coalition coalition);
    double evaluateCoalition(vector<int> weights);
    double evaluateCoalition(Coalition &coalition);
    double expectedSingletonValue(Agent &predictor, int agentName);
    // update agents' wealth signals
    void updateWealth(CoalitionStructure &CS);

    // assuming that the reward to threshold map is 1-1
    map<double, int> rewardToThreshold;
    // given the reward, gives the next higher threshold. If this reward
    // is already the highest, then output 1000
    map<int, double> nextHigherThreshold;

    // a hack to save a list of proposers quickly. Sorry not sorry.
    vector<int> proposerList;
    vector<Proposal> proposals;

    VectorXd weightRangeVec;
    double countCurrentAvgInversions();
    double computeCurrentSignalFidelity();

    // accept = 1, reject = 0. Keep track of acceptance of proposal over time
    vector<int> acceptances;
};
