#include "Game.hpp"
#include <functional>
#include <algorithm>

Game::Game(int numPlayers, int numberOfWeights, vector<Task> tasks,
           mt19937_64 &generator, const vector<int> agentWeights, int minWeight) : weightRange(numberOfWeights),
                                                                                   envs(envs)
{

    this->numPlayers = numPlayers;
    this->minWeight = minWeight;
    this->numberOfWeights = numberOfWeights;
    maxWeight = minWeight + numberOfWeights - 1;
    if (agentWeights.size() != this->numPlayers)
    {
        // randomly initialize agents
        uniform_int_distribution<int> distribution(this->minWeight, this->maxWeight);
        for (int i = 0; i < this->numPlayers; i++)
        {
            agents.push_back(Agent(i, distribution(generator)));
        }
    }
    else
    {
        // initialize agents according to the given weights
        for (int i = 0; i < this->numPlayers; i++)
        {
            agents.push_back(Agent(i, agentWeights[i]));
        }
    }

    //cout << "== Agent weight == " << endl;
    //for (auto &agent : agents)
    //{
    //    cout << agent.weight << " ";
    //}
    //cout << endl;

    // initialize beliefs for agents
    for (auto &agent : this->agents)
    {
        agent.initializeBelief(this->numPlayers, this->numberOfWeights, this->minWeight);
        this->agentWeights.push_back(agent.weight);
    }

    // sort the tasks based on threshold (which also sort them based on
    // rewards as well!)
    for (auto &tasks : envs)
    {
        sort(this->tasks.begin(), this->tasks.end(),
             [](const Task &lhs, const Task &rhs) {
                 return lhs.threshold < rhs.threshold;
             });
    }
    // this doesn't work ....
    // find max/min rewards so that we can do bandit
    // NOTE: for dynamic environments, we'd probably have to make the bandit contextual!
    for (auto &tasks : envs)
    {
        minReward = min(minReward, this->tasks[0].reward);
        maxReward = max(minReward, this->tasks[this->tasks.size() - 1].reward);
    }

    for (auto [envNum, tasks] : enumerate(envs))
    {
        // initialize rewardToThreshold and nextHigherThreshold
        for (int i = 0; i < this->tasks.size() - 1; i++)
        {
            rewardToThreshold[envNum][this->tasks[i].reward] = this->tasks[i].threshold;
            nextHigherThreshold[envNum][this->tasks[i].reward] = this->tasks[i + 1].threshold;
        }

        nextHigherThreshold[envNum][this->tasks[this->tasks.size() - 1].reward] = 100000;
        rewardToThreshold[envNum][this->tasks[this->tasks.size() - 1].reward] =
            this->tasks[this->tasks.size() - 1].threshold;
        rewardToThreshold[envNum][0] = -100000; // reward is always strictly bigger than 0. Reward = 0
                                        // is when the agents' weights are so low that no tasks are feasible
    }

    !

        // generate all division rules once so that agents can re-use those
        for (int i = 1; i <= numPlayers; i++)
    {
        divisionRules[i] = generateDivisionRule(i);
    }

    // initialize proposal space for agents
    for (auto &agent : agents)
    {
        agent.initializeProposalSpace(numPlayers, divisionRules);
    }
    iota(weightRange.begin(), weightRange.end(), minWeight);

    weightRangeVec = VectorXd::LinSpaced(numberOfWeights, minWeight, maxWeight);
    // debug
    //cout << "game weighRangeVec\n " << weightRangeVec << endl;
}

double Game::evaluateCoalition(vector<int> weights, int envNum)
{
    //http://www.cplusplus.com/reference/algorithm/upper_bound/
    int totalWeight = std::accumulate(weights.begin(), weights.end(), 0);
    auto& tasks = envs[envNum];
    // base-case: totalWeight is too large
    if (totalWeight > tasks[tasks.size() - 1].threshold)
    {
        return tasks[tasks.size() - 1].reward;
    }
    // I think the below code should already handle this case but just to be safe
    // check if totalWeight is too small
    if (totalWeight < tasks[0].threshold)
    {
        return 0;
    }

    // upper_bound return the first task whose threshold is strictly
    // larger than our weight. We decrease by 1 to get
    // the first task whose threshold is less than or equal to
    // our weight
    auto bestTask = std::upper_bound(tasks.begin(), tasks.end(), totalWeight,
                                     [](const int &lhs, const Task &rhs) {
                                         return lhs < rhs.threshold;
                                     });

    //bestTask = prev(bestTask);

    return tasks[bestTask - tasks.begin() - 1].reward;
}

double Game::expectedCoalitionValue(Agent &predictor, Coalition coalition, int envNum)
{
    //cout << "expectedCoalitionValue begin!" << endl;
    double ret = 0;
    for (auto &weights : CartesianSelfProduct(weightRange, coalition.size()))
    {
        double prob = 1.0; // probability of this weight config according
        // to predictor's belief
        for (auto [i, agentName] : enumerate(coalition))
        {
            prob *= predictor.belief(agentName, weights[i] - minWeight);
        }
        ret += prob * evaluateCoalition(weights, envNum);
    }

    //cout << "expectedCoalitionValue done!" << endl;
    return ret;
}

double Game::expectedSingletonValue(Agent &predictor, int agentName, int envNum)
{
    double ret = 0;
    for (int weight : weightRange)
    {
        ret += predictor.belief(agentName, weight - minWeight) * evaluateCoalition({weight}, envNum);
    }
    //cout << "expectedSingletonValue of  " << agentName << " according to " << predictor.name
    //<< " is " << ret << endl;
    return ret;
}

// response: "yes" --> 1, "no" --> 0
pair<double, map<int, int>> Game::predictReponses(Agent &predictor, Proposal &proposal, set<int> predictees, int envNum)
{
    map<int, int> responses;
    double expectedCoalValue = expectedCoalitionValue(predictor, proposal.first, envNum);
    for (auto [i, agentName] : enumerate(proposal.first))
    {
        if (predictees.size() > 0 && predictees.find(agentName) == predictees.end())
        {
            // this agent is NOT in the list of predictees!
            continue;
        }
        double gainJoining = expectedCoalValue * proposal.second[i];

        double gainRefuse = expectedSingletonValue(predictor, agentName, envNum);
        //cout << "Predictor " << predictor.name << " for predictee " << agentName << " div " << proposal.second;
        //cout << "\n expectedCoalVal " << expectedCoalValue << " gainJoining " << gainJoining << " gainRefuse "
        //<< gainRefuse << endl;
        if (gainJoining >= gainRefuse)
        {
            responses.insert(pair<int, int>(agentName, 1));
        }
        else
        {
            responses.insert(pair<int, int>(agentName, 0));
        }

        //cout << "predictResponse by " << predictor.name << " for " << agentName
        //<< " for proposal " << proposal.second << " is " << responses[agentName] << endl;
    }
    //cout << "predictResponses done!" << endl;
    return make_pair(expectedCoalValue, responses);
}

double Game::evaluateCoalition(Coalition &coalition, int envNum)
{
    vector<int> weights = {};
    for (int agentName : coalition)
    {
        weights.push_back(agents[agentName].weight);
    }
    return evaluateCoalition(weights, int envNum);
}

// TODO: have to change the runAlgorithms in main.cpp to save the CS correctly!
void Game::updateWealth(CoalitionStructure &CS)
{
    for (auto &coalitionInfo : CS)
    {
        Coalition &coalition = get<0>(coalitionInfo);
        VectorXd &div = get<1>(coalitionInfo);
        double coalitionValue = get<2>(coalitionInfo);
        for (auto [i, agentName] : enumerate(coalition))
        {
            auto &agent = agents[agentName];
            agent.currentWealth += div[i] * coalitionValue;
        }
    }
}

// for each agent, count the inversions of the mean weight prediction
// compared to the actual sorting of players' weight
double Game::countCurrentAvgInversions()
{
    double totInversions = 0;
    for (auto &agent : agents)
    {
        VectorXd agentMeanPrediction = agent.belief * weightRangeVec;
        vector<double> agentMeanPred(agentMeanPrediction.data(), agentMeanPrediction.data() +
                                                                     agentMeanPrediction.size());
        totInversions += countInversions(agentWeights, agentMeanPred);
    }
    return totInversions / numPlayers;
}

// same as countCurrentAvgInversion except here the candidate is NOT agent's mean-prediction but the
// wealth signal.
double Game::computeCurrentSignalFidelity()
{
    vector<double> currentSignal;
    for (auto &agent : agents)
    {
        currentSignal.push_back(agent.currentWealth);
    }

    return countInversions(agentWeights, currentSignal);
}
