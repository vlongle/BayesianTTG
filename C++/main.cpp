#include "Game.hpp"
#include <iostream>
#include "simplex_grid.hpp"
#include <set>
#include "InformedBeliefAlgo.hpp"
#include "SoftmaxQ.hpp"
#include "SignalSoftmaxQ.hpp"
#include "VPIAlgo.hpp"
#include "Bandit.hpp"
#include <string>
#include <fstream>
#include <time.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>

using namespace std::chrono;

//#include <Eigen/CXX11/Tensor>

using namespace std;

float SEED = 1;
int numSteps = 100;

pair<Tensor<double, 3>, vector<CoalitionStructure>> runAlgorithm(Algo &algo, int numSteps, bool exploitSignal = false,
                                                                 double trustLevel = 0.0)
{
    //cout << "Running algorithm!" << endl;

    vector<CoalitionStructure> outcomes;

    int n = algo.game.numPlayers;

    //beliefs[i][j][t]: belief of agent i about j at time t
    //vector<vector<vector<double>>> beliefs(n);
    // tensor since vector was giving weird segfault
    Tensor<double, 3> beliefs(n, n, numSteps);

    //maybe not necessary if each algorithm constructed its own game
    //We need that for parallelization anyway!
    //algo.game.reset_belief();
    for (int t = 0; t < numSteps; t++)
    {
        cout << "t " << t << endl;
        //if (t % 10 == 0)
        //{
        //    cout << t << "/" << numSteps << endl;
        //}

        // formationProcess ==> outcome-based update ==> update wealth ==>
        // belief update based on wealth ==> store result ==> repeat.
        pair<CoalitionStructure, vector<Coalition>> CS_and_coalition = algo.formationProcess();
        CoalitionStructure &CS = CS_and_coalition.first;
        algo.updateBelief(CS_and_coalition.second);
        algo.game.updateWealth(CS);
        if (exploitSignal)
        {
            algo.exploitSignal(trustLevel);
        }
        outcomes.push_back(CS);
        for (auto &agent : algo.game.agents)
        {
            for (auto &otherAgent : algo.game.agents)
            {

                beliefs(agent.name, otherAgent.name, t) =
                    agent.belief(otherAgent.name,
                                 otherAgent.weight - algo.game.minWeight);
            }
        }
    }

    return make_pair(beliefs, outcomes);
}

void writeOutcomes(vector<CoalitionStructure> outcomes, string fileName,
                   string sep = "*")
{
    ofstream out(fileName);
    // format: each line is numberOfAgents, agents, coalitionValue, division, task | ...
    for (auto &outcome : outcomes)
    {
        for (auto &coalitionPlus : outcome)
        {
            Coalition &coalition = get<0>(coalitionPlus);
            VectorXd &divisionRule = get<1>(coalitionPlus);
            double coalitionVal = get<2>(coalitionPlus);
            out << coalition.size() << sep;
            for (auto &agent : coalition)
            {
                out << agent << sep;
            }
            // cout << "coalitionVal:" << coalitionVal << endl;
            out << coalitionVal << sep;
            for (int i = 0; i < divisionRule.size(); i++)
            {
                out << divisionRule[i] << sep;
            }
            out << "Task(C++ NONE)";
            out << "|"; // separate between different coalitions
        }
        out << "\n"; // separate between different time steps
    }
    out.close();
}

void writeBeliefs(Tensor<double, 3> beliefs, string fileName, int numSteps, int numPlayers)
{
    ofstream out(fileName);
    out << numPlayers << endl;
    for (int i = 0; i < numPlayers; i++)
    {
        for (int j = 0; j < numPlayers; j++)
        {
            for (int t = 0; t < numSteps; t++)
            {
                // ugly code. Eww! Avoid writing deliminator ',' at the end! (Python is better at this!)
                if (t == numSteps - 1)
                {
                    out << beliefs(i, j, t);
                    continue;
                }
                out << beliefs(i, j, t) << ",";
            }
            out << endl;
        }
    }
    out.close();
}

void OneExperiment()
{
    mt19937_64 generator(SEED);
    vector<Task> tasks = {
        //{8, 10},
        {1, 1},
        {4, 5.9},
        {2, 3},
    };
    int numPlayers = 5;
    int numWeights = 4;
    //vector<int> agentWeights = {1, 2, 3};
    vector<int> agentWeights = {1, 1, 2, 3, 4};
    Game game(numPlayers, numWeights, tasks, generator, agentWeights);

    //Game game(numPlayers, numWeights, tasks, generator);

    //InformedBeliefAlgo inform(game, generator);
    SoftmaxQ softmaxQ(game, generator);
    // VPIAlgo VPI(game, generator);
    //Bandit bandit(game, generator);

    //softmaxQ.testProposal();
    //VectorXd v(2);
    //v << 0.4, 0.6;
    //Proposal proposal = make_pair(set<int>{1,2}, v);
    //map<int, int> resp = game.predictReponses(game.agents[2], proposal).second;
    //cout << "resp: " << resp[2] << endl;

    //VectorXd v(2);
    //v << 0.4, 0.6;
    //Proposal proposal = make_pair(set<int>{0,1}, v);
    //map<int, int> resp = game.predictReponses(game.agents[0], proposal).second;
    //cout << "resp: " << resp[0] << endl;

    auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps);

    // verify current wealth of players!
    for (auto &agent : game.agents)
    {
        cout << "agent " << agent.name << " wealth " << agent.currentWealth << endl;
    }

    //auto beliefOutcomes = runAlgorithm(inform, numSteps);
    // auto beliefOutcomes = runAlgorithm(VPI, numSteps);
    //auto beliefOutcomes = runAlgorithm(bandit, numSteps);

    //writeOutcomes(beliefOutcomes.second, "cpp_opt_outcomes.txt");
    //writeBeliefs(beliefOutcomes.first, "cpp_opt_beliefs.txt", numSteps, numPlayers);

    writeOutcomes(beliefOutcomes.second, "./data/cpp_softmax_outcomes.txt");
    //writeBeliefs(beliefOutcomes.first, "cpp_softmax_beliefs.txt", numSteps, numPlayers);

    //writeOutcomes(beliefOutcomes.second, "cpp_VPI_outcomes.txt");
    //writeBeliefs(beliefOutcomes.first, "cpp_VPI_beliefs.txt", numSteps, numPlayers);

    //writeOutcomes(beliefOutcomes.second, "cpp_bandit_outcomes.txt");
    //writeBeliefs(beliefOutcomes.first, "cpp_bandit_beliefs.txt", numSteps, numPlayers);
}

void OneExploitSignalExperiment(double trustLevel, string filename, double seed, vector<int> agentWeights)
{
    mt19937_64 generator(seed);
    vector<Task> tasks = {
        {1, 1},
        {4, 5.9},
        {2, 3},
    };
    int numPlayers = 5;
    int numWeights = 4;
    Game game(numPlayers, numWeights, tasks, generator, agentWeights);
    SoftmaxQ softmaxQ(game, generator);
    auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps, true, trustLevel);
    //writeOutcomes(beliefOutcomes.second, "./data/cpp_softmax_outcomes_exploit.txt");
    writeOutcomes(beliefOutcomes.second, filename);
}
void bulkExperiment(int numTrials, int numPlayers, int numWeights, vector<int> &agentWeights, vector<Task> &tasks)
{
#pragma omp parallel for
    for (int i = 0; i < numTrials; i++)
    {
        mt19937_64 generator(i);
        Game game(numPlayers, numWeights, tasks, generator, agentWeights);
        SoftmaxQ softmaxQ(game, generator);
        auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps);
        writeOutcomes(beliefOutcomes.second, "./data/cpp_softmax_outcomes_" + to_string(i) + ".txt");
        writeBeliefs(beliefOutcomes.first, "./data/cpp_softmax_beliefs_" + to_string(i) + ".txt", numSteps, numPlayers);
    }
}
//int main()
//{
//auto start = high_resolution_clock::now();

//// vector<Task> tasks = {
////     {1, 1},
////     {4, 5.9},
////     {2, 3},
//// };
//// int numPlayers = 5;
//// int numWeights = 4;
//// // deliberately give fewer agentWeights than required to trigger random weight initialization!
//// vector<int> agentWeights = {1, 1, 2, 2, 3};
//// int numTrials = 30;

//// bulkExperiment(numTrials, numPlayers, numWeights, agentWeights, tasks);

////OneExperiment();

//vector<double> trustLevels = {
//0.0,
//0.1,
//0.5,
//1.0,
//10.0,
//20.0};

//// vector<double> trustLevels = {
////     0.0,
////     0.1,
////     0.5};

//// deliberately provide unsufficient weights to trigger random weights initialization!
//vector<int> agentWeights = {1, 1, 2, 3};
//int numTrials = 30;

//#pragma omp parallel for num_threads(8) collapse(2)
//for (int i = 0; i < trustLevels.size(); i++)
//{
//{
//for (int j = 0; j < numTrials; j++)
//{
//OneExploitSignalExperiment(trustLevels[i], "./data/cpp_softmax_outcomes_exploit_" + to_string(i) + "_" + to_string(j) + ".txt",
//j, // seed
//agentWeights);
//}
//}
//}

//auto stop = high_resolution_clock::now();

//cout << "took " << duration_cast<seconds>(stop - start).count() << " seconds " << endl;

//}

void testRegularizer()
{
    cout << "testing regularizer" << endl;
    vector<Task> tasks = {
        {1, 1},
        {4, 5.9},
        {2, 3},
    };

    mt19937_64 generator(SEED);
    int numPlayers = 3;
    int numWeights = 4;
    Game game(numPlayers, numWeights, tasks, generator);

    // set currentWealth as custom
    game.agents[0].currentWealth = 1;
    game.agents[1].currentWealth = 3;
    game.agents[2].currentWealth = 2;

    SignalSoftmaxQ softmaxQ(game, generator);
    auto beliefOutcomes = runAlgorithm(softmaxQ, 1);
    //writeOutcomes(beliefOutcomes.second, "./data/cpp_softmax_outcomes_exploit.txt");
    // writeOutcomes(beliefOutcomes.second, filename);
}
int main()
{
    testRegularizer();
}