#include "Game.hpp"
#include <iostream>
#include "simplex_grid.hpp"
#include <set>
#include "InformedBeliefAlgo.hpp"
#include "SoftmaxQ.hpp"
#include "SignalSoftmaxQ.hpp"
#include "SignalBandit.hpp"
#include "VPIAlgo.hpp"
#include "Bandit.hpp"
#include <string>
#include <fstream>
#include <time.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>
#include <omp.h>

using namespace std::chrono;

//#include <Eigen/CXX11/Tensor>

using namespace std;

float SEED = 3;
int numSteps = 100;

double calculateCumulativePayoff(vector<CoalitionStructure> outcomes)
{

    double res = 0;
    for (auto &CS : outcomes)
    {
        for (auto &coalitionInfo : CS)
        {
            res += get<2>(coalitionInfo);
        }
    }
    return res;
}
tuple<Tensor<double, 3>, vector<CoalitionStructure>, vector<vector<double>>,
      vector<double>, vector<double>>
runAlgorithm(Algo &algo, int numSteps, bool exploitSignal = false,
             double trustLevel = 0.0)
{
    //cout << "Running algorithm!" << endl;

    vector<CoalitionStructure> outcomes;
    vector<vector<double>> wealth(numSteps); // wealth[t][player]
    vector<double> avgInversions;
    vector<double> signalFidelity;

    int n = algo.game.numPlayers;

    //beliefs[i][j][t]: belief of agent i about j at time t
    //vector<vector<vector<double>>> beliefs(n);
    // tensor since vector was giving weird segfault
    Tensor<double, 3> beliefs(n, n, numSteps);

    // TODO: Resetting belief might not be needed
    // algo.game.reset_belief();
    for (int t = 0; t < numSteps; t++)
    {
        if (t % 10 == 0)
        {
           cout << t << "/" << numSteps << endl;
        }

        // formationProcess ==> outcome-based update ==> update wealth ==>
        // belief update based on wealth ==> store result ==> repeat.
        pair<CoalitionStructure, vector<Coalition>> CS_and_coalition = algo.formationProcess();
        CoalitionStructure &CS = CS_and_coalition.first;
        algo.updateBelief(CS_and_coalition.second);
        // record wealth as well ...

        algo.game.updateWealth(CS);

        for (int i = 0; i < algo.game.numPlayers; i++)
        {
            wealth[t].push_back(algo.game.agents[i].currentWealth);
        }

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

        avgInversions.push_back(algo.game.countCurrentAvgInversions());
        signalFidelity.push_back(algo.game.computeCurrentSignalFidelity());
    }

    // // debug wealth
    // for (int i = 0; i < algo.game.numPlayers; i++)
    // {
    //     cout << " wealth " << wealth[wealth.size() - 1][i] << endl;
    // }
    // cout << "=============================================>" << endl;
    return make_tuple(beliefs, outcomes, wealth, avgInversions, signalFidelity);
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

void writeWealth(vector<vector<double>> wealth, string filename, string sep = "*")
{
    // each line is a time step. Each line contains of "numPlayers" of entries separated by *
    ofstream out(filename);
    for (auto &curWealth : wealth)
    {
        for (auto curAgentWealth : curWealth)
        {
            out << curAgentWealth << sep;
        }
        out << "\n";
    }
    out.close();
}

template <typename a>
void writeVector(vector<a> v, string filename)
{
    // debug
    ofstream out(filename);
    for (a x : v)
    {
        out << x << '\n';
    }
    out.close();
}

void writeProposals(vector<Proposal> proposals, string filename)
{
    // debug
    ofstream out(filename);
    out << "\n\n";

    for (auto &proposal : proposals)
    {
        Coalition &coalition = proposal.first;
        VectorXd &div = proposal.second;
        out << "coalition: ";
        for (auto agent : coalition)
        {
            out << agent << " ";
        }
        out << " | div : ";
        for (int i = 0; i < div.size(); i++)
        {
            out << div[i] << " ";
        }
        out << "======================================\n";
    }
    out.close();
}
// void OneExperiment()
// {
//     mt19937_64 generator(SEED);
//     vector<Task> tasks = {
//         //{8, 10},
//         {1, 1},
//         {4, 5.9},
//         {2, 3},
//     };
//     int numPlayers = 5;
//     int numWeights = 4;
//     //vector<int> agentWeights = {1, 2, 3};
//     vector<int> agentWeights = {1, 1, 2, 3, 4};
//     Game game(numPlayers, numWeights, tasks, generator, agentWeights);

//     //Game game(numPlayers, numWeights, tasks, generator);

//     //InformedBeliefAlgo inform(game, generator);
//     SoftmaxQ softmaxQ(game, generator);
//     // VPIAlgo VPI(game, generator);
//     //Bandit bandit(game, generator);

//     //softmaxQ.testProposal();
//     //VectorXd v(2);
//     //v << 0.4, 0.6;
//     //Proposal proposal = make_pair(set<int>{1,2}, v);
//     //map<int, int> resp = game.predictReponses(game.agents[2], proposal).second;
//     //cout << "resp: " << resp[2] << endl;

//     //VectorXd v(2);
//     //v << 0.4, 0.6;
//     //Proposal proposal = make_pair(set<int>{0,1}, v);
//     //map<int, int> resp = game.predictReponses(game.agents[0], proposal).second;
//     //cout << "resp: " << resp[0] << endl;

//     auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps);

//     // verify current wealth of players!
//     for (auto &agent : game.agents)
//     {
//         cout << "agent " << agent.name << " wealth " << agent.currentWealth << endl;
//     }

//     //auto beliefOutcomes = runAlgorithm(inform, numSteps);
//     // auto beliefOutcomes = runAlgorithm(VPI, numSteps);
//     //auto beliefOutcomes = runAlgorithm(bandit, numSteps);

//     //writeOutcomes(beliefOutcomes.second, "cpp_opt_outcomes.txt");
//     //writeBeliefs(beliefOutcomes.first, "cpp_opt_beliefs.txt", numSteps, numPlayers);

//     writeOutcomes(get<1>(beliefOutcomes), "./data/cpp_softmax_outcomes.txt");
//     //writeBeliefs(beliefOutcomes.first, "cpp_softmax_beliefs.txt", numSteps, numPlayers);

//     //writeOutcomes(beliefOutcomes.second, "cpp_VPI_outcomes.txt");
//     //writeBeliefs(beliefOutcomes.first, "cpp_VPI_beliefs.txt", numSteps, numPlayers);

//     //writeOutcomes(beliefOutcomes.second, "cpp_bandit_outcomes.txt");
//     //writeBeliefs(beliefOutcomes.first, "cpp_bandit_beliefs.txt", numSteps, numPlayers);
// }

// void OneExploitSignalExperiment(double trustLevel, string filename, double seed, vector<int> agentWeights)

// {
//     mt19937_64 generator(seed);
//     vector<Task> tasks = {
//         {1, 1},
//         {4, 5.9},
//         {2, 3},
//     };
//     int numPlayers = 5;
//     int numWeights = 4;
//     Game game(numPlayers, numWeights, tasks, generator, agentWeights);
//     SoftmaxQ softmaxQ(game, generator);
//     auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps, true, trustLevel);
//     //writeOutcomes(beliefOutcomes.second, "./data/cpp_softmax_outcomes_exploit.txt");
//     writeOutcomes(get<1>(beliefOutcomes), filename);
// }

void bulkExperimentSoftmax(int numTrials, int numPlayers, int numWeights, vector<int> &agentWeights, vector<Task> &tasks,
                           string prefix = "./data/softmax/cpp_softmax")
{
#pragma omp parallel for
    for (int i = 0; i < numTrials; i++)
    {
        mt19937_64 generator(i);
        Game game(numPlayers, numWeights, tasks, generator, agentWeights);
        SoftmaxQ softmaxQ(game, generator);
        auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps);
        writeOutcomes(get<1>(beliefOutcomes), prefix + "_outcomes_" + to_string(i) + ".txt");
        writeBeliefs(get<0>(beliefOutcomes), prefix + "_beliefs_" + to_string(i) + ".txt", numSteps, numPlayers);
        writeWealth(get<2>(beliefOutcomes), prefix + "_wealth_" + to_string(i) + ".txt");
        writeVector(game.proposerList, prefix + "_proposerList_" + to_string(i) + ".txt");
        writeVector(get<3>(beliefOutcomes), prefix + "_inversions_" + to_string(i) + ".txt");
        writeVector(game.agentWeights, prefix + "_agentWeights_" + to_string(i) + ".txt");
        writeProposals(game.proposals, prefix + "_proposals_" + to_string(i) + ".txt");
                writeVector(game.acceptances, prefix + "_acceptance_" + to_string(i) + ".txt");
        writeVector(get<4>(beliefOutcomes), prefix + "_signalFidelity_" + to_string(i) + ".txt");

    }
}

void bulkExperimentInformed(int numTrials, int numPlayers, int numWeights, vector<int> &agentWeights, vector<Task> &tasks,
                            string prefix = "./data/informed/cpp_informed")
{
#pragma omp parallel for
    for (int i = 0; i < numTrials; i++)
    {
        mt19937_64 generator(i);
        Game game(numPlayers, numWeights, tasks, generator, agentWeights);
        InformedBeliefAlgo informed(game, generator);
        auto beliefOutcomes = runAlgorithm(informed, numSteps);
        writeOutcomes(get<1>(beliefOutcomes), prefix + "_outcomes_" + to_string(i) + ".txt");
        writeBeliefs(get<0>(beliefOutcomes), prefix + "_beliefs_" + to_string(i) + ".txt", numSteps, numPlayers);
        writeWealth(get<2>(beliefOutcomes), prefix + "_wealth_" + to_string(i) + ".txt");
        writeVector(game.proposerList, prefix + "_proposerList_" + to_string(i) + ".txt");
        writeVector(get<3>(beliefOutcomes), prefix + "_inversions_" + to_string(i) + ".txt");
        writeVector(game.agentWeights, prefix + "_agentWeights_" + to_string(i) + ".txt");
        writeProposals(game.proposals, prefix + "_proposals_" + to_string(i) + ".txt");
    }
}

void bulkExperimentBeliefInjection(int numTrials, int numPlayers, int numWeights, vector<int> &agentWeights, vector<Task> &tasks,
                                   double trustLevel,
                                   string prefix = "./data/injection/cpp_injection")
{
#pragma omp parallel for
    for (int i = 0; i < numTrials; i++)
    {
        mt19937_64 generator(i);
        Game game(numPlayers, numWeights, tasks, generator, agentWeights);
        SoftmaxQ softmaxQ(game, generator);

        auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps, true, trustLevel);
        writeOutcomes(get<1>(beliefOutcomes), prefix + "_outcomes_" + to_string(i) + ".txt");
        writeBeliefs(get<0>(beliefOutcomes), prefix + "_beliefs_" + to_string(i) + ".txt", numSteps, numPlayers);
        writeWealth(get<2>(beliefOutcomes), prefix + "_wealth_" + to_string(i) + ".txt");
        writeVector(game.proposerList, prefix + "_proposerList_" + to_string(i) + ".txt");
        writeVector(get<3>(beliefOutcomes), prefix + "_inversions_" + to_string(i) + ".txt");
        writeVector(game.agentWeights, prefix + "_agentWeights_" + to_string(i) + ".txt");
        writeProposals(game.proposals, prefix + "_proposals_" + to_string(i) + ".txt");
        writeVector(game.acceptances, prefix + "_acceptance_" + to_string(i) + ".txt");
        writeVector(get<4>(beliefOutcomes), prefix + "_signalFidelity_" + to_string(i) + ".txt");
    }
}

void bulkExperimentSignalSoftmax(int numTrials, int numPlayers, int numWeights, vector<int> &agentWeights, vector<Task> &tasks,
                                 double lambda,
                                 string prefix = "./data/regularizer/cpp_regularizer")
{
#pragma omp parallel for
    for (int i = 0; i < numTrials; i++)
    {
        mt19937_64 generator(i);
        Game game(numPlayers, numWeights, tasks, generator, agentWeights);
        SignalSoftmaxQ signalSoftmaxQ(game, generator);
        signalSoftmaxQ.lambda = lambda;

        auto beliefOutcomes = runAlgorithm(signalSoftmaxQ, numSteps);
        writeOutcomes(get<1>(beliefOutcomes), prefix + "_outcomes_" + to_string(i) + ".txt");
        writeBeliefs(get<0>(beliefOutcomes), prefix + "_beliefs_" + to_string(i) + ".txt", numSteps, numPlayers);
        writeWealth(get<2>(beliefOutcomes), prefix + "_wealth_" + to_string(i) + ".txt");
        writeVector(game.proposerList, prefix + "_proposerList_" + to_string(i) + ".txt");
        writeVector(get<3>(beliefOutcomes), prefix + "_inversions_" + to_string(i) + ".txt");
        writeVector(game.agentWeights, prefix + "_agentWeights_" + to_string(i) + ".txt");
        writeProposals(game.proposals, prefix + "_proposals_" + to_string(i) + ".txt");
        writeVector(game.acceptances, prefix + "_acceptance_" + to_string(i) + ".txt");
        writeVector(get<4>(beliefOutcomes), prefix + "_signalFidelity_" + to_string(i) + ".txt");
    }
}

void bulkExperimentBandit(int numTrials, int numPlayers, int numWeights, vector<int> &agentWeights, vector<Task> &tasks,
                          string prefix = "./data/bandit/cpp_bandit")
{
#pragma omp parallel for
    for (int i = 0; i < numTrials; i++)
    {
        mt19937_64 generator(i);
        Game game(numPlayers, numWeights, tasks, generator, agentWeights);
        Bandit bandit(game, generator);

        auto beliefOutcomes = runAlgorithm(bandit, numSteps);
        writeOutcomes(get<1>(beliefOutcomes), prefix + "_outcomes_" + to_string(i) + ".txt");
        writeBeliefs(get<0>(beliefOutcomes), prefix + "_beliefs_" + to_string(i) + ".txt", numSteps, numPlayers);
        writeWealth(get<2>(beliefOutcomes), prefix + "_wealth_" + to_string(i) + ".txt");
        writeVector(game.proposerList, prefix + "_proposerList_" + to_string(i) + ".txt");
        writeVector(get<3>(beliefOutcomes), prefix + "_inversions_" + to_string(i) + ".txt");
        writeVector(game.agentWeights, prefix + "_agentWeights_" + to_string(i) + ".txt");
        writeProposals(game.proposals, prefix + "_proposals_" + to_string(i) + ".txt");
    }
}

void bulkExperimentSignalBandit(int numTrials, int numPlayers, int numWeights, vector<int> &agentWeights, vector<Task> &tasks,
                                double lambda,
                                string prefix = "./data/signalBandit/cpp_signalBandit")
{
#pragma omp parallel for
    for (int i = 0; i < numTrials; i++)
    {
        mt19937_64 generator(i);
        Game game(numPlayers, numWeights, tasks, generator, agentWeights);
        SignalBandit signalBandit(game, generator);
        signalBandit.lambda = lambda;

        auto beliefOutcomes = runAlgorithm(signalBandit, numSteps);
        writeOutcomes(get<1>(beliefOutcomes), prefix + "_outcomes_" + to_string(i) + ".txt");
        writeBeliefs(get<0>(beliefOutcomes), prefix + "_beliefs_" + to_string(i) + ".txt", numSteps, numPlayers);
        writeWealth(get<2>(beliefOutcomes), prefix + "_wealth_" + to_string(i) + ".txt");
        writeVector(game.proposerList, prefix + "_proposerList_" + to_string(i) + ".txt");
        writeVector(get<3>(beliefOutcomes), prefix + "_inversions_" + to_string(i) + ".txt");
        writeVector(game.agentWeights, prefix + "_agentWeights_" + to_string(i) + ".txt");
        writeProposals(game.proposals, prefix + "_proposals_" + to_string(i) + ".txt");
    }
}

void softmaxBulkExperiment(string prefix = "./data/softmax/cpp_softmax")
{
    vector<Task> tasks = {
        {1, 1},
        {4, 5.9},
        {2, 3},
    };
    int numPlayers = 5;
    int numWeights = 4;
    // deliberately give fewer agentWeights than required to trigger random weight initialization!
    vector<int> agentWeights = {};
    int numTrials = 30;
    bulkExperimentSoftmax(numTrials, numPlayers, numWeights, agentWeights, tasks, prefix);
}

void beliefInjectionBulkExperiment(double trustLevel,
                                   int numTrials = 1, string prefix = "./data/injection/cpp_injection")
{
    vector<Task> tasks = {
        {1, 1},
        {4, 5.9},
        {2, 3},
    };
    int numPlayers = 5;
    int numWeights = 4;
    // deliberately give fewer agentWeights than required to trigger random weight initialization!
    vector<int> agentWeights = {};
    bulkExperimentBeliefInjection(numTrials, numPlayers, numWeights, agentWeights, tasks, trustLevel, prefix);
}

void informedBeliefBulkExperiment(int numTrials = 1, string prefix = "./data/informed/cpp_informed")
{
    vector<Task> tasks = {
        {1, 1},
        {4, 5.9},
        {2, 3},
    };
    int numPlayers = 5;
    int numWeights = 4;
    // deliberately give fewer agentWeights than required to trigger random weight initialization!
    vector<int> agentWeights = {};
    bulkExperimentInformed(numTrials, numPlayers, numWeights, agentWeights, tasks, prefix);
}

void signalSoftmaxBulkExperiment(double lambda, int numTrials = 1, string prefix = "./data/regularizer/cpp_regularizer")
{
    vector<Task> tasks = {
        {1, 1},
        {4, 5.9},
        {2, 3},
    };
    int numPlayers = 5;
    int numWeights = 4;
    // deliberately give fewer agentWeights than required to trigger random weight initialization!
    vector<int> agentWeights = {};
    bulkExperimentSignalSoftmax(numTrials, numPlayers, numWeights, agentWeights, tasks, lambda, prefix);
}

void banditBulkExperiment(int numTrials = 1, string prefix = "./data/bandit/cpp_bandit")
{
    vector<Task> tasks = {
        {1, 1},
        {4, 5.9},
        {2, 3},
    };
    int numPlayers = 5;
    int numWeights = 4;
    // deliberately give fewer agentWeights than required to trigger random weight initialization!
    vector<int> agentWeights = {};
    bulkExperimentBandit(numTrials, numPlayers, numWeights, agentWeights, tasks, prefix);
}

void signalBanditBulkExperiment(double lambda, int numTrials = 1, string prefix = "./data/signalBandit/cpp_signalBandit")
{
    vector<Task> tasks = {
        {1, 1},
        {4, 5.9},
        {2, 3},
    };
    int numPlayers = 5;
    int numWeights = 4;
    // deliberately give fewer agentWeights than required to trigger random weight initialization!
    vector<int> agentWeights = {};
    bulkExperimentSignalBandit(numTrials, numPlayers, numWeights, agentWeights, tasks, lambda, prefix);
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

void runRegularizer()
{

    auto start = high_resolution_clock::now();

    vector<Task> tasks = {
        {1, 1},
        {4, 5.9},
        {2, 3},
    };
    vector<double> lambdas = {
        0.0,
        0.1,
        0.5,
        1.0,
        10.0,
        20.0};

    int numPlayers = 5;
    int numWeights = 4;

    vector<int> agentWeights = {1, 1, 2, 3, 3};
    int numTrials = 10;

#pragma omp parallel for num_threads(8) collapse(2)
    for (int i = 0; i < lambdas.size(); i++)
    {
        {
            for (int j = 0; j < numTrials; j++)
            {

                mt19937_64 generator(j);
                Game game(numPlayers, numWeights, tasks, generator, agentWeights);
                SignalSoftmaxQ softmaxQ(game, generator);
                softmaxQ.lambda = lambdas[i];
                auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps);
                writeOutcomes(get<1>(beliefOutcomes), "./data/cpp_softmax_outcomes_regularizer_" + to_string(i) + "_" + to_string(j) + ".txt");
            }
        }
    }

    auto stop = high_resolution_clock::now();

    cout << "took " << duration_cast<seconds>(stop - start).count() << " seconds " << endl;
}

// void OneSmallSoftmaxExperiment()
// {
//     mt19937_64 generator(SEED);
//     vector<Task> tasks = {
//         {1, 1},
//         {2, 3},
//         {4, 5.9},
//     };
//     int numPlayers = 3;
//     int numWeights = 4;
//     vector<int> agentWeights = {1, 2, 3};
//     Game game(numPlayers, numWeights, tasks, generator, agentWeights);
//     SoftmaxQ softmaxQ(game, generator);
//     auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps);
//     writeOutcomes(get<1>(beliefOutcomes), "./data/cpp_softmax_outcomes_small.txt");
//     writeBeliefs(get<0>(beliefOutcomes), "./data/cpp_softmax_beliefs_small.txt", numSteps, numPlayers);
// }

// very big trust signal!
// void OneSmallExploitSoftmaxExperiment()
// {
//     mt19937_64 generator(SEED);
//     vector<Task> tasks = {
//         {1, 1},
//         {2, 3},
//         {4, 5.9},
//     };
//     int numPlayers = 3;
//     int numWeights = 4;
//     // vector<int> agentWeights = {1,4,1};
//     vector<int> agentWeights = {};
//     Game game(numPlayers, numWeights, tasks, generator, agentWeights);
//     SoftmaxQ softmaxQ(game, generator);
//     double trustLevel = 0.1;
//     numSteps = 20;
//     auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps, true, trustLevel);
//     writeOutcomes(get<1>(beliefOutcomes), "./caseStudyData/cpp_softmax_exploit_outcomes_small.txt");
//     writeBeliefs(get<0>(beliefOutcomes), "./caseStudyData/cpp_softmax_exploit_beliefs_small.txt", numSteps, numPlayers);
//     writeVector(game.proposerList, "./caseStudyData/cpp_softmax_exploit_proposer_small.txt");
//     writeWealth(get<2>(beliefOutcomes), "./caseStudyData/cpp_softmax_exploit_wealth_small.txt");
//     writeVector(game.agentWeights, "./caseStudyData/cpp_softmax_exploit_agentWeights_small.txt");
// }

// very big trust signal!
void OneSmallInformedBeliefExperiment()
{
    mt19937_64 generator(SEED);
    vector<Task> tasks = {
        {1, 1},
        {2, 3},
        {4, 5.9},
    };
    int numPlayers = 3;
    int numWeights = 4;
    vector<int> agentWeights = {1, 2, 3};
    Game game(numPlayers, numWeights, tasks, generator, agentWeights);
    InformedBeliefAlgo inform(game, generator);
    numSteps = 20;
    auto beliefOutcomes = runAlgorithm(inform, numSteps);

    writeOutcomes(get<1>(beliefOutcomes), "./caseStudyData/cpp_inform_outcomes_small.txt");
    writeBeliefs(get<0>(beliefOutcomes), "./caseStudyData/cpp_inform_beliefs_small.txt", numSteps, numPlayers);
    writeVector(game.proposerList, "./caseStudyData/cpp_inform_proposer_small.txt");
}

// void OneSmallSoftmaxLimitingBeliefExperiment()
// {
//     mt19937_64 generator(SEED);
//     vector<Task> tasks = {
//         {1, 1},
//         {2, 3},
//         {4, 5.9},
//     };
//     int numPlayers = 3;
//     int numWeights = 4;
//     //vector<int> agentWeights = {1,4,1};
//     vector<int> agentWeights = {};
//     Game game(numPlayers, numWeights, tasks, generator, agentWeights);

//     // ==========================================
//     // manually setting up the limiting distribution

//     // game.agents[0].belief(1, 2) = 0.0;
//     // game.agents[0].belief(1, 3) = 0.0;
//     // game.agents[0].belief(2, 0) = 0.0;
//     // game.agents[0].belief(2, 1) = 0.0;

//     // game.agents[1].belief(0, 1) = 0.0;
//     // game.agents[1].belief(0, 2) = 0.0;
//     // game.agents[1].belief(0, 3) = 0.0;
//     // game.agents[1].belief(2, 0) = 0.0;
//     // ==========================================

//     // normalize beliefs
//     for (auto &agent : game.agents)
//     {
//         for (int i = 0; i < numPlayers; i++)
//         {
//             agent.belief.row(i) /= agent.belief.row(i).sum();
//         }
//     }

//     SoftmaxQ softmaxQ(game, generator);
//     numSteps = 20;
//     auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps);

//     writeOutcomes(get<1>(beliefOutcomes), "./caseStudyData/cpp_softmax_outcomes_small.txt");
//     writeVector(game.proposerList, "./caseStudyData/cpp_softmax_proposer_small.txt");
//     writeBeliefs(get<0>(beliefOutcomes), "./caseStudyData/cpp_softmax_beliefs_small.txt", numSteps, numPlayers);
//     writeWealth(get<2>(beliefOutcomes), "./caseStudyData/cpp_softmax_wealth_small.txt");
//     writeVector(get<3>(beliefOutcomes), "./caseStudyData/cpp_softmax_inversions_small.txt");
//     writeVector(game.agentWeights, "./caseStudyData/cpp_softmax_agentWeights_small.txt");
// }

void figureSamplePaperv3()
{

    // informedBeliefBulkExperiment(30, "./data_5agents/informed/cpp_informed");
    // softmaxBulkExperiment("./data_5agents/softmax/cpp_softmax");

    // injection
    // beliefInjectionBulkExperiment(0.1, 30, "./data_5agents/0.1_injection/cpp_injection");
    // beliefInjectionBulkExperiment(0.5, 30, "./data_5agents/0.5_injection/cpp_injection");
    // beliefInjectionBulkExperiment(1.0, 30, "./data_5agents/1.0_injection/cpp_injection");
    beliefInjectionBulkExperiment(5.0, 30, "./data_5agents/5.0_injection/cpp_injection");
    // beliefInjectionBulkExperiment(10.0, 30, "./data_5agents/10.0_injection/cpp_injection");
    // beliefInjectionBulkExperiment(20.0, 30, "./data_5agents/20.0_injection/cpp_injection");

    // regularized softmax
    // signalSoftmaxBulkExperiment(0.1, 30, "./data_5agents/0.1_regularizer/cpp_regularizer");
    // signalSoftmaxBulkExperiment(0.5, 30, "./data_5agents/0.5_regularizer/cpp_regularizer");
    // signalSoftmaxBulkExperiment(1.0, 30, "./data_5agents/1.0_regularizer/cpp_regularizer");
    // signalSoftmaxBulkExperiment(5.0, 30, "./data_5agents/5.0_regularizer/cpp_regularizer");
    // signalSoftmaxBulkExperiment(10.0, 30, "./data_5agents/10.0_regularizer/cpp_regularizer");
    // signalSoftmaxBulkExperiment(20.0, 30, "./data_5agents/20.0_regularizer/cpp_regularizer");

    // banditBulkExperiment(30, "./data_5agents/bandit/cpp_bandit");

    // regularized bandit
    // signalBanditBulkExperiment(0.1, 30, "./data_5agents/0.1_regularizedBandit/cpp_regularizedBandit");
    // signalBanditBulkExperiment(0.5, 30, "./data_5agents/0.5_regularizedBandit/cpp_regularizedBandit");
    // signalBanditBulkExperiment(1.0, 30, "./data_5agents/1.0_regularizedBandit/cpp_regularizedBandit");
    // signalBanditBulkExperiment(5.0, 30, "./data_5agents/5.0_regularizedBandit/cpp_regularizedBandit");
    // signalBanditBulkExperiment(10.0, 30, "./data_5agents/10.0_regularizedBandit/cpp_regularizedBandit");
    // signalBanditBulkExperiment(20.0, 30, "./data_5agents/20.0_regularizedBandit/cpp_regularizedBandit");
}


// might need to redo this ...
void caseStudyWhySignalIsGood()
{
    vector<Task> tasks = {
        {1, 1},
        {4, 5.9},
        {2, 3},
    };
    int numPlayers = 2;
    int numWeights = 4;

    vector<int> weightRange = {1, 2, 3, 4};

    double signalSum = 0;
    double softmaxSum = 0;
    for (auto &agentWeights : CartesianSelfProduct(weightRange, 2))
    {

        cout << "weight\n";
        printVec(agentWeights);

        mt19937_64 generator(SEED);
        Game game(numPlayers, numWeights, tasks, generator, agentWeights);

        mt19937_64 generatorSignal(SEED);
        Game gameSignal(numPlayers, numWeights, tasks, generatorSignal, agentWeights);

        SoftmaxQ softmax(game, generator);
        auto beliefOutcomesSoft = runAlgorithm(softmax, numSteps);

        SignalSoftmaxQ softmaxSig(gameSignal, generatorSignal);
        auto beliefOutcomesSignal = runAlgorithm(softmaxSig, numSteps);

        cout << "softmax cumulative payoff " << calculateCumulativePayoff(get<1>(beliefOutcomesSoft)) << " | ";
        cout << "signal cumulative payoff " << calculateCumulativePayoff(get<1>(beliefOutcomesSignal)) << endl;
        softmaxSum += calculateCumulativePayoff(get<1>(beliefOutcomesSoft));
        signalSum += calculateCumulativePayoff(get<1>(beliefOutcomesSignal));
    }

    cout << "In total, softmax : " << softmaxSum << " | signal: " << signalSum << endl;
}

void OneCaseStudyWhySignalIsGood()
{
    // theory: if the limiting belief is correct, then this justice stuff just increases the acceptance rate,
    // which generally leads to more optimal structure. The environment is by design this way since that's kinda
    // the point of cooperative games: to cooperate!
    vector<Task> tasks = {
        {1, 1},
        {4, 5.9},
        {2, 3},
    };
    int numPlayers = 2;
    int numWeights = 4;
    vector<int> agentWeights = {3,4};

    mt19937_64 generator(SEED);
    Game game(numPlayers, numWeights, tasks, generator, agentWeights);

    mt19937_64 generatorSignal(SEED);
    Game gameSignal(numPlayers, numWeights, tasks, generatorSignal, agentWeights);

    SoftmaxQ softmax(game, generator);
    auto beliefOutcomesSoft = runAlgorithm(softmax, numSteps);

    SignalSoftmaxQ softmaxSig(gameSignal, generatorSignal);
    auto beliefOutcomesSignal = runAlgorithm(softmaxSig, numSteps);

    cout << "softmax cumulative payoff " << calculateCumulativePayoff(get<1>(beliefOutcomesSoft)) << " | ";
    cout << "signal cumulative payoff " << calculateCumulativePayoff(get<1>(beliefOutcomesSignal)) << endl;

    string prefix = "./caseStudyWhySignalIsGood/softmax/soft_subadditive";

    writeOutcomes(get<1>(beliefOutcomesSoft), prefix + "_outcomes_" + ".txt");
    writeBeliefs(get<0>(beliefOutcomesSoft), prefix + "_beliefs_" + ".txt", numSteps, numPlayers);
    writeWealth(get<2>(beliefOutcomesSoft), prefix + "_wealth_" + ".txt");
    writeVector(game.proposerList, prefix + "_proposerList_" + ".txt");
    writeProposals(game.proposals, prefix + "_proposals_" + ".txt");
    writeVector(game.acceptances, prefix + "_acceptance_" + ".txt");
    writeVector(get<4>(beliefOutcomesSoft), prefix + "_signalFidelity_" + ".txt");

    prefix = "./caseStudyWhySignalIsGood/signal/signal_subadditive";
    writeOutcomes(get<1>(beliefOutcomesSignal), prefix + "_outcomes_" + ".txt");
    writeBeliefs(get<0>(beliefOutcomesSignal), prefix + "_beliefs_" + ".txt", numSteps, numPlayers);
    writeWealth(get<2>(beliefOutcomesSignal), prefix + "_wealth_" + ".txt");
    writeVector(gameSignal.proposerList, prefix + "_proposerList_" + ".txt");
    writeProposals(gameSignal.proposals, prefix + "_proposals_" + ".txt");
    writeVector(gameSignal.acceptances, prefix + "_acceptance_" + ".txt");
    writeVector(get<4>(beliefOutcomesSignal), prefix + "_signalFidelity_" + ".txt");
}

void regularizedCounterTheoreticalExample(){

vector<Task> tasks = {
        {13, 1.5},
        {35, 2}
    };
    int numPlayers = 3;
    int numWeights = 30;
    vector<int> agentWeights = {3,10,11};


    mt19937_64 generator(SEED);
    Game game(numPlayers, numWeights, tasks, generator, agentWeights);

    mt19937_64 generatorSignal(SEED);
    Game gameSignal(numPlayers, numWeights, tasks, generatorSignal, agentWeights);

    mt19937_64 generatorInformed(SEED);
    Game gameInformed(numPlayers, numWeights, tasks, generatorInformed, agentWeights);

    mt19937_64 generatorInjection(SEED);
    Game gameInjection(numPlayers, numWeights, tasks, generatorInjection, agentWeights);

    // modify the initial currentWealth!
    game.agents[0].currentWealth = 3;
    game.agents[1].currentWealth = 10;
    game.agents[2].currentWealth = 22;


    gameSignal.agents[0].currentWealth = 3;
    gameSignal.agents[1].currentWealth = 10;
    gameSignal.agents[2].currentWealth = 22;

    gameInformed.agents[0].currentWealth = 3;
    gameInformed.agents[1].currentWealth = 10;
    gameInformed.agents[2].currentWealth = 22;

    gameInjection.agents[0].currentWealth = 3;
    gameInjection.agents[1].currentWealth = 10;
    gameInjection.agents[2].currentWealth = 22;

    SoftmaxQ softmax(game, generator);
    SignalSoftmaxQ softmaxSig(gameSignal, generatorSignal);
    InformedBeliefAlgo inform(gameInformed, generatorInformed);
    SoftmaxQ softmaxQInjection(gameInjection, generatorInjection);


    double trustLevel = 5.0;
    auto beliefOutcomesSoft = runAlgorithm(softmax, numSteps);
    auto beliefOutcomesSignal = runAlgorithm(softmaxSig, numSteps);
    auto beliefOutcomesInformed = runAlgorithm(inform, numSteps);
    auto beliefOutcomesInjection = runAlgorithm(softmaxQInjection, numSteps, true, trustLevel);

    cout << "softmax cumulative payoff " << calculateCumulativePayoff(get<1>(beliefOutcomesSoft)) << " | ";
    cout << "signal cumulative payoff " << calculateCumulativePayoff(get<1>(beliefOutcomesSignal)) << " | ";
    cout << "informed cumulative payoff " << calculateCumulativePayoff(get<1>(beliefOutcomesInformed)) << " | ";
    cout << "injection cumulative payoff " << calculateCumulativePayoff(get<1>(beliefOutcomesInjection)) << endl;

}
// int main()
// {
//     auto start = high_resolution_clock::now();
//     //OneSmallInformedBeliefExperiment();
//     // OneSmallSoftmaxLimitingBeliefExperiment();
//     // OneSmallExploitSoftmaxExperiment();
//     // softmaxBulkExperiment();
//     // OneSmallExploitSoftmaxExperiment();
//     // OneSmallSoftmaxLimitingBeliefExperiment();

//     // beliefInjectionBulkExperiment(0.1, 30, "./data_3agents/injection");
//     // figureSamplePaperv3();

//     // // fixing nan
//     // vector<Task> tasks = {
//     //     {1, 1},
//     //     {4, 5.9},
//     //     {2, 3},
//     // };
//     // vector<int> agentWeights = {};
//     // int numPlayers = 3;
//     // int numWeights = 4;

//     // mt19937_64 generator(13);
//     // Game game(numPlayers, numWeights, tasks, generator, agentWeights);
//     // SoftmaxQ softmaxQ(game, generator);
//     // auto beliefOutcomes = runAlgorithm(softmaxQ, numSteps, true, 20);



//     // OneCaseStudyWhySignalIsGood();
//     // caseStudyWhySignalIsGood();

//     regularizedCounterTheoreticalExample();

//     // figureSamplePaperv3();



//     auto stop = high_resolution_clock::now();
//     cout << "took " << duration_cast<seconds>(stop - start).count() << " seconds " << endl;
// }
