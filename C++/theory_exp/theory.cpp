#include <iostream>
#include <set>
#include <vector>
#include "Eigen/Dense"
#include <fstream>
#include <algorithm>
#include <random>
#include <iterator>
#include "Utils.hpp"
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace std::chrono;

// vector of tasks ordered in increasing reward and threshold
typedef vector<pair<double, double>> Environment;
typedef vector<int> Outcome;
typedef vector<Outcome> Path;

// roomOrientedOutcome[roomNumber] = vector of players in that room
typedef vector<vector<int>> roomOrientedOutcome;

template <typename T>
void printVec(vector<T> &v)
{
    for (auto &x : v)
    {
        cout << x << " ";
    }
}

// // one environment
// // NOTE: money(subset) syntax only works for the new 3.4.0 Eigen (dev not stable released) library
void simulation(int T, mt19937 &generator, int N, string filename)
{
    // i = -2, j = -1
    vector<int> players(N);
    iota(players.begin(), players.end(), 0);
    VectorXd weights = VectorXd::LinSpaced(N, 0, N - 1); // LinSpaced(size, low, high)
                                                         // our weight = player name exactly!

    VectorXd money = VectorXd::Constant(N, 1);

    MatrixXd result(T, N);

    int k = N / 2;

    ofstream out(filename);
    for (int t = 0; t < T; t++)
    {
        vector<int> subset;

        sample(players.begin(), players.end(), back_inserter(subset), k, generator);
        money(subset) += (money(subset) / money(subset).sum()) * weights(subset).sum();
        // // singleton as well!
        money += weights;
        money(subset) -= weights(subset); // kinda ugly but we don't want to reward this subset twice!

        //     out << result;
        //     out.close();
        // }
    }
}

void runStaticEnv()
{
    int T = 1000;
    int N = 6;
    int numTrials = 1000;

#pragma omp parallel for
    for (int trial = 0; trial < numTrials; trial++)
    {
        mt19937 generator(trial);
        // simulation(T, generator, N, "theory_" + to_string(trial) + ".txt");
    }
}

VectorXd generatePlayers(int maxPlayers, double maxPlayerWeight, mt19937 &generator)
{
    uniform_int_distribution<int> distribution(1, maxPlayers);
    int numPlayers = distribution(generator);

    VectorXd weights(numPlayers);
    uniform_real_distribution<double> u(0.0, maxPlayerWeight);

    for (int i = 0; i < numPlayers; i++)
    {
        weights(i) = u(generator);
    }
    return weights;
}

Environment generateARandomEnv(int maxNumTasks, double maxReward, double maxThreshold, mt19937 &generator)
{
    uniform_int_distribution<int> distribution(1, maxNumTasks);
    int numTasks = distribution(generator);

    Environment env;
    double lowerReward = 0, lowerThreshold = 0;
    for (int i = 0; i < numTasks; i++)
    {
        double reward = uniform_real_distribution<double>(lowerReward, maxReward)(generator);
        double threshold = uniform_real_distribution<double>(lowerThreshold, maxThreshold)(generator);
        env.push_back(make_pair(threshold, reward));
        lowerReward = reward;
        lowerThreshold = threshold;
    }
    return env;
}

vector<Environment> generateDynamicEnvs(int maxRooms, int maxNumTasks, double maxReward, double maxThreshold, mt19937 &generator)
{
    uniform_int_distribution<int> distribution(1, maxRooms);
    int numRooms = distribution(generator);

    vector<Environment> envs;
    for (int i = 0; i < numRooms; i++)
    {
        envs.push_back(generateARandomEnv(maxNumTasks, maxReward, maxThreshold, generator));
    }

    return envs;
}

vector<Path> generateAllPaths(int numRooms, int numPlayers, int numTimeSteps)
{

    vector<int> roomNames(numRooms);
    iota(roomNames.begin(), roomNames.end(), 0);
    vector<Outcome> outcomes = CartesianSelfProduct(roomNames, numPlayers);

    vector<Path> paths = CartesianSelfProduct(outcomes, numTimeSteps);
    return paths;
}

roomOrientedOutcome transformOutcomeToRoomOriented(Outcome &outcome, int numRoom)
{
    roomOrientedOutcome transformedOutcome(numRoom);
    for (auto [playerName, roomName] : enumerate(outcome))
    {
        transformedOutcome[roomName].push_back(playerName);
    }
    return transformedOutcome;
}

VectorXd divRule(VectorXd alpha)
{
    return alpha.cwiseProduct(VectorXd::Constant(alpha.size(), 1.0 / alpha.sum()));
}

double evaluateWeightInEnvironment(Environment &env, double weight)
{
    double bestReward = 0;

    for (auto &task : env)
    {
        double threshold = task.first;
        double reward = task.second;
        if (weight >= threshold)
        {
            bestReward = max(reward, bestReward);
        }
    }
    return bestReward;
}

// TODO: check for every time as well!!
MatrixXd evaluateAPath(Path &path, VectorXd &playerWeights, vector<Environment> &envs)
{
    int T = path.size();
    MatrixXd M(T + 1, playerWeights.size());
    M.row(0) = VectorXd::Constant(playerWeights.size(), 1);
    //cout << "eval path " << endl;
    for (int t = 1; t < T + 1; t++)
    {
        auto assignment = transformOutcomeToRoomOriented(path[t-1], envs.size());
        for (auto [roomNum, room] : enumerate(assignment))
        {
            // compute v(W) and division rule!
            double vW = evaluateWeightInEnvironment(envs[roomNum], playerWeights(room).sum());
            VectorXd div = divRule(M.row(t - 1)(room));
            M.row(t)(room) = M.row(t-1)(room) + div.cwiseProduct(VectorXd::Constant(room.size(), vW)).transpose();
        }
    }
    //cout << "done eval path " << endl;
    return M;
}

void bruteForceDynamicEnvSetting(VectorXd &playerWeights, vector<Environment> &envs, mt19937 &generator, int maxNumTimeSteps, int seed)
{

    // enumerate all possible paths and then calculate the expected loss and such ...
    vector<Path> allPaths = generateAllPaths(envs.size(), playerWeights.size(), uniform_int_distribution<int>(1, maxNumTimeSteps)(generator));
    //cout << "Number of paths: " << allPaths.size() << endl;

    int n = playerWeights.size();
    int m = envs.size();
    map<tuple<int, int, int>, double> desiredProb; // desiredProb(i, j, t) = Pr that Mi >= Mj given wi > wj at time t
    for (auto &path : allPaths)
    {
        MatrixXd M = evaluateAPath(path, playerWeights, envs);
        //cout << "M : " << M << endl << endl;
        //cout << M.transpose() << endl;

        // saving prob to check for counter-example
        for (int t = 1; t < path.size()+1; t++)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (playerWeights(i) > playerWeights(j) && M(t, i) >= M(t, j))
                    {
                        // https://stackoverflow.com/questions/19354997/whats-the-easiest-way-to-get-pythons-defaultdict-behavior-in-c
                        auto emplace_pair = desiredProb.emplace(make_tuple(i, j, t), 0);

                        // ([m]^n)^T number of possible paths at time t               
                        emplace_pair.first->second += 1.0/allPaths.size();
                    }
                }
            }
        }
    }

    for (auto elem : desiredProb)
    {
        if (elem.second <= 0.5)
        {
            cout << "FOUND COUNTER-EXAMPLE at time " << get<2>(elem.first) << " seed " << seed << " with i, j " << get<0>(elem.first) << " " << get<1>(elem.first) << " Prob: " << elem.second << endl;
        }
    }
    //return desiredProb;
}
// trying to find a counter-example
void runDynamicEnv()
{

    int maxPlayers = 5;
    double maxPlayerWeight = 5.0; // playerWeight in [0, maxWeight]

    // each environment is a TTG
    int maxRooms = 5;
    int maxNumTasks = 5;
    double maxReward = 10.0;
    double maxThreshold = maxPlayers * maxPlayerWeight;

    int numTrials = 500;
    int maxNumTimeSteps = 4;

#pragma omp parallel for
    for (int i = 0; i < numTrials; i++)
    {
        mt19937 generator(i);
        VectorXd playerWeights = generatePlayers(maxPlayers, maxPlayerWeight, generator);
        vector<Environment> envs = generateDynamicEnvs(maxRooms, maxNumTasks, maxReward, maxThreshold, generator);

        // manual
        //VectorXd playerWeights(3);
        //playerWeights << 1,2,3;
        //Environment env = {make_pair(1,1), make_pair(2, 3)};
        //vector<Environment> envs = {env, env};

        bruteForceDynamicEnvSetting(playerWeights, envs, generator, maxNumTimeSteps, i);
    }
}
int main()
{

    // vector<vector<int>> outcomes = CartesianSelfProduct({1,2,3}, 2);

    // for (auto & assignment : outcomes){
    //     printVec(assignment);
    //     cout << endl;
    // }

    // int numRooms = 3;
    // int numPlayers = 2;
    // int numTimeSteps = 2;
    // auto paths = generateAllPaths(numRooms, numPlayers, numTimeSteps);
    // cout << "number of paths: " << paths.size() << endl;


       auto start = high_resolution_clock::now();
    runDynamicEnv();



     auto stop = high_resolution_clock::now();
     cout << "took " << duration_cast<seconds>(stop - start).count() << " seconds " << endl;
}
