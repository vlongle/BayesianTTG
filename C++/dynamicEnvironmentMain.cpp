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


using namespace std;



vector<vector<int>> assignPlayersToRooms(int n, int m, mt19937_64 &generator, uniform_int_distribution<int> &distribution)
{
    vector<vector<int>> playersToRooms; // playersToRoom[k] : list of players in room k
    for (int i = 0; i < m; i++)
    {
        playersToRooms.push_back({});
    }
    for (int i = 0; i < n; i++)
    {
        int roomNumber = distribution(generator);
        playersToRooms[roomNumber].push_back(i);
    }
    return playersToRooms;
}

void writeRoomAssignment(vector<vector<vector<int>>> assignment, string filename,
string sep= "|"){
     ofstream out(filename);
    // assignment[t] per line 
    for (auto& playersToRooms: assignment){
        for (int room=0; room < playersToRooms.size(); room++){
            for (int player : playersToRooms[room]){
                out << player << " ";
            }
            out << sep;
        }
        out << endl;
    }
    out.close();
}
void runDynamicEnvAlgorithm(vector<Agent> &agents, Environments &envs, int numSteps,
                  int seed)
{

    int m = envs.size(); // number of environments
    int n = agents.size();

    mt19937_64 generator(seed);
    uniform_int_distribution<int> distribution(0, m - 1);

    vector<vector<vector<int>>> assignment;
    for (int t = 0; t < numSteps; t++)
    {
        // assign players to environments now!
        vector<vector<int>> playersToRooms = assignPlayersToRooms(n, m, generator, distribution);
        assignment.push_back(playersToRooms);
    }

    writeRoomAssignment(assignment, "./assignment.txt");
}

int main()
{
    vector<Task> env1 = {
        {0, 0},
    };
    vector<Task> env2 = {
        {0, 1},
    };

    int seed = 0;
    int numSteps = 100;
    //Environments envs = {env1, env2};
    Environments envs = {env1};
    vector<Agent> agents = {Agent(0, 0), Agent(0, 1)};

    runDynamicEnvAlgorithm(agents, envs, numSteps, seed);
}