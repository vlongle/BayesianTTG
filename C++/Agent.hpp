#pragma once
#include <Eigen/Dense>
#include <vector>
#include <set> 
#include <map> 
#include "Utils.hpp"
using namespace Eigen;
using namespace std;


class Agent
{
public:
    Agent(int name, int weight);
    int weight;
    int name;
    MatrixXd belief;
    vector<Proposal> proposalSpace;

    void initializeBelief(int numberOfAgents, int numberOfWeights, int minWeight = 1);
    void initializeProposalSpace(int numPlayers, map<int, divisionRule> &divisionRules);
    void printProposalSpace();
    bool beliefChanged = true; // updateBelief should change this!
    vector<Proposal> bestProposals = {};
    VectorXd proposalValues;
    VectorXd QVs;
    double currentWealth=0;
};