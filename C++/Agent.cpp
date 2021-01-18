#include "Agent.hpp"
#include "Utils.hpp"
#include <iostream>
using namespace std;

Agent::Agent(int name, int weight)
{
    this->name = name;
    this->weight = weight;
}

void Agent::initializeBelief(int numberOfAgents, int numberOfWeights, int minWeight)
{
    // initially, uniform belief
    belief = MatrixXd::Constant(numberOfAgents, numberOfWeights, 1.0 / numberOfWeights);
    // agent is always aware of his own type!
    belief.row(name) *= 0;                // set row to 0
    belief(name, weight - minWeight) = 1; // turn on the correct weight
}

void Agent::initializeProposalSpace(int numPlayers, map<int, divisionRule> &divisionRules)
{
    //cout << "initializing proposal Space for agent " << name << endl;

    vector<int> otherAgents;
    for (int i = 0; i < numPlayers; i++)
    {
        if (i != name)
        {
            otherAgents.push_back(i);
        }
    }

    vector<set<int>> allPossibleCoalitions = powerSet(otherAgents);
    for (auto &coalition : allPossibleCoalitions)
    {
        coalition.insert(name);
        divisionRule &div = divisionRules[coalition.size()];
        // iterate through rows of divisionRule
        for (int i = 0; i < div.rows(); i++)
        {
            proposalSpace.push_back(Proposal(coalition, div.row(i)));
        }
    }
}

void Agent::printProposalSpace()
{
        for (auto &proposal : proposalSpace)
        {
            cout << ">> coalition " << endl;
            printSet(proposal.first);
            cout << "\n >> divisionRule" << endl;
            cout << proposal.second << endl;
        }
        cout << "=======================>" << endl;
}
