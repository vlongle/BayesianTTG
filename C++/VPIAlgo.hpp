#pragma once
#include "StateEliminationAlgo.hpp"
#include "Game.hpp"

class VPIAlgo: public StateEliminationAlgo
{
public:
    VPIAlgo(Game &game, mt19937_64 &generator) : StateEliminationAlgo(game, generator),
                                                            distribution(0, game.agents.size() - 1),
                                                            u(0.0, 1.0)
    {
    };
    uniform_int_distribution<int> distribution;
    uniform_real_distribution<double> u;
    pair<int, Proposal> proposalOutcome();
    // second return is a vector of non-singleton coalitions. The number of such coalitions
    // is either 0 or 1
    pair<CoalitionStructure, vector<Coalition>> formationProcess();

    VectorXd calculateVPIs(Agent& proposer, set<int> bestProposals, double bestValue,
    double secondBestValue);
    void populateAgentValuesAndQVs(Agent &agent);
};