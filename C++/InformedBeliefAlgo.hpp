#pragma once
#include "Game.hpp"
#include "Algo.hpp"
class InformedBeliefAlgo : public Algo
{
public:
    InformedBeliefAlgo(Game &game, mt19937_64 &generator) : Algo(game, generator),
                                                            distribution(0, game.agents.size() - 1)
    {
        informBelief();
    };
    uniform_int_distribution<int> distribution;
    pair<int, Proposal> proposalOutcome();
    void informBelief();
    // second return is a vector of non-singleton coalitions. The number of such coalitions
    // is either 0 or 1
    pair<CoalitionStructure, vector<Coalition>> formationProcess();
    void updateBelief(vector<Coalition> &nonSingletonCoals);
     //virtual ~InformedBeliefAlgo(){};
};