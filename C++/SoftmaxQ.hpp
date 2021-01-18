#pragma once
#include "StateEliminationAlgo.hpp"
#include "Game.hpp"

class SoftmaxQ: public StateEliminationAlgo
{
public:
    SoftmaxQ(Game &game, mt19937_64 &generator, int seed) : StateEliminationAlgo(game, generator),
                                                            distribution(0, game.agents.size() - 1),
                                                            u(0.0, 1.0), seed(seed)
    {
        if (seed == 0){
            cout << "proposal weight " << endl;
            for (auto & agent : game.agents){
                cout << "agent.name " << agent.name << " weight " << agent.weight << endl;
            }
        
        }
    };
    int seed;
    uniform_int_distribution<int> distribution;
    uniform_real_distribution<double> u;
    pair<int, Proposal> proposalOutcome();
    // second return is a vector of non-singleton coalitions. The number of such coalitions
    // is either 0 or 1
    pair<CoalitionStructure, vector<Coalition>> formationProcess();
    void testProposal();
    pair<int, int> proposalOutcomeTest();
};