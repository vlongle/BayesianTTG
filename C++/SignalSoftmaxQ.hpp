#pragma once
#include "SoftmaxQ.hpp"

// override proposalOutcome to add a regularizer term based on the signal
// We can incorporate signal into this in several ways ...
class SignalSoftmaxQ : public StateEliminationAlgo
{
public:
    SignalSoftmaxQ(Game &game, mt19937_64 &generator) : StateEliminationAlgo(game, generator),
                                                        distribution(0, game.agents.size() - 1),
                                                        u(0.0, 1.0){};

    double lambda = 0.1; // justice/ regularizer weight
    uniform_int_distribution<int> distribution;
    uniform_real_distribution<double> u;
    pair<int, Proposal> proposalOutcome();
    VectorXd calculateRegularizer(Agent &proposer);
    double countInversionOfProposal(Proposal &proposal);
    // second return is a vector of non-singleton coalitions. The number of such coalitions
    // is either 0 or 1
    pair<CoalitionStructure, vector<Coalition>> formationProcess();
};