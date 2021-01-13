#pragma once
#include "Game.hpp"
#include "Algo.hpp"
class Bandit : public Algo
{
public:
    // NOTICE: need to set gamma
    Bandit(Game &game, mt19937_64 &generator) : Algo(game,generator),
                                                distribution(0, game.agents.size() - 1),
                                                u(0.0, 1.0)
    {
        // all agents should have the same proposal space size (reasonable assumption for now due
        //to symmetry in generating proposal space)
        K = game.agents[0].proposalSpace.size();
        // setting up bandit weights
        for (Agent &agent : game.agents)
        {
            agent.proposalValues = VectorXd::Ones(K);
        }
    };
    uniform_int_distribution<int> distribution;
    pair<int, Proposal> proposalOutcome();
    // second return is a vector of non-singleton coalitions. The number of such coalitions
    // is either 0 or 1
    pair<CoalitionStructure, vector<Coalition>> formationProcess();
    // update the bandit weights for all coalitions formed
    void updateBelief(vector<Coalition> &coalitions);
    double gamma = 0.1;
    int K;
    uniform_real_distribution<double> u;
    void updateProposerBelief(Agent &proposer, Proposal &proposal, double coalitionReward);
    void updateResponderBelief(Agent &responder, Proposal& proposal, int response, double coalitionReward);
    
    void exploitSignal(double trustLevel){
        // pass
    }

};