#pragma once
#include "Algo.hpp"
#include "Game.hpp"
#include <math.h> /* exp , pow */

class StateEliminationAlgo : public Algo
{
public:
    StateEliminationAlgo(Game &game, mt19937_64 &generator) : Algo(game, generator){};

    // update belief and turn beliefChanged = true
    // belief updating rule is state elimination based on weight
    void updateBelief(vector<Coalition> &nonSingletonCoals)
    {
        if (nonSingletonCoals.size() == 0)
        {
            // nothing to update. Beliefs don't change!
            for (auto &agent : game.agents)
            {
                agent.beliefChanged = false;
            }
            return;
        }

        for (auto &nonSingCoal : nonSingletonCoals)
        {
            // kinda wasteful to recompute this reward (should just pass it in!)
            // but whatever for now ...

            double reward = game.evaluateCoalition(nonSingCoal);
            //cout << ">> updating for ";
            //printVec(weights);
            //cout << endl;
            // agentName infers about otherAgentName
            for (int agentName : nonSingCoal)
            {
                //cout << "!!! AgentName " << agentName << " updating for " << endl;
                //printSet(nonSingCoal);
                //cout << endl;
                game.agents[agentName].beliefChanged = true;
                int lowerBoundWeight = game.rewardToThreshold[reward] - game.agents[agentName].weight - (nonSingCoal.size() - 2) * game.maxWeight;
                int upperBoundWeight = game.nextHigherThreshold[reward] - game.agents[agentName].weight - (nonSingCoal.size() - 2) * game.minWeight - 1;

                //cout << "lowerBoundWeight:" << lowerBoundWeight << endl;
                //cout << "upperBoundWeight:" << upperBoundWeight << endl;

                for (int otherAgentName : nonSingCoal)
                {
                    if (otherAgentName == agentName)
                    {
                        continue;
                    }
                    if (lowerBoundWeight - game.minWeight > 0)
                    {
                        // zero out "too low" weights
                        for (int i = 0; i < lowerBoundWeight - game.minWeight; i++)
                        {
                            game.agents[agentName].belief(otherAgentName, i) = 0;
                        }
                    }
                    if (game.maxWeight - upperBoundWeight > 0)
                    {
                        // zero out "too high" weights
                        for (int i = 0; i < game.maxWeight - upperBoundWeight; i++)
                        {
                            game.agents[agentName].belief(otherAgentName, i + upperBoundWeight -
                                                                              game.minWeight + 1) = 0;
                        }
                    }
                    // normalize to probability
                    game.agents[agentName].belief.row(otherAgentName) /= game.agents[agentName].belief.row(otherAgentName).sum();
                }
            }
        }
    }

    // updating based on Gaussian pdf with variance trustLevel
    // trustLevel >= 0, the bigger trustLevel the more weight we put
    // on the signal-based updating! This do not override the state-elimination
    // since 0 * anything = 0.
    void exploitSignal(double trustLevel)
    {
        for (auto &agent : game.agents)
        {
            for (auto &otherAgent : game.agents)
            {
                if (agent.name == otherAgent.name)
                {
                    continue;
                }
                double weightGuess = (agent.weight) * (otherAgent.currentWealth / agent.currentWealth);

                int i = 0;
                for (int weight : game.weightRange)
                {
                    double GaussianPdf = exp(-trustLevel * pow(weight - weightGuess, 2));
                    //cout << "gaussian pdf " << GaussianPdf << endl;
                    agent.belief(otherAgent.name, i) *= GaussianPdf;
                    i++;
                }
                // normalize row to 1.0
                agent.belief.row(otherAgent.name) /= agent.belief.row(otherAgent.name).sum();
            }
        }
    }
};
