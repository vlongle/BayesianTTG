#include "InformedBeliefAlgo.hpp"

void InformedBeliefAlgo::informBelief()
{
    //cout << "informBelief!" << endl;
    for (auto &agent : game.agents)
    {
        for (int otherAgent = 0; otherAgent < game.agents.size(); otherAgent++)
        {
            //cout << "agent: " << agent.name << " changing belief of otherAgent " << otherAgent << endl;
            agent.belief.row(otherAgent) *= 0; // set row to 0
            double correctWeight = game.agents[otherAgent].weight;
            agent.belief(otherAgent, correctWeight - game.minWeight) = 1.0; // turn on the correct weight
            //cout << "res: " << agent.belief.row(otherAgent) << endl;
        }
    }
    //cout << "Agent 0 belief:" << game.agents[0].belief << endl;
    //cout << "InformedBelief address of game: " << &game << endl;
    //cout << "InformedBelief address of agents[0]: " << &game.agents[0] << endl;
}

pair<int, Proposal> InformedBeliefAlgo::proposalOutcome()
{
    Agent &proposer = game.agents[distribution(generator)];
    game.proposerList.push_back(proposer.name);
    game.proposerList.push_back(proposer.name);
    // cout << "inform belief proposal is " << proposer.name << endl;

    // only compute bestProposals again if my belief has changed!
    if (proposer.beliefChanged || proposer.bestProposals.size() == 0)
    {
        //cout << ">> PROPOSER WEIGHT " << proposer.weight
        //<< " |beliefChanged? " << proposer.beliefChanged << " | bestProposal size : "
        //<< proposer.bestProposals.size() << endl;
        double bestReward = numeric_limits<double>::min();
        vector<Proposal> bestProposals;
        //cout << "size of proposal space: " << proposer.proposalSpace.size() << endl;
        for (auto &proposal : proposer.proposalSpace)
        {
            pair<double, map<int, int>> val_and_responses = game.predictReponses(proposer, proposal);
            // proposerValue = coalitionValue * proposerShare
            double proposalValue = (val_and_responses.first) * proposal.second[distance(proposal.first.begin(),
                                                                                        proposal.first.find(proposer.name))];
            map<int, int> &responses = val_and_responses.second;
            // remove my response since I don't care about mine response
            responses.erase(proposer.name);
            // product of values of map
            // if there's any zero (any "no" response), then the product would
            // be 0.
            int agreement = accumulate(begin(responses), end(responses), 1,
                                       [](int value, const std::map<int, int>::value_type &p) { return value * p.second; });

            if (!agreement)
            {
                // get the singleton value ...
                proposalValue = game.evaluateCoalition({proposer.weight});
            }

            if (proposalValue > bestReward)
            {
                bestReward = proposalValue;
                bestProposals = {proposal};
            }
            else if (proposalValue == bestReward)
            {
                bestProposals.push_back(proposal);
            }
        }
        proposer.bestProposals = bestProposals;
    }
    // cout << "best proposal " << endl;
    // // debug
    // for (auto &proposal : proposer.bestProposals)
    // {
    //     cout << "===========================" << endl;
    //     cout << ">> coalition " << endl;
    //     printSet(proposal.first);
    //     cout << "\n >> divisionRule" << endl;
    //     cout << proposal.second << endl;
    //     cout << "===========================" << endl;
    // }
    // randomly choose an element from the best proposals!
    // TODO: test this!!
    uniform_int_distribution<int> dist(0, proposer.bestProposals.size() - 1);
    //cout << "ProposalOutcome done!" << endl;
    int chosen = dist(generator);
    game.proposals.push_back(proposer.proposalSpace[chosen]);

    return make_pair(proposer.name, proposer.bestProposals[chosen]);
}

// second return is a vector of non-singleton coalitions. The number of such coalitions
// is either 0 or 1
pair<CoalitionStructure, vector<Coalition>> InformedBeliefAlgo::formationProcess()
{
    //cout << "FormationProcess begin!" << endl;
    // record the coalition Structure!
    CoalitionStructure CS;
    pair<int, Proposal> proposer_and_proposal = proposalOutcome();
    Proposal &proposal = proposer_and_proposal.second;
    Agent &proposer = game.agents[proposer_and_proposal.first];

    Coalition &coalition = proposal.first;
    VectorXd &div = proposal.second;
    // Here, everyone has the same belief so
    // we just pick the proposer, arbitrarily, to give response prediction
    pair<double, map<int, int>> val_and_responses = game.predictReponses(proposer, proposal);
    double expectedCoalValue = val_and_responses.first;
    map<int, int> &responses = val_and_responses.second;
    int agreement = accumulate(begin(responses), end(responses), 1,
                               [](int value, const std::map<int, int>::value_type &p) { return value * p.second; });

    VectorXd singleShare(1);
    singleShare << 1.0;

    if (agreement)
    {
        // one (potentially) non-singleton coalition
        for (auto &agent : game.agents)
        {
            // if agent is in coalition
            if (coalition.find(agent.name) != coalition.end())
            {
                continue;
            }
            // agent not found in coalition
            CS.push_back(make_tuple(set<int>{agent.name},
                                    singleShare,
                                    game.evaluateCoalition({agent.weight})));
        }
        //CS.push_back(make_tuple(coalition, div, game.evaluateCoalition(proposal.first)));
        CS.push_back(make_tuple(coalition, div, expectedCoalValue));
        return make_pair(CS, vector<Coalition>{coalition});
    }
    else
    {
        // all singleton coalitions
        for (auto &agent : game.agents)
        {
            CS.push_back(make_tuple(set<int>{agent.name},
                                    singleShare,
                                    game.evaluateCoalition({agent.weight})));
        }

        return make_pair(CS, vector<Coalition>{});
    }
}

void InformedBeliefAlgo::updateBelief(vector<Coalition> &nonSingletonCoals)
{
    // pass
}

void InformedBeliefAlgo::exploitSignal(double trustLevel)
{
    // pass
}
