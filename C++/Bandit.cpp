#include "Bandit.hpp"
#include <math.h> /* exp */

// select proposals based on exp3
pair<int, Proposal> Bandit::proposalOutcome()
{
    // https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
    Agent &proposer = game.agents[distribution(generator)];
       game.proposerList.push_back(proposer.name);
    double W = proposer.proposalValues.sum();
    VectorXd probs = (1 - gamma) * proposer.proposalValues;
    probs /= W;
    probs += VectorXd::Constant(K, gamma / K);
    // if (probs.sum() != 1)
    // {
    //     cout << "SOMETHING IS WRONG BANDIT probs = " << probs << " WHICH IS != 1 " << endl;
    // }
    int chosen = selectAction(probs, generator, u);
    // cout << "proposer " << proposer.name << " chooses action " << chosen << endl;
    // printSet(proposer.proposalSpace[chosen].first);
    // cout << endl;
    // cout << proposer.proposalSpace[chosen].second << "\n ======" << endl;
        game.proposals.push_back(proposer.proposalSpace[chosen]);
    return make_pair(proposer.name, proposer.proposalSpace[chosen]);
}

pair<CoalitionStructure, vector<Coalition>> Bandit::formationProcess()
{
    // record the coalition Structure!
    CoalitionStructure CS;
    pair<int, Proposal> proposer_and_proposal = proposalOutcome();
    Proposal &proposal = proposer_and_proposal.second;
    Agent &proposer = game.agents[proposer_and_proposal.first];

    Coalition &coalition = proposal.first;
    VectorXd &div = proposal.second;
    map<int, int> responses;

    VectorXd singleShare(1);
    singleShare << 1.0;

    for (int agentName : coalition)
    {
        Agent &agent = game.agents[agentName];
        // the value of acceptance is as if this agent was to offer the same proposal himself
        // the value of rejection is the singleton offer
        // cout << "agent " << agentName << " responding " << endl;
        int accept_idx = getIndex(agent.proposalSpace, proposal);
        int reject_idx = getIndex(agent.proposalSpace, make_pair(set<int>{agentName}, singleShare));

        // cout << "accept_idx " << accept_idx << " reject_idx: " << reject_idx << endl;
        // cout << "accept proposal is ";
        // printSet(proposer.proposalSpace[accept_idx].first);
        // cout << endl;
        // cout << proposer.proposalSpace[accept_idx].second << "\n ======" << endl;
        // cout << "reject proposal is " << endl;
        // printSet(proposer.proposalSpace[reject_idx].first);
        // cout << endl;
        // cout << proposer.proposalSpace[reject_idx].second << "\n ======" << endl;
        //cout << "agent.QVs.size()" << agent.QVs.size() << endl;
        double accept_weight = agent.proposalValues(accept_idx);
        double reject_weight = agent.proposalValues(reject_idx);

        // cout << "accept weight is " << accept_weight << " reject_weight " << reject_weight << endl;
        double tot_weight = accept_weight + reject_weight;
        accept_weight = (1 - gamma) * (accept_weight / tot_weight) + gamma / 2;
        reject_weight = (1 - gamma) * (reject_weight / tot_weight) + gamma / 2;
        VectorXd accept_or_reject(2);
        accept_or_reject << accept_weight, reject_weight;
        VectorXd probs = softmax(accept_or_reject);
        int chosen = selectAction(probs, generator, u);
        //cout << "chosen:" << chosen << endl;
        if (chosen == 0)
        {
            responses[agentName] = 1; // yes!
        }
        else
        {
            responses[agentName] = 0; // no!
        }
        // cout << "response " << chosen << endl;
    }
    // delete my response since I don't care (exploration in the first place!)
    responses.erase(proposer.name);
    int agreement = accumulate(begin(responses), end(responses), 1,
                               [](int value, const std::map<int, int>::value_type &p) { return value * p.second; });

    // if (proposal.first.size() == 2)
    // {
    //     cout << "proposaal size == 2, agreement? " << agreement << endl;
    // }

    double coalitionReward = game.evaluateCoalition(proposal.first);

    if (agreement)
    {
        // updating beliefs as well ...
        updateProposerBelief(proposer, proposal, coalitionReward * proposal.second(distance(coalition.begin(), coalition.find(proposer.name))));
        for (int agentName : proposal.first)
        {
            if (agentName == proposer.name)
            {
                continue;
            }
            updateResponderBelief(game.agents[agentName], proposal, responses[agentName], coalitionReward);
        }

        //cout << "=>> Coalition agreement!" << endl;
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
        CS.push_back(make_tuple(coalition, div, game.evaluateCoalition(proposal.first)));
        return make_pair(CS, vector<Coalition>{coalition});
    }
    else
    {
        // all singleton coalitions
        // updating beliefs as well ...
        updateProposerBelief(proposer, proposal, game.evaluateCoalition({proposer.weight}));
        for (int agentName : proposal.first)
        {
            if (agentName == proposer.name)
            {
                continue;
            }
            Agent &agent = game.agents[agentName];
            updateResponderBelief(agent, proposal, responses[agentName], game.evaluateCoalition({agent.weight}));
        }

        for (auto &agent : game.agents)
        {
            CS.push_back(make_tuple(set<int>{agent.name},
                                    singleShare,
                                    game.evaluateCoalition({agent.weight})));
        }

        return make_pair(CS, vector<Coalition>{});
    }
}

// update the bandit weights for the proposer!
// success = whether the coalition was formed. 'No' ==> proposer gets singleton value!
void Bandit::updateProposerBelief(Agent &proposer, Proposal &proposal, double coalitionReward)
{
    Coalition &coalition = proposal.first;
    // cout << "updating for proposer" << endl;

    // normalize reward to [0, 1]
    coalitionReward = (coalitionReward - game.minReward) / (game.maxReward - game.minReward);
    double tot_weight = proposer.proposalValues.sum();
    int proposalIdx = getIndex(proposer.proposalSpace, proposal);
    double proposal_prob = (1 - gamma) * (proposer.proposalValues(proposalIdx) /
                                          tot_weight) +
                           gamma / K;
    double estimatedReward = coalitionReward / proposal_prob;
    proposer.proposalValues(proposalIdx) *= exp(gamma * (estimatedReward / K));
}

// response, 1 ==> 'yes', 0 ==> 'no'
void Bandit::updateResponderBelief(Agent &responder, Proposal &proposal, int response, double coalitionReward)
{
    // NOT DOING ANYTHING FOR NOW ...

    // Coalition &coalition = proposal.first;
    // cout << "updating for responder" << endl;

    // // normalize reward to [0, 1]
    // coalitionReward = (coalitionReward - game.minReward) / (game.maxReward - game.minReward);
    // double tot_weight = proposer.proposalValues.sum();
    // int proposalIdx = getIndex(proposer.proposalSpace, proposal);
    // double proposal_prob = (1 - gamma) * (proposer.proposalValues(proposalIdx) /
    //                                       tot_weight) +
    //                        gamma / K;
    // double estimatedReward = coalitionReward / proposal_prob;
    // proposer.proposalValues(proposalIdx) *= exp(gamma * (estimatedReward / K));
}

void Bandit::updateBelief(vector<Coalition> &nonSingletonCoals)
{
    // pass since we're already updating belief in formationProcess
}