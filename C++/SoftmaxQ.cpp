#include "SoftmaxQ.hpp"

// select proposals based on softmax!
pair<int, Proposal> SoftmaxQ::proposalOutcome()
{
    cout << "softmaxQ proposalOutcome " << endl;
    Agent &proposer = game.agents[distribution(generator)];

    int numProposals = proposer.proposalSpace.size();

    //cout << ">> PROPOSER WEIGHT " << proposer.weight
    //     << " |beliefChanged? " << proposer.beliefChanged << " | bestProposal size : "
    //     << proposer.bestProposals.size() << endl;

    //cout << "PRINTING PROPOSAL SPACE:" << endl;
    //proposer.printProposalSpace();
    // only compute bestProposals again if my belief has changed!
    if (proposer.beliefChanged || proposer.proposalValues.size() == 0)
    {
        //cout << "recomputing proposer.proposalValues" << endl;
        if (proposer.proposalValues.size() == 0)
        {
            proposer.proposalValues.resize(numProposals);
        }

        double bestReward = numeric_limits<double>::min();
        //cout << "size of proposal space: " << proposer.proposalSpace.size() << endl;
        int i = 0;
        for (auto &proposal : proposer.proposalSpace)
        {
            pair<double, map<int, int>> val_and_responses = game.predictReponses(proposer, proposal);
            // proposerValue = coalitionValue * proposerShare
            double proposalValue = (val_and_responses.first) * proposal.second[distance(proposal.first.begin(),
                                                                                        proposal.first.find(proposer.name))];
            map<int, int> &responses = val_and_responses.second;
            // remove my response since I don't care about my response
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
            proposer.proposalValues(i) = proposalValue;
            i++;
        }
    }

    //cout << ">> PROPOSER " << proposer.name << " proposalValues:" << proposer.proposalValues << endl;
    //cout << "proposalValues:" << proposer.proposalValues << endl;
    VectorXd probs = softmax(proposer.proposalValues);
    //cout << "softmax " << probs << endl;
    //cout << "softmax probs:" << probs << endl;
    //cout << "softmax sum :" << probs.sum() << endl;

    int chosen = selectAction(probs, generator, u);

    //cout << ">> PROPOSER : " << proposer.name << " CHOOSE " << endl;
    //printSet(proposer.proposalSpace[chosen].first);
    //cout << "\n" << proposer.proposalSpace[chosen].second << endl;
    //cout << "chosen index is " << chosen << " with prob " << probs[chosen] << endl;
    return make_pair(proposer.name, proposer.proposalSpace[chosen]);
}

// select proposals based on softmax!
// return proposer and chosen
pair<int, int> SoftmaxQ::proposalOutcomeTest()
{
    Agent &proposer = game.agents[distribution(generator)];

    int numProposals = proposer.proposalSpace.size();

    if (proposer.proposalValues.size() == 0)
    {
        proposer.proposalValues.resize(numProposals);
    }

    double bestReward = numeric_limits<double>::min();
    //cout << "size of proposal space: " << proposer.proposalSpace.size() << endl;
    int i = 0;
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
        proposer.proposalValues(i) = proposalValue;
        i++;
    }

    VectorXd probs = softmax(proposer.proposalValues);
    int chosen = selectAction(probs, generator, u);

    return make_pair(proposer.name, chosen);
}

void SoftmaxQ::testProposal()
{
    int N = 10000;
    int M = game.agents[0].proposalSpace.size();

    pair<int, int> proposer_and_chosen;
    map<pair<int, int>, int> counts;
    double tot_proposer_first;
    for (int i = 0; i < N; i++)
    {
        proposer_and_chosen = proposalOutcomeTest();
        counts[proposer_and_chosen] += 1;
        if (proposer_and_chosen.first == 0)
        {
            tot_proposer_first += 1;
        }
    }

    cout << "1st proposer appears " << tot_proposer_first << endl;
    cout << "Now let's see their freqs " << endl;

    Agent &proposer = game.agents[0];
    VectorXd probs = softmax(proposer.proposalValues);

    // see the 0st proposer!
    for (int j = 0; j < M; j++)
    {
        double approx = counts[make_pair(0, j)] / tot_proposer_first;
        double rel_error = abs(approx - probs[j]) / (probs[j]);
        cout << "proposalIndex " << j << " count " << counts[make_pair(0, j)]
             << " approx prob " << approx << " actual prob "
             << probs[j] << " difference " << rel_error << endl;
        if (rel_error > 0.01)
        {
            cout << "WARNING!!!! ERROR TOO HIGH!!" << endl;
        }
    }
}

pair<CoalitionStructure, vector<Coalition>> SoftmaxQ::formationProcess()
{
    cout << "softmaxQ formationProcess " << endl;
    // record the coalition Structure!
    CoalitionStructure CS;
    pair<int, Proposal> proposer_and_proposal = proposalOutcome();
    Proposal &proposal = proposer_and_proposal.second;
    Agent &proposer = game.agents[proposer_and_proposal.first];

    Coalition &coalition = proposal.first;
    VectorXd &div = proposal.second;
    map<int, int> responses;

    for (int agentName : coalition)
    {
        responses[agentName] = game.predictReponses(game.agents[agentName], proposal, {agentName}).second[agentName];
    }
    // delete my response since I don't care (exploration in the first place!)
    responses.erase(proposer.name);
    int agreement = accumulate(begin(responses), end(responses), 1,
                               [](int value, const std::map<int, int>::value_type &p) { return value * p.second; });

    //if (proposal.first.size() == 2)
    //{
    //    cout << "proposaal size == 2, agreement? " << agreement << endl;
    //}

    VectorXd singleShare(1);
    singleShare << 1.0;
    if (agreement)
    {
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
        CS.push_back(make_tuple(coalition, div, game.evaluateCoalition(coalition)));
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