#include "SignalSoftmaxQ.hpp"

// calculate the regularizer term for each proposal in proposer's space
VectorXd SignalSoftmaxQ::calculateRegularizer(Agent &proposer)
{
    VectorXd regularizer(proposer.proposalSpace.size());
    for (auto [i, proposal] : enumerate(proposer.proposalSpace))
    {
        int invCount = countInversionOfProposal(proposer, proposal);
        double regularizeTerm = (1 / (1 + invCount)) * lambda;
        regularizer(i) = regularizeTerm;

        // debug
        cout << "===========================" << endl;
        cout << ">> coalition " << endl;
        printSet(proposal.first);
        cout << "\n >> divisionRule" << endl;
        cout << proposal.second << endl;
        cout << "regularizer: " << regularizeTerm << endl;
        cout << "===========================" << endl;
    }
    return regularizer;
}

double SignalSoftmaxQ::countInversionOfProposal(Agent &proposer, Proposal &proposal)
{
    std::vector<std::pair<int, double>> zipped;
    zip(proposal, zipped);
    // sort zipped array based on the division rule
    // sorting in ascending order
    sort(begin(zipped), end(zipped),
         [&](const auto &a, const auto &b) {
             return a.second < b.second;
         });

    // we now have coalition sorted by division share
    VectorXd coalitionSortedByShare(proposal.first.size());
    VectorXd sortedDiv(proposal.second.size());
    unzip(zipped, coalitionSortedByShare, sortedDiv);

    // now sort the coalition based on public signal

    vector<int> coalitionSortedBySignal = set2vec(proposal.first);
    sort(begin(coalitionSortedBySignal), end(coalitionSortedBySignal),
         [&](const auto &a, const auto &b) {
             return game.agents[a].currentWealth < game.agents[b].currentWealth;
         });

    map<int, int> agentToRankBySignal;

    for (auto [i, agentName] : enumerate(coalitionSortedBySignal))
    {
        agentToRankBySignal[agentName] = i;
    }

    int inversionArray[coalitionSortedBySignal.size()];
    for (int i = 0; i < coalitionSortedBySignal.size(); i++)
    {
        inversionArray[i] = agentToRankBySignal[coalitionSortedByShare[i]];
    }

    return getInvCount(inversionArray, coalitionSortedBySignal.size());
}

// select proposals based on softmax plus a regularizer term
pair<int, Proposal> SignalSoftmaxQ::proposalOutcome()
{
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
    // augment proposalValues now with regularizer term!

    //cout << ">> PROPOSER " << proposer.name << " proposalValues:" << proposer.proposalValues << endl;
    //cout << "proposalValues:" << proposer.proposalValues << endl;
    // ===============================================================
    // !!!! NOTE: The following inclusion of regularizer is the difference between SoftmaxQ and SignalSoftmaxQ
    // ===============================================================
    VectorXd probs = softmax(proposer.proposalValues + calculateRegularizer(proposer));
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

// verbatim from SoftmaxQ
pair<CoalitionStructure, vector<Coalition>> SignalSoftmaxQ::formationProcess()
{
    cout << "signalSoftmaxQ formationProcess " << endl;
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