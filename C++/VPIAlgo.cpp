#include "VPIAlgo.hpp"

// select proposals based on softmax on QV values!
// TODO: memorize VPI values as well!
pair<int, Proposal> VPIAlgo::proposalOutcome()
{
    Agent &proposer = game.agents[distribution(generator)];

    //cout << ">> PROPOSER WEIGHT " << proposer.weight
    //     << " |beliefChanged? " << proposer.beliefChanged << " | bestProposal size : "
    //     << proposer.bestProposals.size() << endl;

    //cout << "PRINTING PROPOSAL SPACE:" << endl;
    //proposer.printProposalSpace();
    // only compute bestProposals again if my belief has changed!
    if (proposer.beliefChanged || proposer.proposalValues.size() == 0)
    {
        populateAgentValuesAndQVs(proposer);
    }

    //cout << ">> PROPOSER " << proposer.name << " proposalValues:" << proposer.proposalValues << endl;
    //cout << "proposalValues:" << proposer.proposalValues << endl;
    VectorXd probs = softmax(proposer.QVs);
    //cout << "softmax " << probs << endl;
    //cout << "softmax probs:" << probs << endl;
    //cout << "softmax sum :" << probs.sum() << endl;
    int chosen = selectAction(probs, generator, u);

    return make_pair(proposer.name, proposer.proposalSpace[chosen]);
}

void VPIAlgo::populateAgentValuesAndQVs(Agent &agent)
{

    int numProposals = agent.proposalSpace.size();
    double bestValue = 0;
    double secondBestValue = 0;
    set<int> bestProposals = {};

    if (agent.proposalValues.size() == 0)
    {
        agent.proposalValues.resize(numProposals);
    }

    if (agent.QVs.size() == 0)
    {
        agent.QVs.resize(numProposals);
    }

    double bestReward = numeric_limits<double>::min();
    for (auto [i, proposal] : enumerate(agent.proposalSpace))
    {
        pair<double, map<int, int>> val_and_responses = game.predictReponses(agent, proposal);
        // proposerValue = coalitionValue * proposerShare
        double proposalValue = (val_and_responses.first) * proposal.second[distance(proposal.first.begin(),
                                                                                    proposal.first.find(agent.name))];
        map<int, int> &responses = val_and_responses.second;
        // remove my response since I don't care about my response
        responses.erase(agent.name);
        // product of values of map
        // if there's any zero (any "no" response), then the product would
        // be 0.
        int agreement = accumulate(begin(responses), end(responses), 1,
                                   [](int value, const std::map<int, int>::value_type &p) { return value * p.second; });

        if (!agreement)
        {
            // get the singleton value ...
            proposalValue = game.evaluateCoalition({agent.weight});
        }
        agent.proposalValues(i) = proposalValue;

        if (proposalValue > bestValue)
        {
            secondBestValue = bestValue;
            bestValue = proposalValue;
            bestProposals = set<int>{i};
        }
        else if (proposalValue > secondBestValue and proposalValue < bestValue)
        {
            secondBestValue = proposalValue;
        }
        else if (proposalValue == bestValue)
        {
            bestProposals.insert(i);
        }
    }
    agent.QVs = agent.proposalValues + calculateVPIs(agent, bestProposals, bestValue,
                                                     secondBestValue);
}
// response according to QV values!
pair<CoalitionStructure, vector<Coalition>> VPIAlgo::formationProcess()
{
    //cout << "begin formationProc" << endl;
    // record the coalition Structure!
    CoalitionStructure CS;
    pair<int, Proposal> proposer_and_proposal = proposalOutcome();
    Proposal &proposal = proposer_and_proposal.second;
    Agent &proposer = game.agents[proposer_and_proposal.first];

    Coalition &coalition = proposal.first;
    VectorXd &div = proposal.second;
    //cout << "finish picking proposals!" << endl;
    map<int, int> responses;

    VectorXd singleShare(1);
    singleShare << 1.0;

    for (int agentName : coalition)
    {
        Agent &agent = game.agents[agentName];
        if (agent.beliefChanged || agent.proposalValues.size() == 0)
        {
            populateAgentValuesAndQVs(agent);
        }

        // the value of acceptance is as if this agent was to offer the same proposal himself
        // the value of rejection is the singleton offer
        int accept_idx = getIndex(agent.proposalSpace, proposal);
        int reject_idx = getIndex(agent.proposalSpace, make_pair(set<int>{agentName}, singleShare));

        //cout << "accept_idx " << accept_idx << " reject_idx: " << reject_idx << endl;
        //cout << "agent.QVs.size()" << agent.QVs.size() << endl;
        double accept_QV = agent.QVs(accept_idx);
        double reject_QV = agent.QVs(reject_idx);
        VectorXd accept_or_reject(2);
        accept_or_reject << accept_QV, reject_QV;
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
    }
    // delete my response since I don't care (exploration in the first place!)
    responses.erase(proposer.name);
    int agreement = accumulate(begin(responses), end(responses), 1,
                               [](int value, const std::map<int, int>::value_type &p) { return value * p.second; });

    //cout << "agreement? " << agreement << endl;
    if (proposal.first.size() == 2)
    {
        cout << "proposaal size == 2, agreement? " << agreement << endl;
    }
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

// this VPI calculation does NOT take into account rejection possibility, which might be problematic ...
// although this is probably mitigated by the fact that QV = proposalValue + VPI
VectorXd VPIAlgo::calculateVPIs(Agent &proposer, set<int> bestProposals, double bestValue,
                                double secondBestValue)
{
    VectorXd VPIs(proposer.proposalSpace.size());
    int proposalNum = -1;
    for (auto &proposal : proposer.proposalSpace)
    {
        proposalNum++;
        Coalition &coalition = proposal.first;
        VectorXd &div = proposal.second;
        // In Python, proposerShare = div[coalition.index(proposer)]
        double proposerShare = div(distance(coalition.begin(),
                                            coalition.find(proposer.name)));
        double VPI = 0;
        for (auto &weights : CartesianSelfProduct(game.weightRange, coalition.size()))
        {
            double prob = 1.0;
            for (auto [i, agentName] : enumerate(coalition))
            {
                prob *= proposer.belief(agentName, weights[i] - game.minWeight);
            }

            // TODO: change this to take into account rejection!
            double predictedReward = proposerShare * game.evaluateCoalition(weights);
            // if this proposal is in bestProposals
            if (bestProposals.find(proposalNum) != bestProposals.end() &&
                predictedReward < secondBestValue)
            {
                VPI += prob * (secondBestValue - predictedReward);
            }
            else if (predictedReward > bestValue)
            {
                VPI += prob * (predictedReward - bestValue);
            }
        }
        VPIs(proposalNum) = VPI;
    }
    return VPIs;
}