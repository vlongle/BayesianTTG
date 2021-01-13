from utils import *
#from scipy.special import softmax
from stateEliminationAlgo import *

class softmaxSelectionStateElimination(StateEliminationAlgo):
    def generate_proposals(self):
        # compute all possible proposals that this agent can give
        for agent in self.game.agents:
            agent.proposals = generate_proposals(self.game.state, agent)

    def evaluate_coalition(self, coalition):
        W = [agent.agent_type for agent in coalition]
        return eval_coalition(W, self.game.tasks, False)

    def proposal_outcome_test(self):
        N = 10000
        counts = defaultdict(int)
        tot = 0
        for i in range(N):
            proposer, chosen = self.proposal_outcome()
            counts[(proposer, chosen)] += 1
            if str(proposer) == 'A.1':
                tot +=1

        print('tot:', tot)
        proposerfirst = self.game.agent_lookup['A']
        probs = self.softmax(proposerfirst.proposal_values)

        for j in range(89):
            freq = counts[(proposer, j)]/tot
            p = probs[j]
            diff = abs(freq-p)/p
            print(j, 'freq:', freq, 'prob:', p, 'diff:', diff)


    def proposal_outcome(self):
        # one-step proposer process using informed belief
        # only return feasible coalition!! Singleton for the proposer
        # is always feasible!
        proposer = nature_pick_proposer(self.game.state)


        proposals = [] # list of proposals
        proposal_values = [] # corresponding list of values
        # consider all proposals and all divisions possible!
        # choose proposal based on softmax selection to facilitate exploration
        #for proposal in proposer.proposals:
        for proposal in generate_proposals(self.game.state, proposer):
            coalition, div = proposal
            responses, predicted_reward, continuation_payoff = \
                predict_responses(proposer, proposal, self.game, ret_extra=True)
            disagree = [response == 'no' for player, response in responses.items() if player != proposer]
            if any(disagree):
                proposal_value = continuation_payoff[proposer]# proposal fails! the proposer get the reserve value
            else:
                proposal_value = div[coalition.index(proposer)] * predicted_reward

            #print('proposal, responses, predicted_reward, cont_payoff, proposal_val:', proposal,
            #      responses, predicted_reward, continuation_payoff, proposal_value)

            proposals.append(proposal)
            proposal_values.append(proposal_value)

        #print('proposals:', proposals)
        #print('===> proposer:', proposer)
        probs = self.softmax(proposal_values)
        chosen = np.random.choice(range(len(proposals)), p=probs)
        #print('ret:', proposals[ret])
        if len(proposals[chosen][0]) == 2:
            print('proposer', proposer, 'chooses:', proposals[chosen], 'that has probability: ',
                  probs[chosen])
        return proposals[chosen], proposer

    def proposal_outcome_v2(self):
        # one-step proposer process using informed belief
        # only return feasible coalition!! Singleton for the proposer
        # is always feasible!
        proposer = nature_pick_proposer(self.game.state)


        proposals = [] # list of proposals
        proposal_values = [] # corresponding list of values
        # consider all proposals and all divisions possible!
        # choose proposal based on softmax selection to facilitate exploration
        #for proposal in proposer.proposals:
        if not proposer.proposal_values:
            for proposal in generate_proposals(self.game.state, proposer):
                coalition, div = proposal
                responses, predicted_reward, continuation_payoff = \
                    predict_responses(proposer, proposal, self.game, ret_extra=True)
                disagree = [response == 'no' for player, response in responses.items() if player != proposer]
                if any(disagree):
                    proposal_value = continuation_payoff[proposer]# proposal fails! the proposer get the reserve value
                else:
                    proposal_value = div[coalition.index(proposer)] * predicted_reward

                #print('proposal, responses, predicted_reward, cont_payoff, proposal_val:', proposal,
                #      responses, predicted_reward, continuation_payoff, proposal_value)

                proposals.append(proposal)
                proposal_values.append(proposal_value)
            proposer.proposal_values = proposal_values

        #print('proposals:', proposals)
        #print('===> proposer:', proposer)
        #print('proposal_values:', proposal_values)
        proposals = range(89)
        probs = self.softmax(proposer.proposal_values)
        chosen = np.random.choice(range(len(proposals)), p=probs)
        #print('ret:', proposals[ret])
        #print('proposer', proposer, 'chooses:', proposals[chosen], 'that has probability: ',
        #      probs[chosen])
        #return proposals[chosen], proposer
        return proposer, chosen



    def response(self, proposal):
        coalition, div = proposal
        responses = {}
        for agent in coalition:
            # predict_response:
            # create a version to make it softmax as well!!!
            agent_response = predict_responses(agent, proposal, self.game)[agent]
            responses[agent] = agent_response

        return responses
