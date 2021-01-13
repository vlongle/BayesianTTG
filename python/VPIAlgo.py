from utils import *
#from scipy.special import softmax
from stateEliminationAlgo import *

class VPIAlgo(StateEliminationAlgo):
    def generate_proposals(self):
        # compute all possible proposals that this agent can give
        for agent in self.game.agents:
            agent.proposals = generate_proposals(self.game.state, agent)

    def evaluate_coalition(self, coalition):
        W = [agent.agent_type for agent in coalition]
        return eval_coalition(W, self.game.tasks, False)
    @jit
    def softmax(self, x):
        # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def calculate_VPIs(self, proposer, best_proposals, best_value, second_best, proposals):
        VPIs = [] # value of perfect information of each of the proposals
        for proposal in proposals:
            coalition, div = proposal
            proposer_share =div[coalition.index(proposer)]
            VPI = 0
            for agent_types in product(range(1, self.game.T + 1), repeat=len(coalition)):
                prob = reduce(mul, \
                              [proposer.belief[agent][other_agent_type - 1] for \
                               agent, other_agent_type in zip(coalition, agent_types)])
                predicted_reward = proposer_share * eval_coalition(list(agent_types), self.game.tasks)
                if proposal in best_proposals and predicted_reward < second_best:
                    VPI += prob * (second_best - predicted_reward)
                elif predicted_reward > best_value:
                    VPI += prob * (predicted_reward - best_value)
            VPIs.append(VPI)
        return VPIs
    def proposal_outcome(self):
        # one-step proposer process using informed belief
        # only return feasible coalition!! Singleton for the proposer
        # is always feasible!
        proposer = nature_pick_proposer(self.game.state)


        proposals = [] # list of proposals
        proposal_values = [] # corresponding list of values
        best_value = second_best = 0
        best_proposals = set()
        # consider all proposals and all divisions possible!
        # choose proposal based on softmax selection to facilitate exploration
        #for proposal in proposer.proposals:
        for proposal in generate_proposals(self.game.state, proposer, True):
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
            if proposal_value > best_value:
                second_best = best_value
                best_value = proposal_value
                best_proposals = set(proposal)
            elif proposal_value > second_best and\
                proposal_value < best_value:
                second_best = proposal_value
            elif proposal_value == best_value:
                best_proposals.add(proposal)

        VPIs = self.calculate_VPIs(proposer, best_proposals, best_value, second_best, proposals)

        QVs = np.array(VPIs) + np.array(proposal_value)
        probs = self.softmax(QVs)
        chosen = np.random.choice(range(len(proposals)), p=probs)
        #print('ret:', proposals[ret])
        return proposals[chosen], proposer



    def response(self, proposal):
        coalition, div = proposal
        responses = {}
        for agent in coalition:
            # each agent calculate their QVs value again restricted to only yes (current proposal)
            # and no (singleton proposal) and then use softmax selection!

            accept_reward = div[coalition.index(agent)] * expected_coalition_value(coalition, agent, self.game)
            reject_reward = expected_continuation_payoff(agent, self.game)[agent]
            #rewards = sort
            #self.calculate_VPIs(agent, best_proposals, best_value, second_best, proposals)

            # simply softmax these guys for now to avoid dealing with edge cases when two choices are already equal!
            probs = self.softmax([accept_reward, reject_reward])
            agent_response = np.random.choice(['yes', 'no'], p=probs)
            responses[agent] = agent_response
        return responses
