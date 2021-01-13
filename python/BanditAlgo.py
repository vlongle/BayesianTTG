from utils import *

# Exp3 algorithm
# TODO: set gamma properly!!
class BanditAlgo:
    def __init__(self, game, gamma=0.1):
        self.game = game
        self.setup_bandit_weights()
        self.gamma = gamma
        self.game.reset_belief()

    def setup_bandit_weights(self):
        for agent in self.game.agents:
            agent.proposal_weights = defaultdict(float)
            for proposal in generate_proposals(self.game.state, agent, True):
                #print('proposal:', proposal)
                agent.proposal_weights[proposal] = 1.0

    def evaluate_coalition(self, coalition):
        W = [agent.agent_type for agent in coalition]
        return eval_coalition(W, self.game.tasks, False)

    def proposal_outcome(self):
        # one-step proposer process using informed belief
        # only return feasible coalition!! Singleton for the proposer
        # is always feasible!
        proposer = nature_pick_proposer(self.game.state)


        proposals = [] # list of proposals
        probs = []
        tot_weight = sum(proposer.proposal_weights.values())
        for proposal, weight in proposer.proposal_weights.items():
            probs.append((1-self.gamma) * (weight/tot_weight) + \
                    self.gamma/len(proposer.proposal_weights))
            proposals.append(proposal)

        chosen = np.random.choice(range(len(proposals)), p=probs)
        return proposals[chosen], proposer



    def response(self, proposal):
        # also follows bandit strategy restricted to two choices: accept
        # or reject
        # TODO: think about this more. Do we want bandit strategy here??
        coalition, div = proposal
        responses = {}
        for agent in coalition:
            # accept prob: proportional to the weight of this proposal action
            # reject prob: proportional to the weight of the singleton proposal

            accept_w = agent.proposal_weights[proposal]
            #print('proposal in response:', proposal)
            #print('agent', agent, 'proposer weight is:',  agent.proposal_weights)
            reject_w = agent.proposal_weights[((agent,), (1.0,))]
            #print('accept, reject:', accept_w, reject_w)
            tot_w = accept_w + reject_w
            accept = (1-self.gamma)*(accept_w/tot_w) + self.gamma/2
            reject = (1-self.gamma)*(reject_w/tot_w) + self.gamma/2
            agent_response = np.random.choice(['yes', 'no'], p=[accept, reject])
            responses[agent] = agent_response

        return responses

    def formation_process(self):
        # return CS which is a list of tuple (coalition, actual value of coalition, division_rule,
        # task)
        proposal, proposer = self.proposal_outcome()
        responses = self.response(proposal)
        #print('\n===> \t \t PROPOSAL:', proposal, 'proposer:', proposer, 'response:', responses)
        disagree = [response == 'no' for player, response in responses.items() if player != proposer]
        coalition, div = proposal
        CS = []
        if any(disagree):
            # coalition fails to form everyone forms singleton coalition
            for agent in self.game.state.active_agents:
                task, reward = self.evaluate_coalition([agent])
                CS.append(([agent], reward, [1.0], task))
        else:
            # one non-singleton coalition is formed
            #if len(coalition) > 1:
            #    print('REACH AGREEMENT!!')
            for agent in self.game.state.active_agents:
                if agent in coalition:
                    continue
                task, reward = self.evaluate_coalition([agent])
                CS.append(([agent], reward, [1.0], task))

            task, reward = self.evaluate_coalition(coalition)
            CS.append((coalition, reward, div, task))
        return CS, proposer, proposal
# https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
    def update_belief(self, CS, proposer, proposal):
        # the proposer update his belief!
        for coalition, coalition_value, div, task in CS:
            if proposer in coalition:
                proposer_reward = div[coalition.index(proposer)] * coalition_value
                #print('unnormalized reward:', proposer_reward)
                # normalize proposer_reward to be in [0, 1]
                proposer_reward = (proposer_reward - self.game.min_reward)/(self.game.max_reward-
                                                                            self.game.min_reward)
                #print('normalized reward:', proposer_reward)
                # estimated reward (with propensity correction)
                tot_weight = sum(proposer.proposal_weights.values())
                proposal_prob = (1 - self.gamma) * (proposer.proposal_weights[proposal]/ tot_weight) + \
                self.gamma / len(proposer.proposal_weights)
                est_reward = proposer_reward/proposal_prob
                proposer.proposal_weights[proposal] *= np.exp(self.gamma * est_reward/len(proposer.proposal_weights))
                break
