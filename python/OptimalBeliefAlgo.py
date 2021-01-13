from utils import *
# HORIZON = 1, ONE-STEP PROPOSER
class OptimalBeliefAlgo:
    def __init__(self, game):
        self.game = game
        self.game.reset_belief()
        self.inform_belief()

    def inform_belief(self):
        # change belief of agents to correct belief!
        for agent in self.game.agents:
            # {'A': [0.5, 0.5], 'B': [1, 0]}
            for other_agent in self.game.agents:
                correct_type = other_agent.agent_type
                agent.belief[other_agent] = np.zeros(self.game.T)
                agent.belief[other_agent][correct_type-1] = 1.0

    def evaluate_coalition(self, coalition):
        W = [agent.agent_type for agent in coalition]
        return eval_coalition(W, self.game.tasks, False)

    def proposal_outcome(self):
        # one-step proposer process using informed belief
        # only return feasible coalition!! Singleton for the proposer
        # is always feasible!
        proposer = nature_pick_proposer(self.game.state)
        best_reward, best_proposals = float('-inf'), []

        # consider all proposals and all divisions possible!
        for proposal in generate_proposals(self.game.state, proposer):
            coalition, div = proposal
            responses, predicted_reward, continuation_payoff = \
                predict_responses(proposer, proposal, self.game, ret_extra=True)
            disagree = [response == 'no' for response in responses.values()]
            if any(disagree):
                # proposal fails! We'll short-circuit this for simplicity
                continue
                #proposal_value = continuation_payoff[proposer]# proposal fails! the proposer get the reserve value
            #else#:
                #proposal_value = div[coalition.index(proposer)] * predicted_reward
            proposal_value = div[coalition.index(proposer)] * predicted_reward

            if proposal_value > best_reward:
                best_reward = proposal_value
                best_proposals = [proposal]
            elif proposal_value == best_reward:
                best_proposals.append(proposal)

        i = np.random.randint(low=0, high=len(best_proposals))
        #print('proposal outcome:', proposer, best_proposal, best_div)
        return best_proposals[i], proposer

    def formation_process(self):
        # return CS which is a list of tuple (coalition, actual value of coalition, division_rule,
        # task)
        proposal, proposer = self.proposal_outcome()
        responses = predict_responses(proposer, proposal, self.game) # here, everyone has the same belief so
        # we just pick the proposer, arbitrarily, to give response prediction
        disagree = [response == 'no' for player, response in responses.items() if player != proposer]
        CS = []
        coalition, div = proposal
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
        return CS


    def update_belief(self, outcome, proposer=None, proposal=None):
        pass
