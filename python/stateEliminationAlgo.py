from utils import *

class StateEliminationAlgo:
    def __init__(self, game):
        self.game = game
        self.game.reset_belief()
    def update_belief(self, CS, proposer=None, proposal=None):
        #print('==> updating belief for CS : ', CS)
        # Very conservative belief update based on state elimination
        # agents in each coalition observe the actual coalition value. They
        # now update their belief to eliminate worlds that are not consistent!
        # That is eliminate all weights that are too high or too low
        for coalition, coalition_val, div, task in CS:
            if len(coalition) == 1:
                continue # skip for singleton -- nothing to update!

            print('>> updating for coalition ', coalition)
            higher_threshold_task = find_higher_threshold_task(self.game.tasks, task)
            # for each agent: update the belief about other agent
            for agent in coalition:
                # any other agent must has type at least this threshold!
                lower_bound_weight = task.threshold - agent.agent_type - \
                                     (len(coalition) - 2) * self.game.max_type

                upper_bound_weight = 10000
                if higher_threshold_task:
                    upper_bound_weight = higher_threshold_task.threshold - agent.agent_type\
                                        - (len(coalition)-2) * self.game.min_type
                    # each agent type is >= lower_bound_weight and < (strictly) upper_bound_weight
                    upper_bound_weight -= 1 # now it lower <= weight <= upper
                #print('agent:', agent, 'coalition:', coalition, 'lower_bound:', lower_bound_weight,
                #      'upper_bound:', upper_bound_weight, 'task:', task, 'higher_threshold_task:',
                #      higher_threshold_task)
                # zero-ing out all the types outside this range (world elimination!)

                for other_agent in coalition:
                    if other_agent == agent:
                        continue
                    # zero out "too low" weight
                    if lower_bound_weight - self.game.min_type > 0:
                        for i in range(lower_bound_weight - self.game.min_type):
                            agent.belief[other_agent][i] = 0

                    if higher_threshold_task and self.game.max_type - upper_bound_weight > 0:
                        # zero out "too high" weight
                        for i in range(self.game.max_type - upper_bound_weight):
                            agent.belief[other_agent][i + upper_bound_weight - self.game.min_type + 1] = 0
                    # normalize to probability vector!
                    agent.normalize_to_prob(other_agent)


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
        return CS
    @jit
    def softmax(self, x):
        # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference