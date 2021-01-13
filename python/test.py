import unittest
from BayesianUpdateBelief import *
from BayesianOptimize import *
from Task import *
from Game import *



#logging.basicConfig(level=logging.INFO)




class TestBayesianOptimize(unittest.TestCase):
    def setUp(self):
        T = 2  # no. of types
        N = 3  # no. of players
        horizon = 1  # how many proposal rounds before a game terminate
        tasks = [Task(threshold=1, reward=1),
                 Task(threshold=2, reward=3),
                 Task(threshold=4, reward=5.9)]

        W = [1, 1, 1]
        self.game = Game(T, N, tasks, horizon, W)


    def set_belief(self, beliefs):
        '''
        :param beliefs:
        dictionary of list of beliefs
        Key = the player whose belief we want to reset
        Value = dictionary e.g. {'A': [1, 0, 0], 'B':[0.5, 0.5, 0]}
        :return:
        '''
        for player_name, mus in beliefs.items():
            player = self.game.agent_lookup[player_name]
            for mu in mus:
                player.set_belief(player, mu, self.game)


    def test_expected_coalition_given_type(self):
        agents = list(self.game.agents)

        # test 1: B observes (A | type = 1) ==> 1
        proposer = agents[0]
        observer = agents[1]
        proposer_type = 0
        other_players = []
        self.assertAlmostEqual(expected_coalition_given_type(observer,
                                                             proposer,
                                                             proposer_type,
                                                             other_players,
                                                             self.game,
                                                             self.game.state),
                               1.0)
        # test 2: A observes (A | type = 1) ==> 1
        observer = agents[0]
        self.assertAlmostEqual(expected_coalition_given_type(observer,
                                                             proposer,
                                                             proposer_type,
                                                             other_players,
                                                             self.game,
                                                             self.game.state),
                               1.0)

        # test 3: B observes (A, B | proposer A type = 1) ==> 2
        observer = agents[1]
        other_players = [agents[1]]
        self.assertAlmostEqual(expected_coalition_given_type(observer,
                                                             proposer,
                                                             proposer_type,
                                                             other_players,
                                                             self.game,
                                                             self.game.state),
                               2.0)

        # test 4: A observes (A, B | proposer A type = 1) ==> (0.5) * (2 + 0) = 1.0
        observer = agents[0]
        self.assertAlmostEqual(expected_coalition_given_type(observer,
                                                             proposer,
                                                             proposer_type,
                                                             other_players,
                                                             self.game,
                                                             self.game.state),
                               1.0)
        # test 5: B observes (A | proposer type = 2) ==> 3.0
        proposer = agents[0]
        observer = agents[1]
        proposer_type = 2
        other_players = []
        self.assertAlmostEqual(expected_coalition_given_type(observer,
                                                             proposer,
                                                             proposer_type,
                                                             other_players,
                                                             self.game,
                                                             self.game.state),
                               3.0)

if __name__ == '__main__':
    unittest.main()