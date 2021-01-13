class Task:
    def __init__(self, threshold, reward):
        self.threshold = threshold
        self.reward = reward
        self.name = 'Task(threshold={},reward={})'.format(self.threshold, self.reward)
    def __repr__(self):
        return self.name