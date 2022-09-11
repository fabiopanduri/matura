

class XorEnvNEAT:
    def __init__(self, max_t, plot=False, render=True):
        self.possible_actions = [0, 1]
        self.state_size = len(self.make_observation())

    def nn_base_dimensions(self):
        return [self.state_size, len(self.possible_actions)]

    def current_performance(self):
        return 1

    def fitness(self):
        return 1

    def make_observation(self):
        return 1

    def step(self):
        return 1

    def terminate_episode(self):
        return 1
