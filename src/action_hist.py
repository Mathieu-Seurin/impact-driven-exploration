import numpy as np
import torch

class FlatHisto:
    def __init__(self, n_actions, length=None):
        self.acted = torch.zeros(n_actions, dtype=torch.float)
        self.usage = torch.zeros(n_actions, dtype=torch.float)

    def add(self, action_id, count_action, acted_id, action_acted):
        self.usage[action_id] += count_action
        self.acted[acted_id] += action_acted

    def usage_ratio(self):
        return self.acted / self.usage


class RunningHisto:
    def __init__(self, n_actions, length):
        self.histos = []
        self.length = length
        self.next_switch_timestep = length

    def add(self, actions, acted):
        pass

    def usage_ratio(self):
        pass