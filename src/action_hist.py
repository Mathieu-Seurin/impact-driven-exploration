import numpy as np
import torch
import copy

class FlatHisto:
    def __init__(self, n_actions, length=None):
        self.acted = torch.zeros(n_actions, dtype=torch.float)
        self.usage = torch.zeros(n_actions, dtype=torch.float)

    def add(self, action_id, count_action, acted_id, count_acted):
        self.usage[action_id] += count_action
        self.acted[acted_id] += count_acted

    def usage_ratio(self):
        return self.acted / self.usage

    def return_full_hist(self):
        return dict(
            [('usage', self.usage),
             ('acted', self.acted)]
        )


class WindowedHisto:
    def __init__(self, n_actions, length):
        """
        :param n_actions: number of action in the environement
        :param length: max number of action being remembered, to allow a sliding window
        """

        self.n_actions = n_actions
        self.old_histos = []
        self.current_histo = self._init_hist()

        self.length = length
        self.hist_size = length // 10

        self.total_hist_length = 0

    def _init_hist(self):
        return dict([
            ('acted', torch.zeros(self.n_actions, dtype=torch.float)),
            ('usage', torch.zeros(self.n_actions, dtype=torch.float))
        ])

    def add(self, action_id, count_action, acted_id, count_acted):
        self.current_histo['usage'][action_id] += count_action
        self.current_histo['acted'][acted_id] += count_acted

        self.total_hist_length += count_action.sum()

        if self.current_histo['usage'].sum() > self.hist_size:
            self.old_histos.append(self.current_histo)
            self.current_histo = self._init_hist()

        if self.total_hist_length > self.length:
            h = self.old_histos.pop(0)
            self.total_hist_length -= h['usage'].sum() # reduce total length when you remove an old hist


    def usage_ratio(self):
        acted = torch.zeros(self.n_actions, dtype=torch.float)
        usage = torch.zeros(self.n_actions, dtype=torch.float)

        acted += self.current_histo['acted']
        usage += self.current_histo['usage']

        for h in self.old_histos:
            acted += h['acted']
            usage += h['usage']

        return acted / usage

    def return_full_hist(self):
        acted = torch.zeros(self.n_actions, dtype=torch.float)
        usage = torch.zeros(self.n_actions, dtype=torch.float)

        acted += self.current_histo['acted']
        usage += self.current_histo['usage']

        for h in self.old_histos:
            acted += h['acted']
            usage += h['usage']

        return dict(
            [('usage', usage),
             ('acted', acted)]
        )



if __name__ == "__main__":

    n_actions = 6
    histo = WindowedHisto(n_actions=n_actions, length=int(5e7))

    action_id = torch.LongTensor(range(n_actions))
    acted_id = torch.LongTensor(range(n_actions))

    for i in range(10000):

        usage = torch.randint(0, 100000, size=(n_actions,), dtype=torch.float)
        acted = usage - torch.randint(0, 10000, size=(n_actions,), dtype=torch.float)
        acted = torch.clamp_min(acted, 0)

        histo.add(action_id=action_id,
                  acted_id=acted_id,
                  count_action=usage,
                  count_acted=acted)
        ratio = histo.usage_ratio()








