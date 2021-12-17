from math import ceil

import torch
from torch import tensor, Tensor


class NumberBin:

    def __init__(self, lower=0., upper=1., nr_of_bins=None):
        self.lower = lower
        self.upper = upper
        self.width = ceil(self.upper - self.lower)
        self.nr_of_bins = nr_of_bins
        if self.nr_of_bins is None:
            self.nr_of_bins = min(3, max(1000, self.width * 2))
        if not isinstance(self.nr_of_bins, int):
            self.nr_of_bins = int(self.nr_of_bins)
        assert isinstance(self.nr_of_bins, int)
        self.bins_tensor: Tensor = (tensor(range(self.nr_of_bins)) / (self.nr_of_bins - 1)) * self.width + self.lower
        self.bins_list = self.bins_tensor.tolist()

    def transform_to(self, value):
        if any(isinstance(value, t) for t in (float, int)):
            return max(0, min(self.nr_of_bins - 1, round(self.transform_to_unconstrained(value))))
        elif isinstance(value, Tensor):
            return torch.maximum(tensor(0), torch.minimum(tensor(self.nr_of_bins - 1),
                                                          torch.round(self.transform_to_unconstrained(value))))
        raise TypeError(f"Cannot transform type {type(value).__name__} of value {value} to bin")

    def transform_to_unconstrained(self, value):
        return ((value - self.lower) / self.width) * (self.nr_of_bins - 1)

    def transform_from(self, bin):
        if any(isinstance(bin, t) for t in (float, int)):
            assert 0 <= bin < self.nr_of_bins
            return self.bins_list[bin]
        elif isinstance(bin, Tensor):
            return self.bins_tensor[bin]

    def __repr__(self):
        return f"[{self.lower},..,{self.upper}]@{self.nr_of_bins}"

    def __str__(self):
        return str(self.bins_list)
