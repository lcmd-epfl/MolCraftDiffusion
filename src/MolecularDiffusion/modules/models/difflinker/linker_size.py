"""Linker-size distribution, ported verbatim from DiffLinker's
``src/linker_size.py``. ``SizeGNN`` (a separate auxiliary linker-size
predictor) is explicitly out of scope for this integration — see
docs/model_integrations/difflinker/INTEGRATION_PLAN.md — so only
``DistributionNodes`` is ported here.
"""

import numpy as np
import torch
from torch.distributions.categorical import Categorical


class DistributionNodes:
    """Samples/scores discrete sizes (here: linker sizes) from a histogram."""

    def __init__(self, histogram: dict):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()
        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes: torch.Tensor) -> torch.Tensor:
        assert len(batch_n_nodes.size()) == 1
        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)
        log_p = torch.log(self.prob + 1e-30).to(batch_n_nodes.device)
        return log_p[idcs]
