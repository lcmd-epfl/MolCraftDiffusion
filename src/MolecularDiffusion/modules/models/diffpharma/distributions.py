"""2D (n_ligand x n_pocket) size prior for DiffPharma.

Port of ``DistributionNodes`` from
``others/DiffPharma/equivariant_diffusion/conditional_model.py``.

One deliberate deviation: upstream ``sample(n)`` returns a *pair*
``(n_lig, n_pocket)``, which does not satisfy the platform's
``node_dist_model.sample(n) -> LongTensor of atom counts`` contract. Here
``sample`` returns only the ligand component (i.e. the marginal over pocket
size) and the joint draw is available as :meth:`sample_joint`. Generation
uses :meth:`sample_conditional`, which is what DiffPharma itself calls.
"""

import numpy as np
import torch


class DistributionNodes:
    def __init__(self, histogram):
        histogram = torch.tensor(np.asarray(histogram)).float()
        histogram = histogram + 1e-3  # numerical stability

        prob = histogram / histogram.sum()

        self.idx_to_n_nodes = torch.tensor(
            [[(i, j) for j in range(prob.shape[1])] for i in range(prob.shape[0])]
        ).view(-1, 2)
        self.n_nodes_to_idx = {
            tuple(x.tolist()): i for i, x in enumerate(self.idx_to_n_nodes)
        }
        self.prob = prob
        self.m = torch.distributions.Categorical(prob.view(-1), validate_args=True)
        self.n1_given_n2 = [
            torch.distributions.Categorical(prob[:, j], validate_args=True)
            for j in range(prob.shape[1])
        ]
        self.n2_given_n1 = [
            torch.distributions.Categorical(prob[i, :], validate_args=True)
            for i in range(prob.shape[0])
        ]

    # -- contract ------------------------------------------------------- #
    def sample(self, n_samples=1):
        """Ligand sizes only (pocket size marginalised out)."""
        return self.sample_joint(n_samples)[0]

    @property
    def n_node_dist(self):
        """``{n_ligand_atoms: count}`` -- the ligand-size marginal."""
        marginal = self.prob.sum(dim=1)
        return {int(i): float(v) for i, v in enumerate(marginal)}

    # -- upstream API --------------------------------------------------- #
    def sample_joint(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        num_nodes_lig, num_nodes_pocket = self.idx_to_n_nodes[idx].T
        return num_nodes_lig, num_nodes_pocket

    def sample_conditional(self, n1=None, n2=None):
        assert (n1 is None) ^ (n2 is None), "Exactly one input argument must be None"
        m = self.n1_given_n2 if n2 is not None else self.n2_given_n1
        c = n2 if n2 is not None else n1
        return torch.tensor([m[i].sample() for i in c], device=c.device)

    def log_prob(self, batch_n_nodes_1, batch_n_nodes_2):
        idx = torch.tensor(
            [
                self.n_nodes_to_idx[(n1, n2)]
                for n1, n2 in zip(
                    batch_n_nodes_1.tolist(), batch_n_nodes_2.tolist()
                )
            ]
        )
        return self.m.log_prob(idx).to(batch_n_nodes_1.device)

    def log_prob_n1_given_n2(self, n1, n2):
        log_probs = torch.stack(
            [self.n1_given_n2[c].log_prob(i.cpu()) for i, c in zip(n1, n2)]
        )
        return log_probs.to(n1.device)

    def log_prob_n2_given_n1(self, n2, n1):
        log_probs = torch.stack(
            [self.n2_given_n1[c].log_prob(i.cpu()) for i, c in zip(n2, n1)]
        )
        return log_probs.to(n2.device)
