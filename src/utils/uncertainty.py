import torch
from torch import Tensor
from torchmetrics import Metric


class TotalUncertainty(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("entropy_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mcd_preds: Tensor, probs: bool) -> None:
        """
        :param mcd_preds: Tensor of shape [T, B, C], probabilities from MC Dropout
        """
        if mcd_preds.ndim != 3:
            raise ValueError("Expected mc_preds to have shape [T, B, C]")

        if probs is False:
            # Convert logits to softmax probabilities
            mcd_preds = torch.softmax(mcd_preds, dim=-1)

        # Mean predictive distribution over MC samples
        mean_probs = mcd_preds.mean(dim=0)  # [B, C]

        # Predictive entropy
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)  # [B]
        batch_entropy = entropy.mean()  # scalar

        self.entropy_sum += batch_entropy
        self.n_batches += 1

    def compute(self) -> Tensor:
        return self.entropy_sum / self.n_batches


class AleatoricUncertainty(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("entropy_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mcd_preds: Tensor, probs: bool) -> None:
        """
        :param mcd_preds: Tensor of shape [T, B, C], probabilities from MC Dropout
        """
        if mcd_preds.ndim != 3:
            raise ValueError("Expected mc_preds to have shape [T, B, C]")

        if probs is False:
            # Convert logits to softmax probabilities
            mcd_preds = torch.softmax(mcd_preds, dim=-1)  # [T,B,C]

        entropies = -torch.sum(mcd_preds * torch.log(mcd_preds + 1e-10), dim=-1)  # [T,B]
        batch_entropy = torch.mean(entropies, axis=0).mean()  # scalar

        self.entropy_sum += batch_entropy
        self.n_batches += 1

    def compute(self) -> Tensor:
        return self.entropy_sum / self.n_batches


class EpistemicUncertainty(Metric):
    def __init__(self, total_metric: Metric, aleatoric_metric: Metric, **kwargs):
        super().__init__(**kwargs)
        self.total_metric = total_metric
        self.aleatoric_metric = aleatoric_metric

    def update(self, mcd_preds: Tensor, probs: bool) -> None:
        self.total_metric.update(mcd_preds, probs=probs)
        self.aleatoric_metric.update(mcd_preds, probs=probs)

    def compute(self) -> Tensor:
        total = self.total_metric.compute()
        aleatoric = self.aleatoric_metric.compute()
        return total - aleatoric
