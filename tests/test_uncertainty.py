from pathlib import Path

import pytest
import torch

from src.utils.uncertainty import (
    AleatoricUncertainty,
    EpistemicUncertainty,
    TotalUncertainty,
)


@pytest.mark.parametrize("mcd_samples, batch_size, num_classes", [(10, 128, 10)])
def test_total_uncertainty(mcd_samples: int, batch_size: int, num_classes: int) -> None:
    """Test TotalUncertainty with synthetic softmaxed MC Dropout predictions."""
    T = mcd_samples
    B = batch_size
    C = num_classes

    # Simulate softmax probabilities
    logits = torch.randn(T, B, C)  # suppose these were stacked after T forward passes
    probs = torch.softmax(logits, dim=-1)

    metric = TotalUncertainty()
    metric.update(probs)
    result = metric.compute()

    # Basic checks
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0
    assert result >= 0, "Entropy must be non-negative"
    assert result <= torch.log(torch.tensor(float(C))) + 1e-3, "Max entropy is log(C)"


@pytest.mark.parametrize("mcd_samples, batch_size, num_classes", [(10, 128, 10)])
def test_aleatoric_uncertainty(mcd_samples: int, batch_size: int, num_classes: int) -> None:
    """Test AleatoricUncertainty with synthetic softmaxed MC Dropout predictions."""
    T = mcd_samples
    B = batch_size
    C = num_classes

    # Simulate softmax probabilities
    logits = torch.randn(T, B, C)  # suppose these were stacked after T forward passes
    probs = torch.softmax(logits, dim=-1)

    metric = AleatoricUncertainty()
    metric.update(probs)
    result = metric.compute()

    # Basic checks
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0
    assert result >= 0, "Entropy must be non-negative"
    assert result <= torch.log(torch.tensor(float(C))) + 1e-3, "Max entropy is log(C)"


@pytest.mark.parametrize("mcd_samples, batch_size, num_classes", [(10, 128, 10)])
def test_epistemic_uncertainty(mcd_samples: int, batch_size: int, num_classes: int) -> None:
    """Test AleatoricUncertainty with synthetic softmaxed MC Dropout predictions."""
    T = mcd_samples
    B = batch_size
    C = num_classes

    # Simulate softmax probabilities
    logits = torch.randn(T, B, C)  # suppose these were stacked after T forward passes
    probs = torch.softmax(logits, dim=-1)

    tu_metric = TotalUncertainty()
    au_metric = AleatoricUncertainty()
    eu_metric = EpistemicUncertainty(tu_metric, au_metric)
    tu_metric.update(probs)
    au_metric.update(probs)
    tu_result = tu_metric.compute()
    au_result = au_metric.compute()
    eu_result = eu_metric.compute()

    # Basic checks
    assert isinstance(eu_result, torch.Tensor)
    assert eu_result.ndim == 0
    assert eu_result >= 0, "EU must be non-negative"
    assert eu_result == tu_result - au_result, "EU is not matching the definition TU = AU + EU"
