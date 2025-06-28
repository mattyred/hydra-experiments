from pathlib import Path

import pytest
import torch

from src.data.cifar10_datamodule import CIFAR10DataModule


@pytest.mark.parametrize("batch_size, train_subset", [(32, 5), (128, 10)])
def test_mnist_datamodule(batch_size: int, train_subset: float) -> None:
    """Tests `CIFAR10DataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    :param train_subset: Percentage of samples to be sampled from the training set
    """
    data_dir = "data/"

    dm = CIFAR10DataModule(data_dir=data_dir, batch_size=batch_size, train_subset=train_subset)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "cifar-10-batches-py").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    # num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    # assert num_datapoints == 60_000

    num_tr_datapoints = len(dm.data_train)
    assert num_tr_datapoints == (train_subset / 100) * 50_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
