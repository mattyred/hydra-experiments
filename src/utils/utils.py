import warnings
from collections import defaultdict
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Subset

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def get_data_subset(
    train_dataset: torch.utils.data.Dataset, train_subset: float, num_classes: int, seed: int
) -> torch.utils.data.Subset:
    """
    Uniformly sample an absolute number of training points (n_subset)
    across classes from train_dataset.
    """
    print(
        f"Creating training subset of {train_subset} samples from dataset of size {len(train_dataset)}"
    )
    if train_subset > len(train_dataset):
        raise ValueError(
            f"Requested train_subset={train_subset}, but dataset only has {len(train_dataset)} samples."
        )

    targets = np.array(train_dataset.targets)
    class_indices = defaultdict(list)  # for each label, the indices of samples with that label

    # Collect indices per class
    for idx, label in enumerate(targets):
        class_indices[int(label)].append(idx)

    # Determine how many samples to take per class
    samples_per_class = train_subset // num_classes
    remainder = train_subset % num_classes  # leftover samples after uniform split

    if samples_per_class == 0:
        raise ValueError(
            f"train_subset={train_subset} is too small to allocate at least one sample per class "
            f"for num_classes={num_classes}."
        )

    rng = np.random.default_rng(seed=seed)
    subset_indices = []

    # Uniform baseline: same number from each class
    for c in range(num_classes):
        cls_indices = class_indices[c]
        if len(cls_indices) < samples_per_class:
            raise ValueError(
                f"Not enough samples in class {c} to satisfy uniform sampling "
                f"({len(cls_indices)} available, {samples_per_class} requested)."
            )
        subset_indices.extend(rng.choice(cls_indices, samples_per_class, replace=False))

    # Distribute any remainder samples, one extra at a time to random classes
    if remainder > 0:
        # choose which classes get one extra sample
        extra_classes = rng.choice(np.arange(num_classes), size=remainder, replace=False)
        for c in extra_classes:
            cls_indices = class_indices[c]
            # avoid picking an index already chosen
            already_chosen = {idx for idx in subset_indices if targets[idx] == c}
            available = [idx for idx in cls_indices if idx not in already_chosen]
            if not available:
                # If a class is exhausted, just skip; we end up with slightly < n_subset
                continue
            subset_indices.append(rng.choice(available, 1, replace=False)[0])

    # Shuffle final subset indices to avoid class ordering
    subset_indices = rng.permutation(subset_indices)

    return Subset(train_dataset, subset_indices)
