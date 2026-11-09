"""Read the released dataset manifests and five-fold partitions."""
import json
from typing import Dict, List, Tuple


def load_splits(splits_json: str, fold: int) -> Tuple[List[str], List[str]]:
    """Load train and val sample ids for a given fold.

    Args:
        splits_json: Path to splits.json.
        fold: Fold index (0-based).

    Returns:
        (train_ids, val_ids)
    """
    with open(splits_json) as f:
        splits = json.load(f)

    if fold >= splits["n_folds"]:
        raise ValueError(
            f"Requested fold {fold} but splits.json only has {splits['n_folds']} folds."
        )

    entry = splits["folds"][fold]
    return entry["train"], entry["val"]


def load_manifest(manifest_json: str) -> dict:
    """Load and return manifest dict."""
    with open(manifest_json) as f:
        return json.load(f)


def get_sample_dict(manifest: dict) -> Dict[str, dict]:
    """Return a mapping from sample id to sample entry dict."""
    return {s["id"]: s for s in manifest["samples"]}
