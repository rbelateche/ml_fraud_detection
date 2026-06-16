"""CLI: prepare the dataset (``fraud-data``).

Usage:
    fraud-data                 # build/refresh the default backend dataset
    fraud-data --force         # ignore cache and regenerate
    fraud-data --source kaggle # use the Kaggle backend
"""

from __future__ import annotations

import argparse

from fraud_detection.data.loader import load_dataset
from fraud_detection.data.split import time_based_split
from fraud_detection.logging_utils import get_logger

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the fraud detection dataset.")
    parser.add_argument("--source", default=None, help="synthetic | kaggle")
    parser.add_argument("--force", action="store_true", help="ignore cache and rebuild")
    args = parser.parse_args(argv)

    df = load_dataset(source=args.source, force=args.force)
    split = time_based_split(df)

    log.info("Dataset ready: %d rows, %d columns", len(df), df.shape[1])
    print("\nTime-based split summary:")
    print(split.summary().to_string(index=False))


if __name__ == "__main__":
    main()
