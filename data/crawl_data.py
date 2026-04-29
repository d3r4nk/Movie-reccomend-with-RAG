
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub


DATASET_ID = "parthdande/imdb-dataset-2024-updated"
RAW_FILENAMES = ("IMDb_Dataset.csv", "IMDb_Dataset_2.csv", "IMDb_Dataset_3.csv")


def crawl_dataset(output_dir: str | Path = "data/raw") -> Path:
    """Download the Kaggle dataset and copy the expected CSV files locally."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
    copied = []

    for filename in RAW_FILENAMES:
        source = dataset_path / filename
        if not source.exists():
            print(f"Warning: {source} not found. Skipping.")
            continue

        destination = output_path / filename
        shutil.copy2(source, destination)
        copied.append(destination)
        print(f"Copied {filename} -> {destination}")

    if not copied:
        raise FileNotFoundError(
            f"No expected CSV files were found after downloading {DATASET_ID}."
        )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download IMDb Kaggle CSV files.")
    parser.add_argument("--output-dir", default="data/raw", help="Directory for raw CSVs.")
    args = parser.parse_args()

    crawl_dataset(args.output_dir)


if __name__ == "__main__":
    main()
