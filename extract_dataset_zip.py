import argparse
from pathlib import Path
from zipfile import ZipFile

from utils.dataset_paths import find_dataset_root


def parse_args():

    parser = argparse.ArgumentParser(
        description="Extract a dataset zip and locate the train/validation/test root."
    )
    parser.add_argument(
        "--zip-path",
        required=True,
        help="Path to the dataset zip file."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where the zip contents will be extracted."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow extraction into a non-empty output directory."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --force or choose another folder."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    dataset_root = find_dataset_root(output_dir)

    print("Dataset extraction finished.")
    print(f"Zip file: {zip_path}")
    print(f"Extracted to: {output_dir}")
    print(f"Dataset root: {dataset_root}")
    print(f"Train dir: {dataset_root / 'train'}")
    print(f"Validation dir: {dataset_root / 'validation'}")
    print(f"Test dir: {dataset_root / 'test'}")


if __name__ == "__main__":
    main()
