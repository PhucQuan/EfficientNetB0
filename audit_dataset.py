import argparse
import json
from pathlib import Path

from utils.dataset_paths import find_dataset_root
from utils.data_preparation import inspect_split_leakage


def parse_args():

    parser = argparse.ArgumentParser(
        description="Audit train/validation/test splits for possible leakage."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Prepared dataset root containing split folders."
    )
    parser.add_argument(
        "--output-json",
        default="dataset_audit_report.json"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    dataset_dir = find_dataset_root(Path(args.dataset_dir))
    output_json = Path(args.output_json)

    report = inspect_split_leakage(dataset_dir)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("Dataset audit finished.")
    print(f"Report: {output_json}")
    print(
        "Overlap groups: "
        f"md5={report['md5']['overlap_group_count']}, "
        f"average_hash={report['average_hash']['overlap_group_count']}"
    )


if __name__ == "__main__":
    main()
