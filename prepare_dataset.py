import argparse
import json
from pathlib import Path

from utils.data_preparation import assign_groups_to_splits
from utils.data_preparation import attach_average_hash
from utils.data_preparation import build_hash_groups
from utils.data_preparation import collect_records_from_config
from utils.data_preparation import copy_split_dataset
from utils.data_preparation import deduplicate_exact_matches
from utils.data_preparation import ensure_output_dir_is_clean
from utils.data_preparation import inspect_split_leakage
from utils.data_preparation import load_sources_config
from utils.data_preparation import save_manifest_csv
from utils.data_preparation import summarize_records


def parse_args():

    parser = argparse.ArgumentParser(
        description="Prepare a merged fruit dataset with duplicate-aware group split."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the dataset config JSON."
    )
    parser.add_argument(
        "--output-dir",
        default="prepared_dataset",
        help="Folder that will contain train/validation/test."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


def main():

    args = parse_args()

    config = load_sources_config(args.config)
    output_dir = Path(args.output_dir)
    ensure_output_dir_is_clean(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = collect_records_from_config(config)
    kept_records, removed_duplicates, removed_conflicts = deduplicate_exact_matches(
        raw_records
    )
    attach_average_hash(kept_records)

    grouped_records = build_hash_groups(kept_records)
    split_records = assign_groups_to_splits(
        grouped_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    manifest_rows = copy_split_dataset(
        split_records,
        output_dir
    )
    manifest_path = output_dir / "dataset_manifest.csv"
    save_manifest_csv(manifest_rows, manifest_path)

    leakage_report = inspect_split_leakage(output_dir)
    prepare_report = {
        "raw_image_count": len(raw_records),
        "exact_duplicate_removed_count": len(removed_duplicates),
        "cross_label_conflict_removed_count": len(removed_conflicts),
        "kept_image_count": len(kept_records),
        "raw_summary": summarize_records(raw_records),
        "kept_summary": summarize_records(kept_records),
        "split_counts": {
            split_name: len(records)
            for split_name, records in split_records.items()
        },
        "leakage_report": leakage_report
    }
    report_path = output_dir / "prepare_report.json"
    report_path.write_text(
        json.dumps(prepare_report, indent=2),
        encoding="utf-8"
    )

    print("Dataset preparation finished.")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    print(json.dumps(prepare_report["split_counts"], indent=2))
    print(
        "Leakage summary: "
        f"md5={leakage_report['md5']['overlap_group_count']}, "
        f"average_hash={leakage_report['average_hash']['overlap_group_count']}"
    )


if __name__ == "__main__":
    main()
