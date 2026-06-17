import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def validate_ratio_sum(train_ratio, val_ratio, test_ratio):

    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        )


def validate_image_directory(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {path}")


def ensure_output_dir_is_clean(path):

    path = Path(path)

    if not path.exists():
        return

    if any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. "
            "Use a new folder or delete the old prepared dataset first."
        )


def iter_image_files(path):

    path = Path(path)

    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            yield file_path


def compute_md5(file_path):

    hasher = hashlib.md5()

    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def average_hash(file_path, hash_size=8):

    with Image.open(file_path) as image:
        image = image.convert("L")
        image = image.resize((hash_size, hash_size))
        pixels = np.asarray(image, dtype=np.float32)

    average_value = pixels.mean()
    bits = pixels > average_value

    return "".join("1" if bit else "0" for bit in bits.flatten())


def load_sources_config(config_path):

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    if "sources" not in config or not config["sources"]:
        raise ValueError("Config must contain a non-empty 'sources' list.")

    return config


def collect_records_from_config(config):

    records = []

    for source in config["sources"]:
        source_name = source["name"]
        source_root = Path(source["root"])
        class_map = source["class_map"]

        validate_image_directory(source_root)

        for source_label, target_label in class_map.items():
            label_root = source_root / source_label

            if not label_root.exists():
                print(
                    f"Warning: source folder not found for "
                    f"{source_name}/{source_label}: {label_root}"
                )
                continue

            for image_path in iter_image_files(label_root):
                records.append(
                    {
                        "source_name": source_name,
                        "source_label": source_label,
                        "target_label": target_label,
                        "path": image_path
                    }
                )

    if not records:
        raise ValueError("No images were collected from the provided config.")

    return records


def deduplicate_exact_matches(records):

    md5_groups = defaultdict(list)

    for record in records:
        record["md5"] = compute_md5(record["path"])
        md5_groups[record["md5"]].append(record)

    kept_records = []
    removed_duplicates = []
    removed_conflicts = []

    for _, group in md5_groups.items():
        group = sorted(
            group,
            key=lambda item: (
                item["target_label"],
                item["source_name"],
                str(item["path"])
            )
        )
        labels = {item["target_label"] for item in group}

        if len(labels) > 1:
            removed_conflicts.extend(group)
            continue

        kept_records.append(group[0])
        removed_duplicates.extend(group[1:])

    return kept_records, removed_duplicates, removed_conflicts


def attach_average_hash(records):

    for record in records:
        record["average_hash"] = average_hash(record["path"])

    return records


def build_hash_groups(records):

    grouped_by_label = defaultdict(lambda: defaultdict(list))

    for record in records:
        grouped_by_label[record["target_label"]][record["average_hash"]].append(record)

    groups = {}

    for target_label, hash_groups in grouped_by_label.items():
        groups[target_label] = []

        for index, (_, members) in enumerate(sorted(hash_groups.items())):
            group_id = f"{target_label}_group_{index:05d}"

            for member in members:
                member["group_id"] = group_id

            groups[target_label].append(members)

    return groups


def choose_split(current_counts, desired_counts):

    ranked_splits = sorted(
        desired_counts,
        key=lambda split_name: (
            current_counts[split_name] / max(desired_counts[split_name], 1e-9),
            current_counts[split_name]
        )
    )

    return ranked_splits[0]


def assign_groups_to_splits(
        grouped_records,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42):

    validate_ratio_sum(train_ratio, val_ratio, test_ratio)
    rng = random.Random(seed)

    assigned = {
        "train": [],
        "validation": [],
        "test": []
    }

    for target_label, groups in grouped_records.items():
        groups = list(groups)
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)

        total_images = sum(len(group) for group in groups)
        desired_counts = {
            "train": total_images * train_ratio,
            "validation": total_images * val_ratio,
            "test": total_images * test_ratio
        }
        current_counts = {
            "train": 0,
            "validation": 0,
            "test": 0
        }

        for group in groups:
            split_name = choose_split(current_counts, desired_counts)
            assigned[split_name].extend(group)
            current_counts[split_name] += len(group)

    return assigned


def build_output_filename(record, index):

    extension = record["path"].suffix.lower()
    prefix = f"{record['source_name']}_{record['target_label']}"

    return f"{prefix}_{index:05d}{extension}"


def copy_split_dataset(records_by_split, output_dir):

    output_dir = Path(output_dir)
    manifest_rows = []

    for split_name, records in records_by_split.items():
        counters = defaultdict(int)

        for record in sorted(
                records,
                key=lambda item: (
                    item["target_label"],
                    item["source_name"],
                    str(item["path"])
                )):
            target_label = record["target_label"]
            destination_dir = output_dir / split_name / target_label
            destination_dir.mkdir(parents=True, exist_ok=True)

            file_index = counters[target_label]
            counters[target_label] += 1

            destination_path = destination_dir / build_output_filename(
                record,
                file_index
            )
            shutil.copy2(record["path"], destination_path)

            manifest_rows.append(
                {
                    "split": split_name,
                    "target_label": target_label,
                    "source_name": record["source_name"],
                    "source_label": record["source_label"],
                    "group_id": record["group_id"],
                    "md5": record["md5"],
                    "average_hash": record["average_hash"],
                    "original_path": str(record["path"]),
                    "copied_path": str(destination_path)
                }
            )

    return manifest_rows


def save_manifest_csv(rows, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    columns = [
        "split",
        "target_label",
        "source_name",
        "source_label",
        "group_id",
        "md5",
        "average_hash",
        "original_path",
        "copied_path"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_records(records):

    summary = defaultdict(int)
    class_summary = defaultdict(int)

    for record in records:
        summary[record["source_name"]] += 1
        class_summary[record["target_label"]] += 1

    return {
        "by_source": dict(sorted(summary.items())),
        "by_class": dict(sorted(class_summary.items()))
    }


def inspect_split_leakage(dataset_dir):

    dataset_dir = Path(dataset_dir)
    split_names = [
        split_name
        for split_name in ["train", "validation", "test"]
        if (dataset_dir / split_name).is_dir()
    ]

    if not split_names:
        raise ValueError(
            f"No split folders found in {dataset_dir}. "
            "Expected at least one of: train, validation, test."
        )

    records = []

    for split_name in split_names:
        split_dir = dataset_dir / split_name

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for image_path in iter_image_files(class_dir):
                records.append(
                    {
                        "split": split_name,
                        "label": class_dir.name,
                        "path": image_path,
                        "md5": compute_md5(image_path),
                        "average_hash": average_hash(image_path)
                    }
                )

    grouped_stats = {}

    for key_name in ["md5", "average_hash"]:
        key_groups = defaultdict(list)

        for record in records:
            key_groups[record[key_name]].append(record)

        overlap_groups = []
        cross_label_conflicts = []
        pair_counts = defaultdict(int)

        for group in key_groups.values():
            involved_splits = sorted({item["split"] for item in group})
            involved_labels = sorted({item["label"] for item in group})

            if len(involved_labels) > 1:
                cross_label_conflicts.append(
                    {
                        "labels": involved_labels,
                        "count": len(group),
                        "paths": [str(item["path"]) for item in group[:10]]
                    }
                )

            if len(involved_splits) > 1:
                overlap_groups.append(
                    {
                        "labels": involved_labels,
                        "splits": involved_splits,
                        "count": len(group),
                        "paths": [str(item["path"]) for item in group[:10]]
                    }
                )

                for index, left_split in enumerate(involved_splits):
                    for right_split in involved_splits[index + 1:]:
                        pair_counts[f"{left_split}__{right_split}"] += 1

        grouped_stats[key_name] = {
            "overlap_group_count": len(overlap_groups),
            "cross_label_conflict_count": len(cross_label_conflicts),
            "pair_overlap_counts": dict(sorted(pair_counts.items())),
            "sample_overlap_groups": overlap_groups[:20],
            "sample_cross_label_conflicts": cross_label_conflicts[:20]
        }

    counts_by_split = defaultdict(int)
    counts_by_split_and_class = defaultdict(lambda: defaultdict(int))

    for record in records:
        counts_by_split[record["split"]] += 1
        counts_by_split_and_class[record["split"]][record["label"]] += 1

    return {
        "total_images": len(records),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "counts_by_split_and_class": {
            split_name: dict(sorted(class_counts.items()))
            for split_name, class_counts in sorted(counts_by_split_and_class.items())
        },
        "md5": grouped_stats["md5"],
        "average_hash": grouped_stats["average_hash"]
    }
