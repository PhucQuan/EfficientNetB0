from pathlib import Path


SPLIT_NAMES = ("train", "validation", "test")
MAX_SEARCH_DEPTH = 4


def find_dataset_root(path):

    path = Path(path)

    if not path.exists():
        fallback_candidate = _find_fallback_dataset_root(path)

        if fallback_candidate is not None:
            return fallback_candidate

        raise FileNotFoundError(f"Dataset path not found: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Expected a dataset directory: {path}")

    if _has_expected_splits(path):
        return path

    for child in _iter_subdirectories(path, max_depth=MAX_SEARCH_DEPTH):
        if _has_expected_splits(child):
            return child

    raise ValueError(
        f"Could not find train/validation/test inside: {path}"
    )


def resolve_train_val_dirs(dataset_root=None, train_dir=None, val_dir=None):

    if dataset_root:
        root = find_dataset_root(dataset_root)
        return root / "train", root / "validation"

    if not train_dir or not val_dir:
        raise ValueError(
            "Provide either --dataset-root or both --train-dir and --val-dir."
        )

    return Path(train_dir), Path(val_dir)


def resolve_test_dir(dataset_root=None, test_dir=None):

    if dataset_root:
        root = find_dataset_root(dataset_root)
        return root / "test"

    if not test_dir:
        raise ValueError(
            "Provide either --dataset-root or --test-dir."
        )

    return Path(test_dir)


def _has_expected_splits(path):

    return all((path / split_name).is_dir() for split_name in SPLIT_NAMES)


def _iter_subdirectories(path, max_depth):

    path = Path(path)

    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue

        yield child

        if max_depth > 1:
            yield from _iter_subdirectories(child, max_depth - 1)


def _find_fallback_dataset_root(original_path):

    original_path = Path(original_path)

    kaggle_input_root = Path("/kaggle/input")

    if not str(original_path).startswith("/kaggle/input"):
        return None

    if not kaggle_input_root.exists():
        return None

    exact_name_matches = []
    split_matches = []

    for candidate in _iter_subdirectories(
            kaggle_input_root,
            max_depth=MAX_SEARCH_DEPTH + 2):
        if candidate.name == original_path.name:
            exact_name_matches.append(candidate)

        if _has_expected_splits(candidate):
            split_matches.append(candidate)

    for candidate in exact_name_matches:
        if _has_expected_splits(candidate):
            return candidate

        nested_match = _find_nested_split_root(candidate)
        if nested_match is not None:
            return nested_match

    for candidate in split_matches:
        return candidate

    return None


def _find_nested_split_root(path):

    path = Path(path)

    if _has_expected_splits(path):
        return path

    for candidate in _iter_subdirectories(path, max_depth=MAX_SEARCH_DEPTH):
        if _has_expected_splits(candidate):
            return candidate

    return None
