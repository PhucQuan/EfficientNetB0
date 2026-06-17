import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from utils.dataset_loader import create_eval_generator
from utils.dataset_loader import get_class_names_from_generator
from utils.dataset_loader import validate_directory
from utils.dataset_paths import resolve_test_dir
from utils.metrics import save_confusion_matrix_plot
from utils.transforms import IMAGE_SIZE


def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate the EfficientNetB0 fruit classifier."
    )
    parser.add_argument(
        "--dataset-root",
        help="Path containing train/validation/test subfolders."
    )
    parser.add_argument(
        "--test-dir",
        default="FruitDataset/test"
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/fruit8_efficientnet_group_regularized.keras"
    )
    parser.add_argument(
        "--class-names",
        default="artifacts/class_names.json"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE
    )

    return parser.parse_args()


def save_csv_report(report_dict, output_path):

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])

        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                writer.writerow([
                    label,
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("f1-score"),
                    metrics.get("support")
                ])


def save_confusion_matrix_csv(matrix, class_names, output_path):

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *class_names])

        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def main():

    args = parse_args()

    test_dir = resolve_test_dir(
        dataset_root=args.dataset_root,
        test_dir=args.test_dir
    )
    model_path = Path(args.model_path)
    class_names_path = Path(args.class_names)
    output_dir = Path(args.output_dir)

    validate_directory(test_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    test_generator = create_eval_generator(
        test_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False
    )

    class_names = json.loads(
        class_names_path.read_text(encoding="utf-8")
    )
    generator_class_names = get_class_names_from_generator(test_generator)

    if class_names != generator_class_names:
        raise ValueError(
            "The provided class_names.json does not match the test directory labels."
        )

    model = keras.models.load_model(model_path)

    test_generator.reset()
    test_loss, test_accuracy = model.evaluate(
        test_generator,
        verbose=1
    )
    test_generator.reset()
    probabilities = model.predict(
        test_generator,
        verbose=1
    )

    y_true = test_generator.classes
    y_pred = np.argmax(probabilities, axis=1)
    top3_accuracy = float(
        np.mean([
            true_label in top3
            for true_label, top3 in zip(
                y_true,
                np.argsort(probabilities, axis=1)[:, -3:]
            )
        ])
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "top3_accuracy": top3_accuracy,
        "num_test_images": int(test_generator.samples),
        "test_dir": str(test_dir),
        "model_path": str(model_path)
    }

    summary_path = output_dir / "evaluation_summary.json"
    report_json_path = output_dir / "classification_report.json"
    report_csv_path = output_dir / "classification_report.csv"
    confusion_csv_path = output_dir / "confusion_matrix.csv"
    confusion_png_path = output_dir / "confusion_matrix.png"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )
    report_json_path.write_text(
        json.dumps(report_dict, indent=2),
        encoding="utf-8"
    )
    save_csv_report(report_dict, report_csv_path)
    save_confusion_matrix_csv(matrix, class_names, confusion_csv_path)
    save_confusion_matrix_plot(
        matrix,
        class_names,
        confusion_png_path
    )

    print("Evaluation finished.")
    print(json.dumps(summary, indent=2))
    print(f"Classification report: {report_json_path}")
    print(f"Confusion matrix: {confusion_png_path}")


if __name__ == "__main__":
    main()
