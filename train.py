import argparse
import json
from pathlib import Path

from tensorflow import keras

from models.efficientnet_model import create_model
from utils.dataset_loader import create_eval_generator
from utils.dataset_loader import create_train_generator
from utils.dataset_loader import get_class_names_from_generator
from utils.dataset_loader import validate_directory
from utils.dataset_paths import resolve_train_val_dirs
from utils.metrics import save_training_curves
from utils.transforms import IMAGE_SIZE


DEFAULT_MODEL_NAME = "fruit8_efficientnet_group_regularized.keras"


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train EfficientNetB0 with the thesis configuration."
    )
    parser.add_argument(
        "--dataset-root",
        help="Path containing train/validation/test subfolders."
    )
    parser.add_argument(
        "--train-dir",
        default="FruitDataset/train"
    )
    parser.add_argument(
        "--val-dir",
        default="FruitDataset/validation"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10
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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.55
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3
    )
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=2
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--weights",
        default="imagenet",
        help="Use 'imagenet' or 'none'."
    )
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="Unfreeze EfficientNetB0 during training."
    )

    return parser.parse_args()


def normalize_weights_arg(weights_arg):

    if weights_arg.lower() == "none":
        return None

    return weights_arg


def main():

    args = parse_args()

    train_dir, val_dir = resolve_train_val_dirs(
        dataset_root=args.dataset_root,
        train_dir=args.train_dir,
        val_dir=args.val_dir
    )
    output_dir = Path(args.output_dir)

    validate_directory(train_dir)
    validate_directory(val_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keras.utils.set_random_seed(args.seed)

    train_generator = create_train_generator(
        train_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed
    )
    val_generator = create_eval_generator(
        val_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False
    )

    class_names = get_class_names_from_generator(train_generator)
    val_class_names = get_class_names_from_generator(val_generator)

    if class_names != val_class_names:
        raise ValueError(
            "Train and validation directories must contain the same classes."
        )

    num_classes = len(class_names)

    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    if args.dataset_root:
        print(f"Dataset root: {Path(args.dataset_root)}")
    print(f"Training images: {train_generator.samples}")
    print(f"Validation images: {val_generator.samples}")

    if num_classes != 8:
        print(
            "Warning: the thesis configuration expects 8 classes, "
            f"but found {num_classes}."
        )

    class_names_path = output_dir / "class_names.json"
    class_names_path.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    model_path = output_dir / DEFAULT_MODEL_NAME
    final_model_path = output_dir / "fruit8_efficientnet_group_regularized_last.keras"
    training_plot_path = output_dir / "training_curves.png"
    training_summary_path = output_dir / "training_summary.json"
    training_log_path = output_dir / "training_log.csv"

    try:
        model = create_model(
            num_classes=num_classes,
            image_size=args.image_size,
            dropout_rate=args.dropout,
            l2_value=args.l2,
            weights=normalize_weights_arg(args.weights),
            train_backbone=args.train_backbone
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not build EfficientNetB0 with the requested weights. "
            "If Kaggle Internet is disabled, try '--weights none' or "
            "provide cached weights."
        ) from exc

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=args.learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=args.reduce_lr_patience,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=str(training_log_path)
        )
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    model.save(final_model_path)
    save_training_curves(
        history.history,
        training_plot_path
    )

    best_epoch = min(
        range(len(history.history["val_loss"])),
        key=history.history["val_loss"].__getitem__
    )
    training_summary = {
        "model_name": DEFAULT_MODEL_NAME,
        "best_epoch": best_epoch + 1,
        "best_val_loss": float(history.history["val_loss"][best_epoch]),
        "best_val_accuracy": float(history.history["val_accuracy"][best_epoch]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "dropout": args.dropout,
        "l2": args.l2,
        "learning_rate": args.learning_rate,
        "weights": args.weights,
        "train_backbone": args.train_backbone
    }
    training_summary_path.write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8"
    )

    print("\nTraining finished.")
    print(f"Best model: {model_path}")
    print(f"Last model: {final_model_path}")
    print(f"Class names: {class_names_path}")
    print(f"Training curves: {training_plot_path}")
    print(f"Summary: {training_summary_path}")


if __name__ == "__main__":
    main()
