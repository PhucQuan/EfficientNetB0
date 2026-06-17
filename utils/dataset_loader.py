from pathlib import Path

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def validate_directory(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {path}")

    class_directories = [item for item in path.iterdir() if item.is_dir()]

    if not class_directories:
        raise ValueError(
            f"No class subdirectories were found inside: {path}"
        )


def create_train_generator(
        path,
        image_size=224,
        batch_size=32,
        seed=42):

    generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        shear_range=10,
        horizontal_flip=True,
        brightness_range=(0.80, 1.20),
        fill_mode="nearest"
    )

    return generator.flow_from_directory(
        directory=str(path),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed
    )


def create_eval_generator(
        path,
        image_size=224,
        batch_size=32,
        shuffle=False):

    generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    return generator.flow_from_directory(
        directory=str(path),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=shuffle
    )


def get_class_names_from_generator(generator):

    return sorted(
        generator.class_indices,
        key=generator.class_indices.get
    )
