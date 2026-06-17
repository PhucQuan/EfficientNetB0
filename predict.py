import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from utils.transforms import IMAGE_SIZE
from utils.transforms import load_image_for_inference


def parse_args():

    parser = argparse.ArgumentParser(
        description="Predict one fruit or vegetable image with top-3 output."
    )
    parser.add_argument(
        "--image-path",
        required=True
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
        "--image-size",
        type=int,
        default=IMAGE_SIZE
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3
    )

    return parser.parse_args()


def main():

    args = parse_args()

    image_path = Path(args.image_path)
    model_path = Path(args.model_path)
    class_names_path = Path(args.class_names)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")

    class_names = json.loads(
        class_names_path.read_text(encoding="utf-8")
    )
    model = keras.models.load_model(model_path)

    image_array = load_image_for_inference(
        image_path,
        image_size=args.image_size
    )
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    top_k = min(args.top_k, len(class_names))
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    predicted_index = int(top_indices[0])
    predicted_label = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    result = {
        "image_path": str(image_path),
        "predicted_label": predicted_label,
        "confidence": confidence,
        "top_predictions": [
            {
                "label": class_names[int(index)],
                "confidence": float(probabilities[int(index)])
            }
            for index in top_indices
        ]
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
