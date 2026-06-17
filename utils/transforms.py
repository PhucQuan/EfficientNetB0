import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


IMAGE_SIZE = 224


def load_image_for_inference(image_path, image_size=IMAGE_SIZE):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image_array = np.asarray(image, dtype="float32")
    image_array = preprocess_input(image_array)

    return np.expand_dims(image_array, axis=0)
