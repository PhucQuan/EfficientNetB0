import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input

from utils.transforms import IMAGE_SIZE


DEFAULT_MODEL_PATH = Path("artifacts/fruit8_efficientnet_group_regularized.keras")
DEFAULT_CLASS_NAMES_PATH = Path("artifacts/class_names.json")


@st.cache_resource
def load_artifacts(model_path, class_names_path):

    model = keras.models.load_model(model_path)
    class_names = json.loads(
        Path(class_names_path).read_text(encoding="utf-8")
    )

    return model, class_names


def predict_image(model, image, image_size):

    image = image.convert("RGB").resize((image_size, image_size))
    image_array = np.asarray(image, dtype="float32")
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    return model.predict(image_array, verbose=0)[0]


st.set_page_config(
    page_title="Fruit And Vegetable Classifier",
    layout="centered"
)

st.title("Fruit And Vegetable Image Classification")
st.caption("EfficientNetB0 transfer learning demo")

model_path = st.sidebar.text_input(
    "Model path",
    str(DEFAULT_MODEL_PATH)
)
class_names_path = st.sidebar.text_input(
    "Class names path",
    str(DEFAULT_CLASS_NAMES_PATH)
)
image_size = st.sidebar.number_input(
    "Image size",
    min_value=128,
    max_value=512,
    value=IMAGE_SIZE,
    step=32
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    preview_image = Image.open(uploaded_file)
    st.image(preview_image, caption="Uploaded image", use_container_width=True)

    model, class_names = load_artifacts(model_path, class_names_path)
    probabilities = predict_image(model, preview_image, image_size)
    top_indices = np.argsort(probabilities)[-3:][::-1]

    st.subheader("Prediction")
    st.success(
        f"{class_names[int(top_indices[0])]} "
        f"({probabilities[int(top_indices[0])] * 100:.2f}%)"
    )

    st.subheader("Top 3 Results")
    for rank, index in enumerate(top_indices, start=1):
        st.write(
            f"{rank}. {class_names[int(index)]}: "
            f"{probabilities[int(index)] * 100:.2f}%"
        )
