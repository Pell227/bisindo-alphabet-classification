import io,json,time
import numpy as np
import streamlit as st
import tensorflow as tf

from pathlib import Path
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILES = {
    "MobileNetV2": BASE_DIR / "models" / "mobilenetv2_bisindo.keras",
    "ResNet50": BASE_DIR / "models" / "resnet50_bisindo.keras",
    "EfficientNetB0": BASE_DIR / "models" / "efficientnetb0_bisindo.keras",
}

DEFAULT_CLASSES = list("ABCDEFGHIJKLMN") + ["NOTHING"] + list("OPQRSTUVWXYZ")
LOW_CONFIDENCE = 60.0

def get_available_models() -> dict[str, Path]:
    return {name: path for name, path in MODEL_FILES.items() if path.exists()}


def load_class_names() -> list[str]:
    class_file = BASE_DIR / "class_names.json"
    if class_file.exists():
        with class_file.open(encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CLASSES


@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)


def prepare_image(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")

    arr = tf.image.resize_with_pad(
        np.array(img, dtype=np.float32), 224, 224
    ).numpy()

    batch = arr[np.newaxis, ...]
    return img, batch


def predict(model, batch, class_names: list[str], top_k: int = 3):
    t0 = time.time()
    probs = model.predict(batch, verbose=0)[0]
    ms = (time.time() - t0) * 1000
    top_indices = np.argsort(probs)[::-1][:top_k]
    results = [(class_names[i], float(probs[i]) * 100) for i in top_indices]
    return results, ms