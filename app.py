import io
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).parent
MODEL_FILES = {
    "MobileNetV2": BASE_DIR / "models" / "mobilenetv2_bisindo.keras",
    "ResNet50": BASE_DIR / "models" / "resnet50_bisindo.keras",
    "EfficientNetB0": BASE_DIR / "models" / "efficientnetb0_bisindo.keras",
}

DEFAULT_CLASSES = list("ABCDEFGHIJKLMN") + ["NOTHING"] + list("OPQRSTUVWXYZ")
LOW_CONFIDENCE = 60.0


def load_class_names():
    f = BASE_DIR / "class_names.json"
    return json.load(open(f)) if f.exists() else DEFAULT_CLASSES


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


def prepare_image(file_bytes, model_name):
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")

    # Resize dengan pad langsung ke ukuran 224x224
    arr = tf.image.resize_with_pad(
        np.array(img, dtype=np.float32), 224, 224
    ).numpy()

    batch = arr[np.newaxis, ...]

    # Terapkan preprocessing spesifik sesuai model
    if model_name == "MobileNetV2":
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    elif model_name == "ResNet50":
        batch = tf.keras.applications.resnet50.preprocess_input(batch)
    elif model_name == "EfficientNetB0":
        batch = tf.keras.applications.efficientnet.preprocess_input(batch)

    return img, batch


def predict(model, batch, class_names, top_k=3):
    t0 = time.time()
    probs = model.predict(batch, verbose=0)[0]
    ms = (time.time() - t0) * 1000
    top = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i]) * 100) for i in top], ms


def main():
    st.set_page_config(page_title="Klasifikasi BISINDO")
    st.title("Klasifikasi Alfabet BISINDO")
    st.caption("Unggah atau ambil foto isyarat tangan untuk diprediksi oleh model.")

    class_names = load_class_names()
    available = {n: p for n, p in MODEL_FILES.items() if p.exists()}
    if not available:
        st.error(
            "File model tidak ditemukan. Letakkan file .keras di folder `models/`."
        )
        st.stop()

    model_name = st.selectbox("Pilih model", list(available))
    model = load_model(str(available[model_name]))

    if model.output_shape[-1] != len(class_names):
        st.error(
            f"Jumlah kelas model ({model.output_shape[-1]}) tidak sama dengan class_names ({len(class_names)})."
        )
        st.stop()

    tab_upload, tab_cam = st.tabs(["Unggah foto", "Kamera"])
    with tab_upload:
        up = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])
    with tab_cam:
        cam = st.camera_input("Ambil foto")

    source = cam or up
    if source is None:
        st.info("Belum ada gambar.")
        return

    img, batch = prepare_image(source.getvalue(), model_name)
    st.image(img, caption="Gambar Masukan", width=320)

    results, ms = predict(model, batch, class_names)
    label, conf = results[0]

    if label == "NOTHING":
        st.subheader("Tidak ada isyarat terdeteksi")
    else:
        st.subheader(f"Prediksi: {label}")

    st.metric("Confidence", f"{conf:.2f}%")
    if conf < LOW_CONFIDENCE:
        st.warning(
            "Confidence rendah. Pastikan pencahayaan cukup dan posisi tangan jelas."
        )

    st.write("Tiga prediksi teratas")
    st.table(
        {
            "Huruf": [r[0] for r in results],
            "Confidence (%)": [round(r[1], 2) for r in results],
        }
    )
    st.caption(f"Model: {model_name} | Waktu inferensi: {ms:.0f} ms")


if __name__ == "__main__":
    main()