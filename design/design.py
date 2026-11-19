import streamlit as st

def setup_page():
    st.set_page_config(page_title="Klasifikasi BISINDO", layout="wide")

def render_top_grid(model_names: list[str]):
    col_left, col_right = st.columns([1, 1.3], gap="large")
    with col_left:
        st.title("Klasifikasi Alphabet Bahasa Isyarat Indonesia")
        st.caption("Unggah atau ambil foto isyarat tangan untuk diprediksi oleh model.")
        model_name = st.selectbox("Pilih model", model_names)

    with col_right:
        tab_upload, tab_cam = st.tabs(["Unggah foto", "Kamera"])
        with tab_upload:
            up = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])
        with tab_cam:
            cam = st.camera_input("Ambil foto")
    return model_name, (cam or up)

def render_result_grid(img, results, ms: float, model_name: str, low_conf_threshold: float):
    st.divider()
    col_preview, col_result = st.columns([1, 1.3], gap="large")
    with col_preview:
        st.image(img, caption="Hasil", use_container_width=True)
    with col_result:
        label, conf = results[0]
        if label == "NOTHING":
            st.subheader("Tidak ada bahasa isyarat yang terdeteksi")
        else:
            st.subheader(f"Prediksi: {label}")
        st.metric("Confidence", f"{conf:.2f}%")

        st.write("Tiga prediksi teratas")
        st.table(
            {
                "Huruf": [r[0] for r in results],
                "Confidence (%)": [round(r[1], 2) for r in results],
            }
        )
        st.caption(f"Model: {model_name} | Waktu inferensi: {ms:.0f} ms")