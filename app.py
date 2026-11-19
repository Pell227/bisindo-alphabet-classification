import streamlit as st

from design.design import (
    render_result_grid,
    render_top_grid,
    setup_page,
)
from visual.interface import (
    LOW_CONFIDENCE,
    get_available_models,
    load_class_names,
    load_model,
    predict,
    prepare_image,
)

def main():
    setup_page()
    class_names = load_class_names()
    available = get_available_models()
    if not available:
        st.error(
            "File model tidak ditemukan"
        )
        st.stop()

    model_name, source = render_top_grid(list(available))
    model = load_model(str(available[model_name]))

    if model.output_shape[-1] != len(class_names):
        st.error(
            f"Jumlah kelas model ({model.output_shape[-1]}) tidak sama dengan class_names ({len(class_names)})."
        )
        st.stop()

    img, batch = prepare_image(source.getvalue())
    results, ms = predict(model, batch, class_names)
    render_result_grid(img, results, ms, model_name, LOW_CONFIDENCE)

if __name__ == "__main__":
    main()