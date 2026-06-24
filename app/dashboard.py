"""Streamlit dashboard for Driver Distraction Classification.

Run:  streamlit run app/dashboard.py

Features:
  * Upload a dashboard-camera driver image.
  * See the predicted behavior class and confidence.
  * See a SAFE / MEDIUM / HIGH risk level and a recommended action.
  * See a Grad-CAM heatmap highlighting the region the model focused on.
"""

import os
import sys

import numpy as np
import streamlit as st
from PIL import Image

# Allow running via `streamlit run app/dashboard.py` from the project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
from src.preprocess import preprocess_array, make_batch
from src.predict import DistractionPredictor
from src.gradcam import compute_gradcam, overlay_heatmap

CONFIG_PATH = "config.yaml"
RISK_COLORS = {"SAFE": "#1a9850", "MEDIUM": "#fdae61", "HIGH": "#d73027"}

st.set_page_config(page_title="Driver Distraction Detector", page_icon="🚗",
                   layout="wide")


@st.cache_resource
def get_predictor(model_path):
    """Cache the loaded model across reruns (expensive to load)."""
    return DistractionPredictor(CONFIG_PATH, model_path)


def main():
    config = load_config(CONFIG_PATH)
    default_model = os.path.join(
        config["paths"]["models_dir"], "driver_distraction_model.keras"
    )

    st.title("🚗 Driver Distraction Classification")
    st.caption(
        "Detects 10 driver behaviors from a dashboard-camera image, "
        "assigns a risk level and explains the decision with Grad-CAM."
    )

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model path", value=default_model)
        st.markdown("---")
        st.markdown("**Classes**")
        for code, label in config["classes"].items():
            st.markdown(f"`{code}` — {label}")

    if not os.path.exists(model_path):
        st.warning(
            "No trained model found yet. Train one first with "
            "`python -m src.train`, then reload this page."
        )
        return

    predictor = get_predictor(model_path)
    image_size = tuple(config["data"]["image_size"])

    uploaded = st.file_uploader(
        "Upload a driver image", type=["jpg", "jpeg", "png"]
    )
    if uploaded is None:
        st.info("Upload a dashboard-camera driver image to begin.")
        return

    pil_image = Image.open(uploaded).convert("RGB")
    original = np.array(pil_image)

    # Preprocess once and reuse for prediction + Grad-CAM.
    proc = preprocess_array(original, image_size)
    result = predictor.predict_array(proc)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded image")
        st.image(pil_image, use_column_width=True)

    with col2:
        st.subheader("Prediction")
        st.metric("Predicted behavior", result["label"])
        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

        color = RISK_COLORS[result["risk_level"]]
        st.markdown(
            f"<div style='padding:0.6rem 1rem;border-radius:8px;"
            f"background:{color};color:white;font-weight:600;'>"
            f"Risk level: {result['risk_level']}</div>",
            unsafe_allow_html=True,
        )
        st.write(f"**Recommended action:** {result['recommended_action']}")

    # ---- Grad-CAM ---------------------------------------------------------
    st.subheader("Grad-CAM explanation")
    try:
        last_conv = config["gradcam"]["last_conv_layer"] or None
        heatmap, _, _ = compute_gradcam(
            predictor.model, make_batch(proc), last_conv_layer_name=last_conv
        )
        overlay = overlay_heatmap(
            original, heatmap, alpha=config["gradcam"]["alpha"]
        )
        gc1, gc2 = st.columns(2)
        gc1.image(original, caption="Original", use_column_width=True)
        gc2.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
        st.caption(
            "Warm (red) regions are where the model looked most when making "
            "its decision — ideally the hands, phone or steering area."
        )
    except Exception as e:  # noqa: BLE001 — keep the dashboard alive on GC errors
        st.error(f"Could not compute Grad-CAM: {e}")

    # ---- Probability breakdown -------------------------------------------
    st.subheader("Class probabilities")
    probs = result["probabilities"]
    st.bar_chart(probs)


if __name__ == "__main__":
    main()
