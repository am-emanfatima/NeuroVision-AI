from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import plotly.express as px
import pandas as pd
from PIL import Image

# ------------------------------------------------------------
# NeuroVision AI — BRISC2025 four-class MRI research demo
# Model input follows the training pipeline: RGB, 224x224,
# float pixel values in the 0-255 range. EfficientNetB0 includes
# its own input rescaling layer.
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "BRISC2025_EfficientNetB0_finetuned.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No tumor",
    "pituitary": "Pituitary",
}
CLASS_COLORS = {
    "glioma": "#38BDF8",
    "meningioma": "#A78BFA",
    "no_tumor": "#34D399",
    "pituitary": "#FB7185",
}

st.set_page_config(
    page_title="NeuroVision AI | MRI Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    .stApp {
      background: radial-gradient(circle at 8% 0%, rgba(14,165,233,.13), transparent 29%),
                  radial-gradient(circle at 92% 8%, rgba(139,92,246,.12), transparent 30%),
                  #08111f;
      color: #e7eef9;
      font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(8,17,31,.84); }
    [data-testid="stSidebar"] > div:first-child {
      background: linear-gradient(180deg, #0d192b, #091321);
      border-right: 1px solid rgba(148,163,184,.14);
    }
    .block-container { max-width: 1440px; padding-top: 1.6rem; padding-bottom: 2.8rem; }
    .hero {
      padding: 28px 30px; border-radius: 22px; margin-bottom: 22px;
      border: 1px solid rgba(94,234,212,.23);
      background: linear-gradient(120deg, rgba(15,34,57,.98), rgba(18,26,49,.94), rgba(24,28,61,.91));
      box-shadow: 0 18px 55px rgba(0,0,0,.16);
    }
    .eyebrow { color:#5eead4; font-size:11px; font-weight:800; letter-spacing:2px; text-transform:uppercase; }
    .hero h1 { margin:10px 0 8px; color:#f8fafc; font-size:clamp(30px,4vw,45px); line-height:1.12; }
    .hero h1 span { color:#5eead4; }
    .hero p { color:#b7c6d9; max-width:820px; line-height:1.65; margin:0; }
    .chip { display:inline-block; margin:13px 7px 0 0; padding:6px 11px; border:1px solid rgba(148,163,184,.23); border-radius:30px; background:rgba(255,255,255,.045); color:#dce7f4; font-size:12px; }
    .section-label { color:#5eead4; font-size:11px; font-weight:800; letter-spacing:1.7px; text-transform:uppercase; }
    .result-card { padding:21px; border-radius:18px; border:1px solid rgba(94,234,212,.25); background:linear-gradient(135deg,rgba(13,148,136,.16),rgba(30,41,59,.76)); }
    .result-kicker { color:#5eead4; font-size:11px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; }
    .result-class { color:#f8fafc; font-size:30px; font-weight:800; margin-top:7px; }
    .result-sub { color:#cbd5e1; font-size:14px; margin-top:6px; }
    div[data-testid="stMetric"] { background:rgba(19,34,54,.76); border:1px solid rgba(148,163,184,.16); border-radius:15px; padding:14px 17px; }
    div[data-testid="stMetricLabel"] { color:#a8b8cd; }
    div[data-testid="stMetricValue"] { color:#f1f5f9; }
    [data-testid="stFileUploaderDropzone"] { background:rgba(15,29,47,.82); border:1.5px dashed rgba(94,234,212,.48); border-radius:16px; }
    div[data-testid="stAlert"] { border-radius:13px; }
    hr { border-color:rgba(148,163,184,.17); }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the NeuroVision AI model...")
def load_model(model_path: str, modified_time: float):
    # modified_time invalidates cache if the model file is replaced.
    return tf.keras.models.load_model(model_path, compile=False)


def prepare_image(uploaded_image: Image.Image) -> np.ndarray:
    image = uploaded_image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def class_label(name: str) -> str:
    return CLASS_LABELS.get(name, name.replace("_", " ").title())


# Sidebar
with st.sidebar:
    st.markdown("<div style='font-size:27px'>🧠</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:21px;font-weight:800;color:#f1f5f9'>NeuroVision <span style='color:#5eead4'>AI</span></div>", unsafe_allow_html=True)
    st.caption("BRISC2025 · MRI classification research demo")
    st.divider()
    st.markdown("**Model**")
    st.write("EfficientNetB0")
    st.markdown("**Input format**")
    st.write("JPG · JPEG · PNG")
    st.markdown("**Input size**")
    st.write("224 × 224 RGB")
    st.divider()
    st.markdown("**Classes**")
    for name in CLASS_NAMES:
        st.markdown(
            f"<span style='color:{CLASS_COLORS[name]}'>●</span> {class_label(name)}",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Research prototype only. Not intended for medical diagnosis or treatment decisions.")

# Hero
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">BRISC2025 · AI RESEARCH WORKSPACE</div>
      <h1>Brain MRI <span>Classification</span></h1>
      <p>Upload a brain MRI image to explore the output of an EfficientNetB0 four-class image classifier. Review the image, compare class scores, and inspect the model's ranking in one workspace.</p>
      <span class="chip">EfficientNetB0</span>
      <span class="chip">Transfer learning</span>
      <span class="chip">Four-class output</span>
      <span class="chip">Interactive analysis</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error(
        "Model file not found. Place `BRISC2025_EfficientNetB0_finetuned.keras` "
        "in the same folder as `app.py`, then redeploy."
    )
    st.stop()

# Load model once per app instance.
try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"The saved model could not be loaded: {exc}")
    st.stop()

analysis_tab, about_tab = st.tabs(["✦  MRI Analysis", "ⓘ  About this model"])

with analysis_tab:
    st.markdown('<div class="section-label">IMAGE WORKSPACE</div>', unsafe_allow_html=True)
    st.subheader("Upload an MRI image")
    st.write("Select a JPG, JPEG, or PNG image. It will be converted to RGB and resized to 224 × 224 for inference.")

    uploaded = st.file_uploader(
        "Choose a brain MRI image",
        type=["jpg", "jpeg", "png"],
        max_upload_size=10,
        help="Use sample or appropriately de-identified images. Avoid uploading identifiable personal medical images to a public demo.",
    )

    if uploaded is None:
        st.markdown(
            """
            <div class="result-card">
              <div class="result-kicker">READY FOR ANALYSIS</div>
              <div class="result-class">Your workspace is ready</div>
              <div class="result-sub">Upload an image above to view the MRI preview, top-scoring class, four class scores, and detailed ranking.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            image = Image.open(uploaded).convert("RGB")
            original_width, original_height = image.size
            image_array = prepare_image(image)

            with st.spinner("Running image classification..."):
                raw_scores = model.predict(image_array, verbose=0)[0]

            scores = np.asarray(raw_scores, dtype=np.float64)
            if scores.shape != (len(CLASS_NAMES),):
                st.error("The model output does not match the four configured class labels.")
                st.stop()
            if not np.all(np.isfinite(scores)):
                st.error("The model returned invalid numeric scores.")
                st.stop()

            ranked_indices = np.argsort(scores)[::-1]
            top_index = int(ranked_indices[0])
            second_index = int(ranked_indices[1])
            top_class = CLASS_NAMES[top_index]
            top_score = float(scores[top_index] * 100)
            second_score = float(scores[second_index] * 100)
            score_gap = top_score - second_score

            left, right = st.columns([0.95, 1.15], gap="large")
            with left:
                st.markdown('<div class="section-label">INPUT IMAGE</div>', unsafe_allow_html=True)
                st.image(image, caption="Uploaded MRI image", use_container_width=True)
                st.caption(f"Original dimensions: {original_width} × {original_height} pixels")
                st.caption("Model input: RGB · 224 × 224 pixels")

            with right:
                st.markdown('<div class="section-label">PREDICTION SUMMARY</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="result-card">
                      <div class="result-kicker">HIGHEST-SCORING CLASS</div>
                      <div class="result-class" style="color:{CLASS_COLORS[top_class]}">{class_label(top_class)}</div>
                      <div class="result-sub">Top model score: <b>{top_score:.2f}%</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                metric_a, metric_b = st.columns(2)
                metric_a.metric("Top model score", f"{top_score:.2f}%")
                metric_b.metric("Top-two score gap", f"{score_gap:.2f} percentage points")
                st.caption("Scores are model outputs, not clinically calibrated probabilities or a diagnosis.")

            st.divider()
            st.markdown('<div class="section-label">CLASS SCORE DISTRIBUTION</div>', unsafe_allow_html=True)
            st.subheader("Compare all four classes")

            chart_data = pd.DataFrame(
                {
                    "Class": [class_label(CLASS_NAMES[i]) for i in ranked_indices],
                    "Model score (%)": [float(scores[i] * 100) for i in ranked_indices],
                    "Class key": [CLASS_NAMES[i] for i in ranked_indices],
                }
            )
            fig = px.bar(
                chart_data,
                x="Model score (%)",
                y="Class",
                orientation="h",
                text=chart_data["Model score (%)"].map(lambda x: f"{x:.2f}%"),
                color="Class key",
                color_discrete_map=CLASS_COLORS,
                template="plotly_dark",
            )
            fig.update_layout(
                height=350,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=5, r=35, t=15, b=45),
                xaxis=dict(title="Model score (%)", range=[0, 112], gridcolor="rgba(148,163,184,.13)"),
                yaxis=dict(title="", autorange="reversed"),
                font=dict(color="#DCE7F4", family="DM Sans, sans-serif"),
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

            detail_rows = [
                {
                    "Rank": rank,
                    "Class": class_label(CLASS_NAMES[idx]),
                    "Model score (%)": round(float(scores[idx] * 100), 2),
                }
                for rank, idx in enumerate(ranked_indices, start=1)
            ]
            st.subheader("Detailed class ranking")
            st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)

            second_class = class_label(CLASS_NAMES[second_index])
            st.subheader("Interpretation")
            st.markdown(
                f"""
                - **Highest-scoring class:** {class_label(top_class)}.
                - **Second-highest class:** {second_class}.
                - **Difference between the two scores:** {score_gap:.2f} percentage points.
                - The ranking reflects this model's output for this image only; it does not establish whether a person has or does not have a condition.
                """
            )

            report = pd.DataFrame(
                [
                    {"Class": class_label(name), "Model score (%)": round(float(scores[i] * 100), 4)}
                    for i, name in enumerate(CLASS_NAMES)
                ]
            )
            st.download_button(
                "Download class scores (CSV)",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="neurovision_class_scores.csv",
                mime="text/csv",
            )
            st.info("Uploaded images are processed in the app session and are not intentionally saved by this code. Public demo users should avoid uploading identifiable or private medical images.")

        except Exception as exc:
            st.error(f"Unable to process this image: {exc}")

with about_tab:
    st.markdown('<div class="section-label">PROJECT DOCUMENTATION</div>', unsafe_allow_html=True)
    st.subheader("About NeuroVision AI")
    st.write(
        "NeuroVision AI is an educational research demonstration for four-class "
        "brain MRI image classification using an EfficientNetB0 model trained on BRISC2025."
    )
    st.markdown("**Configured classes**")
    st.write("Glioma · Meningioma · No tumor · Pituitary")
    st.markdown("**Inference workflow**")
    st.markdown(
        "1. Upload a supported 2D MRI image.\n"
        "2. Convert it to RGB and resize it to 224 × 224 pixels.\n"
        "3. Run the saved EfficientNetB0 model.\n"
        "4. Display the four output scores, ranking, and chart."
    )
    st.markdown("**Important limitations**")
    st.markdown(
        "- This is a 2D image classifier, not a 3D MRI segmentation system.\n"
        "- Performance may differ on images from other scanners, hospitals, datasets, or image-processing pipelines.\n"
        "- Model scores may not be calibrated probabilities.\n"
        "- This project has not been validated for clinical use."
    )
    st.error("This research prototype is not a medical device. Do not use its outputs to diagnose, rule out a condition, or make treatment decisions.")

st.divider()
st.caption("NEUROVISION AI · BRISC2025 RESEARCH DEMO · EFFICIENTNETB0 · NOT FOR CLINICAL USE")
