from pathlib import Path
import json

import streamlit as st
from PIL import Image

from src.inference.predictor import SignalScopePredictor
from src.evaluation.gradcam import make_overlay
from src.evaluation.metadata import extract_exif
from src.evaluation.faithful_explanation import (
    analyze_localization,
    build_grounded_evidence,
    draw_evidence_region,
    make_heatmap_image,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SignalScope",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .signalscope-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .signalscope-subtitle {
        font-size: 1.15rem;
        color: #64748b;
        margin-bottom: 2rem;
    }

    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
        margin-bottom: 1rem;
    }

    .result-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 750;
        margin-top: 0.35rem;
        color: #0f172a;
    }

    .ai-result {
        color: #b91c1c;
    }

    .real-result {
        color: #047857;
    }

    /* Section headings */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    /* Disclaimer */
    .disclaimer {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #475569;
        font-size: 0.9rem;
        margin-top: 2rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

@st.cache_resource
def load_predictor():
    return SignalScopePredictor()


predictor = load_predictor()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="signalscope-title">SignalScope</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="signalscope-subtitle">
        Telling Real From Synthetic in the Age of Generative Media
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    SignalScope analyzes visual patterns in an image and provides a
    <b>likelihood assessment</b> of whether the image is likely real
    or likely AI-generated.
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Upload
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Analyze an image</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload an image for forensic analysis",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG and WebP.",
)


# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    # Temporary file for predictor
    temp_path = PROJECT_ROOT / "_signalscope_uploaded_image.jpg"

    image.save(
        temp_path,
        format="JPEG",
        quality=95,
    )

    try:
        with st.spinner("Analyzing image..."):
            result = predictor.predict(temp_path)

            exif_metadata = extract_exif(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Detection Result</div>',
        unsafe_allow_html=True,
    )

    result_col1, result_col2, result_col3 = st.columns(3)

    predicted_class = result["predicted_class"]
    ai_probability = float(result["ai_probability"])
    decision_strength = float(result["confidence"])

    with result_col1:
        if predicted_class == 1:
            assessment = "Likely AI-generated"
            assessment_class = "ai-result"
        else:
            assessment = "Likely real"
            assessment_class = "real-result"

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Assessment</div>
                <div class="result-value {assessment_class}">
                    {assessment}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col2:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">AI probability</div>
                <div class="result-value">
                    {ai_probability:.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col3:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Decision strength</div>
                <div class="result-value">
                    {decision_strength:.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Probability bar
    st.markdown("**AI-generated likelihood**")

    ai_probability = float(result["ai_probability"])

    st.progress(ai_probability)

    probability_col1, probability_col2 = st.columns([1, 6])

    with probability_col1:
        st.metric(
            "AI likelihood",
            f"{ai_probability:.2%}",
        )

    with probability_col2:
        st.caption(
            "0% = model evidence favors real imagery | "
            "100% = model evidence favors AI-generated imagery"
        )

    st.caption(
        "This score represents the model's learned evidence for the "
        "AI-generated class. It is a likelihood assessment, not proof "
        "of synthetic origin."
    )

        # -----------------------------------------------------
        # Image comparison
        # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">Forensic Evidence Map</div>',
        unsafe_allow_html=True,
    )

    heatmap_image = make_overlay(
        image,
        result["heatmap"],
    )

    image_col1, image_col2 = st.columns(2)

    with image_col1:

        st.image(
            image,
            caption="Original image",
            width="stretch",
        )

    with image_col2:

        st.image(
            heatmap_image,
            caption="Model evidence overlay",
            width="stretch",
        )

    st.caption(
        "Highlighted regions show where the model's visual evidence "
        "was concentrated. The map explains model attention and does "
        "not prove that a specific region was generated or manipulated."
    )

    # -----------------------------------------------------
    # Explanation
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Why the model made this assessment'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(result["evidence"])

    st.caption(
        result["uncertainty"]
    )
    st.divider()
    # -----------------------------------------------------
    # Bonus A - Faithful Explanation / Localization
    # -----------------------------------------------------
    localization = analyze_localization(result["heatmap"])
    explanation = build_grounded_evidence(
        ai_probability=ai_probability,
        predicted_class=predicted_class,
        localization=localization,
    )

    st.divider()
    st.markdown(
        '<div class="section-title">Faithful Explanation</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**MODEL EVIDENCE**")
    for item in explanation["evidence"]:
        st.write(f"- {item}")

    st.markdown("**UNCERTAINTY**")
    st.info(explanation["uncertainty"])

    original_image = image.copy()
    heatmap_image = make_heatmap_image(
        result["heatmap"],
        size=original_image.size,
    )
    overlay_image = make_overlay(
        original_image,
        result["heatmap"],
    )
    localized_image = draw_evidence_region(
        original_image,
        explanation["bbox"],
    )

    visual_col1, visual_col2, visual_col3 = st.columns(3)

    with visual_col1:
        st.image(
            original_image,
            caption="Original image",
            width="stretch",
        )

    with visual_col2:
        st.image(
            heatmap_image,
            caption="Artifact heatmap",
            width="stretch",
        )

    with visual_col3:
        overlay_caption = (
            "Model evidence overlay - localized"
            if explanation["bbox"] is not None
            else "Model evidence overlay - diffuse evidence"
        )
        st.image(
            overlay_image,
            caption=overlay_caption,
            width="stretch",
        )

    if explanation["bbox"] is not None:
        st.image(
            localized_image,
            caption="Primary evidence region",
            width="stretch",
        )
        st.caption(
            "The region box is shown only when the measured activation is "
            "genuinely localized. It is not forced around diffuse evidence."
        )
    else:
        st.info(
            "The model evidence is distributed across the image rather than "
            "concentrated in a single region, so no primary evidence box is shown."
        )

    st.markdown(
        f"""
        **Localization diagnostics**

        - Heatmap mean: `{localization.mean_activation:.3f}`
        - Maximum activation: `{localization.max_activation:.3f}`
        - High-activation area: `{localization.active_fraction:.1%}`
        - Top-10% / mean concentration: `{localization.concentration_ratio:.2f}x`
        - Decision explained: `{"AI-generated" if predicted_class == 1 else "REAL"}`
        """
    )

    st.caption(
        "The explanation is derived deterministically from the existing "
        "trained model's Grad-CAM output and final prediction. It does not "
        "invent semantic artifacts and does not use an external LLM or API."
    )

    st.markdown('<div class="section-title">Image Metadata</div>', unsafe_allow_html=True)

    if exif_metadata:
        st.success(
            f"EXIF metadata detected: {len(exif_metadata)} fields"
        )

        metadata_rows = []

        for key, value in exif_metadata.items():
            metadata_rows.append(
                {
                    "Field": key,
                    "Value": value,
                }
            )

        st.dataframe(
            metadata_rows,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No EXIF metadata detected in this image."
        )

    st.caption(
        "Metadata is shown as supporting forensic information only. "
        "Missing or present metadata does not by itself establish "
        "whether an image is real or AI-generated."
    )

# -----------------------------------------------------
# Global Active Defence & Robustness (200-image diagnostic)
# -----------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">Global Active Defence & Robustness</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Fixed 200-image robustness benchmark. These values are intentionally "
    "the same for every uploaded image; they describe the detector globally, "
    "not the current image individually."
)

active_defense_summary = (
    PROJECT_ROOT / "results" / "active_defense" / "attack_summary.json"
)

if active_defense_summary.exists():
    try:
        with active_defense_summary.open("r", encoding="utf-8") as handle:
            attack_summary = json.load(handle)

        if attack_summary:
            worst_attack = attack_summary[0]

            st.warning(
                "[Warning] The detector is most sensitive to aggressive resizing. "
                "Heavily resized images should receive additional human review."
            )

            defence_col1, defence_col2, defence_col3 = st.columns(3)

            with defence_col1:
                st.metric(
                    "Most sensitive attack",
                    worst_attack["attack"].replace("_", " "),
                )

            with defence_col2:
                st.metric(
                    "Overall flip rate",
                    f'{worst_attack["flip_rate"]:.1%}',
                )

            with defence_col3:
                st.metric(
                    "AI-image flip rate",
                    f'{worst_attack["ai_flip_rate"]:.1%}',
                )

            table_rows = []
            for row in attack_summary:
                table_rows.append(
                    {
                        "Attack": row["attack"].replace("_", " ").title(),
                        "Overall flip": f'{row["flip_rate"]:.1%}',
                        "REAL flip": f'{row["real_flip_rate"]:.1%}',
                        "AI flip": f'{row["ai_flip_rate"]:.1%}',
                    }
                )

            st.dataframe(
                table_rows,
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "Fixed 200-image active-defence diagnostic using the "
                "production model, temperature 0.95, and threshold 0.56. "
                "No retraining or threshold tuning was performed."
            )
    except (OSError, ValueError, TypeError, KeyError):
        st.info(
            "Active-defence results could not be read. "
            "Run the active-defence analysis again to regenerate them."
        )
else:
    st.info(
        "Active-defence results are not available yet. Run the analysis "
        "script once to generate results/active_defense/attack_summary.json."
    )

# -----------------------------------------------------

# Responsible-use message
# -----------------------------------------------------

st.markdown(
    """
    <div class="disclaimer">
    [Warning] <b>Responsible use:</b>
    SignalScope provides a model-based likelihood assessment,
    not definitive proof of authenticity or synthetic origin.
    Results should be interpreted alongside other forensic evidence.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        SignalScope - AI Media Forensics - Research Prototype
    </div>
    """,
    unsafe_allow_html=True,
)
