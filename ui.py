from __future__ import annotations

import pandas as pd
import streamlit as st

from login import require_login
from model_utils import MODEL_PATH, predict_image


st.set_page_config(
    page_title="MedAI Vision",
    page_icon="M",
    layout="wide",
)


def apply_medical_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --med-bg: #f6fbfb;
            --med-surface: #ffffff;
            --med-line: #d8eeee;
            --med-primary: #007f7a;
            --med-primary-dark: #005f5b;
            --med-accent: #18a999;
            --med-positive: #16825d;
            --med-alert: #c2413b;
            --med-ink: #173635;
            --med-muted: #5f7776;
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(232, 248, 247, 0.95), rgba(246, 251, 251, 1) 320px),
                var(--med-bg);
            color: var(--med-ink);
        }

        section[data-testid="stSidebar"] {
            background: #edf8f7;
            border-right: 1px solid var(--med-line);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        .med-hero {
            border: 1px solid var(--med-line);
            border-left: 7px solid var(--med-primary);
            border-radius: 8px;
            background: var(--med-surface);
            padding: 1.5rem 1.7rem;
            box-shadow: 0 14px 35px rgba(0, 127, 122, 0.08);
            margin-bottom: 1.25rem;
            overflow: hidden;
            position: relative;
        }

        .med-hero-grid {
            align-items: center;
            display: grid;
            gap: 1.1rem;
            grid-template-columns: 1fr 290px;
            position: relative;
            z-index: 1;
        }

        .medical-signal {
            align-items: center;
            background: #f0fbfa;
            border: 1px solid var(--med-line);
            border-radius: 8px;
            display: grid;
            gap: 0.7rem;
            grid-template-columns: 62px 1fr;
            min-height: 108px;
            padding: 0.85rem;
        }

        .medical-cross {
            align-items: center;
            background: var(--med-primary);
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(0, 127, 122, 0.22);
            display: flex;
            height: 58px;
            justify-content: center;
            position: relative;
            width: 58px;
        }

        .medical-cross::before,
        .medical-cross::after {
            background: #ffffff;
            border-radius: 3px;
            content: "";
            position: absolute;
        }

        .medical-cross::before {
            height: 34px;
            width: 11px;
        }

        .medical-cross::after {
            height: 11px;
            width: 34px;
        }

        .ecg-card {
            min-width: 0;
        }

        .ecg-label {
            color: var(--med-primary-dark);
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }

        .ecg-svg {
            display: block;
            height: 54px;
            width: 100%;
        }

        .ecg-grid-line {
            stroke: rgba(0, 127, 122, 0.16);
            stroke-width: 1;
        }

        .ecg-path {
            animation: ecg-flow 2.8s linear infinite;
            fill: none;
            stroke: var(--med-accent);
            stroke-dasharray: 360;
            stroke-dashoffset: 360;
            stroke-linecap: round;
            stroke-linejoin: round;
            stroke-width: 4;
        }

        .ecg-dot {
            animation: pulse-dot 1.4s ease-in-out infinite;
            fill: var(--med-alert);
            transform-origin: center;
        }

        @keyframes ecg-flow {
            0% {
                stroke-dashoffset: 360;
            }
            45% {
                stroke-dashoffset: 0;
            }
            100% {
                stroke-dashoffset: -360;
            }
        }

        @keyframes pulse-dot {
            0%, 100% {
                opacity: 0.5;
                transform: scale(0.8);
            }
            50% {
                opacity: 1;
                transform: scale(1.2);
            }
        }

        .med-kicker {
            color: var(--med-primary-dark);
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .med-title {
            color: var(--med-ink);
            font-size: 2.4rem;
            line-height: 1.1;
            font-weight: 800;
            margin: 0;
        }

        .med-subtitle {
            color: var(--med-muted);
            font-size: 1rem;
            margin-top: 0.65rem;
            max-width: 760px;
        }

        .med-panel {
            border: 1px solid var(--med-line);
            border-radius: 8px;
            background: var(--med-surface);
            padding: 1.15rem;
            box-shadow: 0 10px 28px rgba(23, 54, 53, 0.06);
            min-height: 100%;
        }

        .med-panel h3 {
            color: var(--med-ink);
            margin-top: 0;
            margin-bottom: 0.45rem;
            font-size: 1.05rem;
        }

        .med-panel p {
            color: var(--med-muted);
            margin-bottom: 0;
        }

        .result-normal,
        .result-disease {
            border-radius: 8px;
            padding: 1rem 1.1rem;
            border: 1px solid;
            margin: 0.75rem 0;
        }

        .result-normal {
            background: #edf9f3;
            border-color: #b8e5d0;
            color: var(--med-positive);
        }

        .result-disease {
            background: #fff1f0;
            border-color: #f0b7b3;
            color: var(--med-alert);
        }

        .result-title {
            font-size: 1.3rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .result-copy {
            font-size: 0.96rem;
            margin: 0;
        }

        div[data-testid="stFileUploader"] {
            border: 1px dashed #8fcfca;
            border-radius: 8px;
            background: #fbfefe;
            padding: 0.4rem;
        }

        div.stButton > button:first-child {
            background: var(--med-primary);
            border: 1px solid var(--med-primary);
            border-radius: 8px;
            color: white;
            font-weight: 700;
            padding: 0.55rem 1.1rem;
        }

        div.stButton > button:first-child:hover {
            background: var(--med-primary-dark);
            border-color: var(--med-primary-dark);
            color: white;
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--med-line);
            border-radius: 8px;
            background: #fbfefe;
            padding: 0.85rem 1rem;
        }

        @media (max-width: 760px) {
            .med-hero-grid {
                grid-template-columns: 1fr;
            }

            .medical-signal {
                grid-template-columns: 52px 1fr;
            }

            .medical-cross {
                height: 50px;
                width: 50px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="med-hero">
            <div class="med-hero-grid">
                <div>
                    <div class="med-kicker">Clinical image assessment</div>
                    <h1 class="med-title">MedAI Vision</h1>
                    <div class="med-subtitle">
                        Machine learning assisted screening for X-ray and MRI images with
                        Normal or Disease classification and confidence scoring.
                    </div>
                </div>
                <div class="medical-signal">
                    <div class="medical-cross" aria-label="Medical cross"></div>
                    <div class="ecg-card">
                        <div class="ecg-label">Live ECG Signal</div>
                        <svg class="ecg-svg" viewBox="0 0 260 70" role="img" aria-label="Animated ECG line">
                            <line class="ecg-grid-line" x1="0" y1="18" x2="260" y2="18"></line>
                            <line class="ecg-grid-line" x1="0" y1="36" x2="260" y2="36"></line>
                            <line class="ecg-grid-line" x1="0" y1="54" x2="260" y2="54"></line>
                            <path class="ecg-path" d="M0 38 H38 L48 38 L58 18 L70 58 L84 34 L104 34 L116 38 H146 L158 26 L170 46 L182 34 H214 L226 38 H260"></path>
                            <circle class="ecg-dot" cx="70" cy="58" r="4"></circle>
                        </svg>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Clinical Workflow")
        st.write("1. Log in securely")
        st.write("2. Upload a scan image")
        st.write("3. Review prediction confidence")
        st.write("4. Confirm findings with a medical professional")
        st.markdown("---")
        st.markdown("### Model")
        st.write("Algorithm: Random Forest")
        st.write("Classes: Normal, Disease")
        st.write("Input: JPG, JPEG, PNG")


def render_prediction_result(label: str, confidence: float | None) -> None:
    if label.lower() == "disease":
        result_class = "result-disease"
        result_title = "Disease pattern detected"
        result_copy = "The image contains abnormal visual features according to the trained model."
    else:
        result_class = "result-normal"
        result_title = "Normal pattern detected"
        result_copy = "The image is closest to the normal class according to the trained model."

    st.markdown(
        f"""
        <div class="{result_class}">
            <div class="result-title">{result_title}</div>
            <p class="result-copy">{result_copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence is not None:
        st.metric("Confidence", f"{confidence * 100:.2f}%")
    else:
        st.warning("Confidence score is not available for this model.")


def main() -> None:
    apply_medical_theme()

    if not require_login():
        return

    render_sidebar()
    render_header()

    st.warning(
        "This tool is for educational use only and is not a replacement for professional medical diagnosis."
    )

    intro_col, status_col = st.columns([1.7, 1])

    with intro_col:
        st.markdown(
            """
            <div class="med-panel">
                <h3>Scan Analysis</h3>
                <p>
                    Upload a medical image to receive a binary prediction and confidence score.
                    Clear, centered scan images produce the most consistent results.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with status_col:
        model_status = "Ready" if MODEL_PATH.exists() else "Training required"
        st.markdown(
            f"""
            <div class="med-panel">
                <h3>Model Status</h3>
                <p>{model_status}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Image Upload")
    uploaded_file = st.file_uploader("Choose an X-ray or MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload a JPG or PNG image to begin analysis.")
        return

    image_bytes = uploaded_file.getvalue()
    image_col, action_col = st.columns([1.35, 1])

    with image_col:
        st.image(image_bytes, caption="Uploaded medical image", use_container_width=True)

    with action_col:
        st.markdown("### Diagnosis Assistant")
        st.write("Run the trained classifier on the uploaded scan.")

        if st.button("Analyze Scan", type="primary", use_container_width=True):
            if not MODEL_PATH.exists():
                st.error("Trained model not found.")
                st.code("python train_model.py", language="bash")
                st.stop()

            try:
                result = predict_image(image_bytes)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

            render_prediction_result(result["label"], result["confidence"])

            if result["probabilities"]:
                st.subheader("Class Probabilities")
                probability_data = pd.DataFrame(
                    {
                        "Class": list(result["probabilities"].keys()),
                        "Probability": [value * 100 for value in result["probabilities"].values()],
                    }
                ).set_index("Class")
                st.bar_chart(probability_data)


if __name__ == "__main__":
    main()
