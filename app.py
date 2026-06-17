

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Page configuration (must be the first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Student Health Monitoring",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODERS_PATH = BASE_DIR / "encoders.pkl"
METADATA_PATH = BASE_DIR / "feature_metadata.json"

# --------------------------------------------------------------------------- #
# Friendly labels & help text (falls back gracefully for unknown columns)
# --------------------------------------------------------------------------- #
FRIENDLY_LABELS = {
    # Reworded (gentler) phrasing — these still map to the exact model inputs.
    "Have you ever had suicidal thoughts ?": "Have you recently felt persistent hopelessness or distressing thoughts?",
    "Family History of Mental Illness": "Has anyone in your family experienced mental-health difficulties?",
    "Work/Study Hours": "Work / Study Hours (per day)",
    "CGPA": "CGPA (0 - 4 scale)",
    "Academic Pressure": "Academic Pressure (0 = none, 5 = extreme)",
    "Study Satisfaction": "Study Satisfaction (0 = low, 5 = high)",
    "Financial Stress": "Financial Stress (1 = low, 5 = high)",
}

# Soft grouping for a clean, organised layout. Any feature not listed here is
# automatically placed in an "Other Information" group, so the app keeps working
# even if the dataset columns change.
FEATURE_GROUPS = [
    ("👤 Demographics", ["Gender", "Age"]),
    (
        "🎓 Academic",
        ["Academic Pressure", "CGPA", "Study Satisfaction", "Degree", "Work/Study Hours"],
    ),
    ("🌙 Lifestyle", ["Sleep Duration", "Dietary Habits", "Financial Stress"]),
    (
        "🧠 Emotional Wellbeing & Family Background",
        ["Have you ever had suicidal thoughts ?", "Family History of Mental Illness"],
    ),
]

# --------------------------------------------------------------------------- #
# Presentation-only conversions (the model/metadata are NOT changed)
# --------------------------------------------------------------------------- #
# CGPA is trained/stored on a 0-10 scale but shown to users on a 0-4 scale.
# We convert the user's 0-4 value back to 0-10 before feeding the model.
CGPA_SCALE_FACTOR = 2.5  # 10 / 4
CGPA_DISPLAY_MAX = 4.0

# Degree is shown as a full programme name; the model was trained on the
# original abbreviations, so we map back to the abbreviation before predicting.
DEGREE_FULL_NAMES = {
    "B.Arch": "Bachelor of Architecture (B.Arch)",
    "B.Com": "Bachelor of Commerce (B.Com)",
    "B.Ed": "Bachelor of Education (B.Ed)",
    "B.Pharm": "Bachelor of Pharmacy (B.Pharm)",
    "B.Tech": "Bachelor of Technology (B.Tech)",
    "BA": "Bachelor of Arts (BA)",
    "BBA": "Bachelor of Business Administration (BBA)",
    "BCA": "Bachelor of Computer Applications (BCA)",
    "BE": "Bachelor of Engineering (BE)",
    "BHM": "Bachelor of Hotel Management (BHM)",
    "BSc": "Bachelor of Science (BSc)",
    "Class 12": "Class 12 (High School)",
    "LLB": "Bachelor of Laws (LLB)",
    "LLM": "Master of Laws (LLM)",
    "M.Com": "Master of Commerce (M.Com)",
    "M.Ed": "Master of Education (M.Ed)",
    "M.Pharm": "Master of Pharmacy (M.Pharm)",
    "M.Tech": "Master of Technology (M.Tech)",
    "MA": "Master of Arts (MA)",
    "MBA": "Master of Business Administration (MBA)",
    "MBBS": "Bachelor of Medicine and Bachelor of Surgery (MBBS)",
    "MCA": "Master of Computer Applications (MCA)",
    "MD": "Doctor of Medicine (MD)",
    "ME": "Master of Engineering (ME)",
    "MHM": "Master of Hotel Management (MHM)",
    "MSc": "Master of Science (MSc)",
    "Others": "Others",
    "PhD": "Doctor of Philosophy (PhD)",
}
# Reverse lookup: full name -> original abbreviation used by the encoder.
DEGREE_FULL_TO_ABBR = {full: abbr for abbr, full in DEGREE_FULL_NAMES.items()}


# --------------------------------------------------------------------------- #
# Responsive, healthcare-themed styling
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    """Custom CSS for a polished, mobile-first responsive UI."""
    st.markdown(
        """
        <style>
        /* Constrain content width on large screens, full width on small ones */
        .block-container {
            max-width: 1100px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        /* App header banner */
        .app-header {
            background: linear-gradient(135deg, #1f6feb 0%, #2ea043 100%);
            color: #ffffff;
            padding: 1.4rem 1.6rem;
            border-radius: 16px;
            margin-bottom: 1.2rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        }
        .app-header h1 {
            margin: 0;
            font-size: 1.9rem;
            line-height: 1.2;
        }
        .app-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.02rem;
            opacity: 0.95;
        }
        .section-card {
            background: var(--secondary-background-color);
            border-radius: 14px;
            padding: 1.0rem 1.2rem 0.4rem 1.2rem;
            margin-bottom: 0.6rem;
            border: 1px solid rgba(128,128,128,0.15);
        }
        /* Big, full-width predict button */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #1f6feb 0%, #2ea043 100%);
            color: #ffffff;
            font-size: 1.15rem;
            font-weight: 700;
            padding: 0.8rem 1rem;
            border: none;
            border-radius: 12px;
            margin-top: 0.5rem;
        }
        div.stButton > button:hover { filter: brightness(1.05); color:#ffffff; }
        /* Prevent any horizontal scrolling on small screens */
        html, body, [data-testid="stAppViewContainer"] { overflow-x: hidden; }
        /* Mobile tweaks: smaller header, comfortable spacing */
        @media (max-width: 640px) {
            .app-header h1 { font-size: 1.4rem; }
            .app-header p  { font-size: 0.92rem; }
            .block-container { padding-left: 0.6rem; padding-right: 0.6rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Artifact loading (cached so it only happens once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model and preprocessing objects...")
def load_artifacts():
    """Load model, scaler, encoders, and feature metadata."""
    missing = [
        p.name
        for p in (MODEL_PATH, SCALER_PATH, ENCODERS_PATH, METADATA_PATH)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s): "
            + ", ".join(missing)
            + ". Run `python train_model.py` first to generate them."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    return model, scaler, encoders, metadata


def label_for(name: str) -> str:
    """Return a human-friendly label for a column name."""
    return FRIENDLY_LABELS.get(name, name)


# --------------------------------------------------------------------------- #
# Dynamic input rendering
# --------------------------------------------------------------------------- #
def render_feature_input(feature: dict):
    """Render a single Streamlit widget for one feature and return its value."""
    name = feature["name"]
    label = label_for(name)

    if feature["type"] == "categorical":
        options = feature["options"]
        # Degree -> dropdown of full programme names (mapped back when predicting).
        if name == "Degree":
            display_options = [DEGREE_FULL_NAMES.get(o, o) for o in options]
            default_index = (
                options.index(feature["default"])
                if feature["default"] in options
                else 0
            )
            return st.selectbox(label, display_options, index=default_index, key=name)

        default_index = (
            options.index(feature["default"]) if feature["default"] in options else 0
        )
        return st.selectbox(label, options, index=default_index, key=name)

    # CGPA -> number box on a 0-4 scale (converted to the model's 0-10 later).
    if name == "CGPA":
        return st.number_input(
            label,
            min_value=0.0,
            max_value=CGPA_DISPLAY_MAX,
            value=round(float(feature["default"]) / CGPA_SCALE_FACTOR, 2),
            step=0.01,
            format="%.2f",
            key=name,
        )

    # Numeric feature -> number input box (type a value or use the - / + steppers).
    if feature.get("is_integer"):
        return st.number_input(
            label,
            min_value=int(feature["min"]),
            max_value=int(feature["max"]),
            value=int(feature["default"]),
            step=1,
            key=name,
        )
    return st.number_input(
        label,
        min_value=float(feature["min"]),
        max_value=float(feature["max"]),
        value=float(feature["default"]),
        step=float(feature.get("step", 0.01)),
        key=name,
    )


def build_groups(features: list[dict]):
    """Organise features into display groups (with an automatic fallback)."""
    by_name = {f["name"]: f for f in features}
    grouped, used = [], set()

    for title, names in FEATURE_GROUPS:
        members = [by_name[n] for n in names if n in by_name]
        if members:
            grouped.append((title, members))
            used.update(f["name"] for f in members)

    leftovers = [f for f in features if f["name"] not in used]
    if leftovers:
        grouped.append(("📋 Other Information", leftovers))
    return grouped


def collect_inputs(features: list[dict]) -> dict:
    """Render all inputs (grouped, two responsive columns) and collect values."""
    values = {}
    for title, members in build_groups(features):
        st.markdown(f"#### {title}")
        with st.container():
            # Two columns stack vertically on mobile -> no horizontal scroll.
            cols = st.columns(2)
            for i, feature in enumerate(members):
                with cols[i % 2]:
                    values[feature["name"]] = render_feature_input(feature)
        st.markdown("")  # small spacer
    return values


# --------------------------------------------------------------------------- #
# Prediction pipeline (mirrors training preprocessing exactly)
# --------------------------------------------------------------------------- #
def predict(user_inputs, model, scaler, encoders, metadata):
    """Encode -> order -> scale -> predict. Returns (prediction, risk_prob)."""
    feature_order = metadata["feature_order"]

    row = {}
    for col in feature_order:
        value = user_inputs[col]

        # CGPA is entered on a 0-4 scale; the model was trained on a 0-10 scale.
        if col == "CGPA":
            value = float(value) * CGPA_SCALE_FACTOR
        # Degree is shown as a full name; convert back to the trained abbreviation.
        elif col == "Degree":
            value = DEGREE_FULL_TO_ABBR.get(value, value)

        if col in encoders:
            # Same label encoding learned at training time.
            value = int(encoders[col].transform([str(value)])[0])
        row[col] = value

    # Build a single-row DataFrame with the exact training column names/order so
    # the scaler receives the feature names it was fitted with (no warnings).
    X = pd.DataFrame([row], columns=feature_order).astype(float)
    X_scaled = scaler.transform(X)

    prediction = int(model.predict(X_scaled)[0])

    risk_prob = None
    if hasattr(model, "predict_proba"):
        # Probability of the positive class (Depression = 1).
        risk_prob = float(model.predict_proba(X_scaled)[0][1])
    return prediction, risk_prob


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def show_result(prediction: int, risk_prob, user_inputs: dict):
    st.markdown("---")
    st.subheader("🔎 Prediction Result")

    if prediction == 1:
        st.warning("### ⚠️ Depression Risk Detected")
        st.warning(
            "The model indicates that this student **may be at risk** and should be "
            "encouraged to seek professional support."
        )
    else:
        st.success("### ✅ No Depression Risk Detected")
        st.success(
            "The model indicates a **lower likelihood** of depression risk based on "
            "the provided information."
        )

    # ----------------------------- Risk score ----------------------------- #
    if risk_prob is not None:
        risk_pct = int(round(risk_prob * 100))
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Risk Score", f"{risk_pct}%")
        with m2:
            st.metric(
                "Assessment",
                "High Risk" if prediction == 1 else "Low Risk",
            )
        st.progress(min(max(risk_pct, 0), 100))
        st.caption(
            "Risk Score = model's estimated probability that the student is at "
            "risk of depression."
        )



# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()

    # Header banner
    st.markdown(
        """
        <div class="app-header">
            <h1>🧠 Student Health Monitoring and Early Detection</h1>
            <p>This application predicts the likelihood of depression risk among
            students based on demographic, academic, lifestyle, and psychological
            factors, using a Support Vector Machine (SVM) model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load artifacts (with a friendly error if training hasn't been run).
    try:
        model, scaler, encoders, metadata = load_artifacts()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()

    st.markdown("### 📝 Enter Student Information")
    user_inputs = collect_inputs(metadata["features"])

    # Large primary action button.
    predict_clicked = st.button("🔍 Predict Depression Risk", type="primary")

    if predict_clicked:
        with st.spinner("Analysing the provided information..."):
            prediction, risk_prob = predict(
                user_inputs, model, scaler, encoders, metadata
            )
        show_result(prediction, risk_prob, user_inputs)

    # Footer
    st.markdown("---")
    st.caption(
        "Final-year project · Student Health Monitoring · Powered by Streamlit & "
        "scikit-learn (SVM)."
    )


if __name__ == "__main__":
    main()
