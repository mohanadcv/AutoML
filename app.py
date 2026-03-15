"""
Streamlit AutoML App - User Interface

Beautiful, intuitive UI for non-technical users to run AutoML pipeline.
Features complete workflows with visualizations at every step.
"""

from pathlib import Path
import streamlit as st
from main_pipeline import AutoMLPipeline
from Config.config import Config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import traceback
import joblib

from src.data_processing.validator import DataValidator
from src.visualizations.eda import EDAGenerator
from src.data_processing.preprocessing_pipeline import PreprocessingPipeline
from src.models.trainers.regression import RegressionTrainer
from src.models.trainers.classification import ClassificationTrainer
from src.models.hyperparameter_tuning_setup import HyperparameterTuner
from src.visualizations.training import (
    plot_classification_results,
    plot_regression_results
)
from src.visualizations.tuning import (
    plot_tuned_classification_results,
    plot_tuned_regression_results
)
from src.models.final_evaluation import (
    plot_test_classification,
    plot_test_regression
)

# ================================================================
# VISUAL CONFIGURATION & CUSTOMIZATION
# ================================================================

# --- COLOR SCHEME ---
BACKGROUND_COLOR_1 = "#0F172A"
BACKGROUND_COLOR_2 = "#020617"
PRIMARY_COLOR = "#3B82F6"
SECONDARY_COLOR = "#06B6D4"
TEXT_COLOR = "#F8FAFC"
SUCCESS_COLOR = "#22C55E"
WARNING_COLOR = "#F59E0B"
ERROR_COLOR = "#EF4444"
FONT_FAMILY = "Inter, sans-serif"
HEADING_FONT = "Poppins, sans-serif"
FONT_SIZE_BODY = "16px"
FONT_SIZE_HEADING = "42px"
FONT_SIZE_SUBHEADING = "30px"
CONTAINER_PADDING = "2rem"
SECTION_MARGIN = "3rem"
BUTTON_PADDING = "0.75rem 2rem"
BORDER_RADIUS = "12px"
BOX_SHADOW = "0 4px 6px rgba(0, 0, 0, 0.3)"
BUTTON_BG_COLOR = PRIMARY_COLOR
BUTTON_TEXT_COLOR = "white"
BUTTON_HOVER_COLOR = "#2563EB"
SIDEBAR_BG_COLOR = "#020617"
SIDEBAR_TEXT_COLOR = "#E2E8F0"

# ================================================================
# APPLY CUSTOM CSS
# ================================================================

def load_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, {BACKGROUND_COLOR_1} 0%, {BACKGROUND_COLOR_2} 100%);
        font-family: {FONT_FAMILY};
        color: {TEXT_COLOR};
    }}
    h1 {{
        font-family: {HEADING_FONT};
        font-size: {FONT_SIZE_HEADING};
        color: {TEXT_COLOR};
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }}
    h2, h3 {{
        font-family: {HEADING_FONT};
        font-size: {FONT_SIZE_SUBHEADING};
        color: {TEXT_COLOR};
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }}
    p, div, span, label {{
        font-size: {FONT_SIZE_BODY};
        color: {TEXT_COLOR};
    }}
    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG_COLOR};
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }}
    [data-testid="stSidebar"] * {{
        color: {SIDEBAR_TEXT_COLOR} !important;
    }}
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {{
        color: {PRIMARY_COLOR} !important;
    }}
    .stButton > button {{
        background-color: {BUTTON_BG_COLOR};
        color: {BUTTON_TEXT_COLOR};
        border: none;
        border-radius: {BORDER_RADIUS};
        padding: {BUTTON_PADDING};
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: {BOX_SHADOW};
        width: 100%;
    }}
    .stButton > button:hover {{
        background-color: {BUTTON_HOVER_COLOR};
        box-shadow: 0 6px 12px rgba(139, 92, 246, 0.4);
        transform: translateY(-2px);
    }}
    .stAlert {{
        background-color: rgba(139, 92, 246, 0.1);
        border-left: 4px solid {PRIMARY_COLOR};
        border-radius: {BORDER_RADIUS};
        padding: {CONTAINER_PADDING};
        color: {TEXT_COLOR};
    }}
    .success-box {{
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid {SUCCESS_COLOR};
        border-radius: {BORDER_RADIUS};
        padding: 1rem;
        margin: 1rem 0;
    }}
    .warning-box {{
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid {WARNING_COLOR};
        border-radius: {BORDER_RADIUS};
        padding: 1rem;
        margin: 1rem 0;
    }}
    .error-box {{
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid {ERROR_COLOR};
        border-radius: {BORDER_RADIUS};
        padding: 1rem;
        margin: 1rem 0;
    }}
    [data-testid="stFileUploader"] {{
        background-color: rgba(139, 92, 246, 0.05);
        border: 2px dashed {PRIMARY_COLOR};
        border-radius: {BORDER_RADIUS};
        padding: 2rem;
    }}
    .stSelectbox > div > div {{
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: {BORDER_RADIUS};
        border: 1px solid rgba(139, 92, 246, 0.3);
    }}
    .stMultiSelect > div > div {{
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: {BORDER_RADIUS};
        border: 1px solid rgba(139, 92, 246, 0.3);
    }}
    .stTextInput > div > div > input {{
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: {BORDER_RADIUS};
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: {TEXT_COLOR};
    }}
    .dataframe {{
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: {BORDER_RADIUS};
        overflow: hidden;
    }}
    .dataframe th {{
        background-color: {PRIMARY_COLOR} !important;
        color: white !important;
        font-weight: 600;
    }}
    .dataframe td {{
        color: {TEXT_COLOR} !important;
    }}
    .stProgress > div > div {{
        background-color: rgba(139, 92, 246, 0.2);
    }}
    .stProgress > div > div > div {{
        background-color: {PRIMARY_COLOR};
    }}
    .streamlit-expanderHeader {{
        background-color: rgba(139, 92, 246, 0.1);
        border-radius: {BORDER_RADIUS};
        border-left: 4px solid {PRIMARY_COLOR};
    }}
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: {PRIMARY_COLOR};
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_COLOR};
        font-weight: 500;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(139, 92, 246, 0.1);
        border-radius: {BORDER_RADIUS};
        padding: 0.75rem 1.5rem;
        color: {TEXT_COLOR};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY_COLOR};
        color: white;
    }}
    .stSpinner > div {{
        border-top-color: {PRIMARY_COLOR} !important;
    }}
    .hero {{
        text-align: center;
        padding: 3rem 0;
        margin-bottom: 2rem;
    }}
    .hero h1 {{
        font-size: 5rem;
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        font-family: 'Poppins', sans-serif;
    }}
    .hero p {{
        font-size: 1.75rem;
        color: {SIDEBAR_TEXT_COLOR};
        max-width: 700px;
        margin: 0 auto;
    }}
    .step-indicator {{
        display: inline-block;
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        line-height: 40px;
        text-align: center;
        font-weight: 700;
        margin-right: 1rem;
    }}
    .section-divider {{
        height: 2px;
        background: linear-gradient(90deg, transparent, {PRIMARY_COLOR}, transparent);
        margin: {SECTION_MARGIN} 0;
    }}
    </style>
    """, unsafe_allow_html=True)


# ================================================================
# INITIALIZE SESSION STATE
# ================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = AutoMLPipeline()
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'target_selected' not in st.session_state:
        st.session_state.target_selected = False
    if 'task_detected' not in st.session_state:
        st.session_state.task_detected = False
    if 'preprocessed' not in st.session_state:
        st.session_state.preprocessed = False
    if 'split_done' not in st.session_state:
        st.session_state.split_done = False
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'tuned' not in st.session_state:
        st.session_state.tuned = False
    if 'test_evaluated' not in st.session_state:
        st.session_state.test_evaluated = False
    if 'show_visualizations' not in st.session_state:
        st.session_state.show_visualizations = True


# ================================================================
# VISUALIZATION FUNCTIONS
# ================================================================

def show_eda_visualizations(df, target_column=None, task_type=None):
    """Generate EDA visualizations and display them once."""
    figures = {}  # Dictionary to store figures

    if not st.session_state.show_visualizations or df is None or len(df) == 0:
        return figures

    try:
        eda = EDAGenerator()

        # Only show header if we're generating new visualizations
        if 'eda_figures' not in st.session_state or not st.session_state.eda_figures:
            st.markdown("### 📊 EDA Visualizations")

        # Generate feature distributions
        fig_dist = eda.plot_feature_distributions(df)
        figures['distributions'] = fig_dist

        fig_quality = eda.plot_data_quality(df)
        figures['quality'] = fig_quality

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            fig_corr, fig_corr_table = eda.plot_correlation_matrix(df[numeric_cols])
            figures['correlation'] = fig_corr
            if fig_corr_table is not None:
                figures['correlation_table'] = fig_corr_table

        # Generate target analysis if applicable
        if target_column and target_column in df.columns and task_type:
            y = df[target_column]
            X = df.drop(columns=[target_column])
            fig_target = eda.plot_target_analysis(X, y, target_column, task_type)
            figures['target_analysis'] = fig_target

        # Display all figures
        for fig_name, fig in figures.items():
            if fig is not None:
                st.pyplot(fig)
                st.markdown("---")
                plt.close(fig)

    except Exception as e:
        st.warning(f"Could not generate EDA visualizations: {e}")

    return figures


# ================================================================
# NAVIGATION HELPER FUNCTIONS
# ================================================================

def navigate_to_step(step_num):
    """Helper function to navigate between steps."""
    st.session_state.step = step_num


def go_to_next_step():
    """Go to the next step."""
    st.session_state.step += 1


def go_to_previous_step():
    """Go to the previous step."""
    st.session_state.step = max(1, st.session_state.step - 1)


# ================================================================
# PREDICTION FUNCTION
# ================================================================


def make_predictions(model_package, new_data_processed):
    """
    Make predictions on new data using the saved model package.

    Args:
        model_package: Dictionary containing model, scaler
        new_data_processed: Numpy array (already preprocessed)

    Returns:
        predictions: Array of predictions
        probabilities: Array of probabilities (for classification)
    """
    # Extract model
    model = model_package['model']

    # Make predictions directly on preprocessed data
    predictions = model.predict(new_data_processed)

    # Get probabilities for classification
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(new_data_processed)

    return predictions, probabilities

# ================================================================
# PROGRESS BAR
# ================================================================
def show_pipeline_progress():
    step = st.session_state.get("step", 1)
    total_steps = 7

    st.markdown("### Pipeline Progress")
    st.progress(step / total_steps)

    steps = ["Upload Data", "Select Target", "Task Detection", "Preprocessing",
             "Model Training", "Hyperparameter Tuning", "Evaluation"]

    # Build HTML table with equal spacing
    html = "<div style='display: flex; justify-content: space-between; text-align: center;'>"

    for i, name in enumerate(steps, 1):
        if i < step:
            html += f"<div style='flex: 1;'>✅ <b>{name}</b></div>"
        elif i == step:
            html += f"<div style='flex: 1;'><span style='color: #02FCF4; font-size: 1.4em;'>●</span> <b style='color: white;'>{name}</b></div>"
        else:
            html += f"<div style='flex: 1;'><span style='color: #6B7280; font-size: 1.4em;'>○</span> <span style='color: #9CA3AF;'>{name}</span></div>"

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ================================================================
# MAIN APP
# ================================================================

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="AutoML System",
        # page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_custom_css()
    init_session_state()

    st.markdown(f"""
    <div class="hero">
        <div style="display:flex; align-items:center; justify-content:center; gap:0.6rem; margin-bottom:0.2rem;">
            <svg width="68" height="68" viewBox="0 0 52 52" xmlns="http://www.w3.org/2000/svg">
                <rect x="2"  y="38" width="7" height="10" fill="#3B82F6" rx="1.5" opacity="0.92"/>
                <rect x="12" y="30" width="7" height="18" fill="#2D8EF5" rx="1.5" opacity="0.92"/>
                <rect x="22" y="22" width="7" height="26" fill="#1A9AEB" rx="1.5" opacity="0.92"/>
                <rect x="32" y="13" width="7" height="35" fill="#0EA5D4" rx="1.5" opacity="0.92"/>
                <rect x="42" y="6"  width="7" height="42" fill="#06B6D4" rx="1.5" opacity="0.92"/>
                <rect x="2"  y="38" width="7" height="2" fill="white" rx="1" opacity="0.4"/>
                <rect x="12" y="30" width="7" height="2" fill="white" rx="1" opacity="0.4"/>
                <rect x="22" y="22" width="7" height="2" fill="white" rx="1" opacity="0.4"/>
                <rect x="32" y="13" width="7" height="2" fill="white" rx="1" opacity="0.4"/>
                <rect x="42" y="6"  width="7" height="2" fill="white" rx="1" opacity="0.4"/>
                <polyline points="5.5,38 15.5,30 25.5,22 35.5,13 45.5,6"
                          fill="none" stroke="#06B6D4" stroke-width="2" opacity="0.95"/>
                <circle cx="5.5"  cy="38" r="2.2" fill="#06B6D4"/>
                <circle cx="15.5" cy="30" r="2.2" fill="#06B6D4"/>
                <circle cx="25.5" cy="22" r="2.2" fill="#06B6D4"/>
                <circle cx="35.5" cy="13" r="2.2" fill="#06B6D4"/>
                <circle cx="45.5" cy="6"  r="2.2" fill="#06B6D4"/>
                <line x1="45.5" y1="6" x2="51" y2="1" stroke="#06B6D4" stroke-width="2"/>
                <polyline points="47,1 51,1 51,5" fill="none" stroke="#06B6D4" stroke-width="2"/>
            </svg>
            <h1>AutoML System</h1>
        </div>
        <p style="margin-top:0.1rem;">From Raw Data to Production-Ready Models</p>
        <p style='font-size: 1rem; color: #3B82F6; margin-top: 0.3rem; letter-spacing: 2px;'>
            Validate · Preprocess · Train · Tune · Deploy
        </p>
    </div>
    """, unsafe_allow_html=True)

    show_pipeline_progress()

    pipeline = st.session_state.pipeline

    # ================================================================
    # SIDEBAR
    # ================================================================
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.session_state.show_visualizations = st.checkbox(
            "Show Visualizations",
            value=st.session_state.show_visualizations
        )
        st.markdown("---")

        st.markdown("## 📋 Pipeline Steps")

        # Navigation
        st.markdown("### 🧭 Navigation")
        current_step = st.session_state.get('step', 1)
        st.info(f"**Step {current_step} of 7**")

        col1, col2 = st.columns(2)
        with col1:
            if current_step > 1:
                st.button("⬅️ Back", on_click=go_to_previous_step, key="sidebar_back")

        with col2:
            if current_step < 7:
                st.button("Next ➡️", on_click=go_to_next_step, key="sidebar_next")

        st.markdown("---")

        # Steps progress
        steps_info = [
            ("1️⃣ Upload Data", st.session_state.data_loaded),
            ("2️⃣ Select Target", st.session_state.target_selected),
            ("3️⃣ Task Detection", st.session_state.task_detected),
            ("4️⃣ Preprocessing", st.session_state.preprocessed),
            ("5️⃣ Model Training", st.session_state.trained),
            ("6️⃣ Hyperparameter Tuning", st.session_state.tuned),
            ("7️⃣ Test Evaluation", st.session_state.test_evaluated)
        ]

        for step_text, completed in steps_info:
            if completed:
                st.success(f"✅ {step_text}")
            elif step_text.startswith(f"{st.session_state.step}️⃣"):
                st.info(f"▶️ {step_text}")
            else:
                st.text(f"⏸️ {step_text}")




    # ================================================================
    # STEP 1: UPLOAD DATA
    # ================================================================
    if st.session_state.step == 1:
        st.markdown("## Step 1: Load Your Data")
        st.markdown("Upload your dataset to begin the AutoML process.")

        # Check if data is already loaded
        if st.session_state.get('data_loaded', False) and pipeline.raw_data is not None:
            st.success("✅ Data already loaded!")
            df = pipeline.raw_data

            st.dataframe(df.head(10), use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicate Rows", df.duplicated().sum())

            # Show stored EDA visualizations
            if 'eda_figures' in st.session_state and st.session_state.eda_figures:
                st.markdown("### 📊 Exploratory Data Analysis")
                for fig_name, fig in st.session_state.eda_figures.items():
                    if fig is not None:
                        st.pyplot(fig)
                        st.markdown("---")

            # Using on_click callback
            st.button(
                "Proceed to Target Selection",
                key="proceed_to_target_loaded",
                on_click=navigate_to_step,
                args=(2,)
            )

        else:
            # File upload section
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Supported formats: CSV, Excel (.xlsx, .xls)"
            )

            if uploaded_file:
                try:
                    with st.spinner("Loading data..."):
                        df = pipeline.loader.load_from_streamlit_upload(uploaded_file)
                        pipeline.raw_data = df

                        validation_result = pipeline.validate_data()
                        is_valid = validation_result['valid']

                        if is_valid:
                            st.success("✅ Data loaded and validated successfully!")
                            st.dataframe(df.head(10), use_container_width=True)

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Rows", df.shape[0])
                            with col2:
                                st.metric("Columns", df.shape[1])
                            with col3:
                                st.metric("Missing Values", df.isnull().sum().sum())
                            with col4:
                                st.metric("Duplicate Rows", df.duplicated().sum())

                            # Show EDA visualizations
                            if 'eda_figures' not in st.session_state or not st.session_state.eda_figures:
                                st.session_state.eda_figures = show_eda_visualizations(df)
                            st.session_state.data_loaded = True

                            # Using on_click callback
                            st.button(
                                "Proceed to Target Selection",
                                key="proceed_to_target_new",
                                on_click=navigate_to_step,
                                args=(2,)
                            )
                        else:
                            st.error("❌ Data validation failed:")
                            for issue in validation_result.get('issues', []):
                                st.error(f"  - {issue}")

                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")

    # ================================================================
    # STEP 2: SELECT TARGET
    # ================================================================
    elif st.session_state.step == 2:
        st.markdown("## Step 2: Select Target Column")
        st.markdown("Choose the column you want to predict.")

        df = pipeline.raw_data
        st.markdown(f"**Dataset:** {df.shape[0]} rows × {df.shape[1]} columns")

        target_column = st.selectbox(
            "Target Column",
            options=df.columns.tolist(),
            help="This is what the model will predict"
        )

        if target_column:
            st.markdown(f"### Target: `{target_column}`")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Values", df[target_column].nunique())
                st.metric("Missing Values", df[target_column].isnull().sum())
            with col2:
                st.metric("Data Type", str(df[target_column].dtype))
                st.metric("Unique Ratio", f"{df[target_column].nunique() / len(df):.2%}")

            st.markdown("**Sample Values:**")
            st.write(df[target_column].value_counts().head(10))

            # Validate target has minimum 3 samples per class
            validator = DataValidator()
            is_valid, msg = validator.validate_target(df[target_column], target_column)

            if not is_valid:
                st.error(f"❌ {msg}")
                st.stop()  # Prevent proceeding
            else:
                # After validation
                st.success("✅ Target column is valid")
                if is_valid:
                    min_samples = df[target_column].value_counts().min()
                    if min_samples < 4:
                        st.warning(f"⚠️ Class with only {min_samples} samples - may not appear in validation/test sets")

            # Handle button click with callback
            if st.button("Confirm Target Selection", key="confirm_target"):
                pipeline.select_target(target_column)
                st.session_state.target_selected = True
                navigate_to_step(3)
                st.rerun()

    # ================================================================
    # STEP 3: TASK DETECTION
    # ================================================================
    elif st.session_state.step == 3:
        st.markdown("## Step 3: Task Type Detection")
        st.markdown("Automatically detect if this is a classification or regression task.")

        if not st.session_state.task_detected:
            with st.spinner("Analyzing target variable..."):
                task_type, confidence = pipeline.detect_task_type()
                st.session_state.task_detected = True
            st.session_state.task_type_display = pipeline.task_type
            st.session_state.task_confidence_display = confidence
        else:
            st.info(f"Task already detected: {pipeline.task_type}")

            # Always show these regardless of detection state
        if st.session_state.task_detected:
            st.markdown(f"### Detected Task: **{pipeline.task_type.upper()}**")
            st.progress(st.session_state.get('task_confidence_display', 1.0))
            st.markdown(f"**Confidence:** {st.session_state.get('task_confidence_display', 1.0):.1%}")

            if st.session_state.show_visualizations:
                if 'step3_target_fig' not in st.session_state:
                    try:
                        eda = EDAGenerator()
                        fig = eda.plot_target_analysis(
                            pipeline.X, pipeline.y,
                            pipeline.target_column, pipeline.task_type
                        )
                        st.session_state.step3_target_fig = fig
                    except Exception as e:
                        st.warning(f"Could not generate target analysis: {e}")

                if 'step3_target_fig' in st.session_state:
                    st.markdown("### 🎯 Target Analysis")
                    st.pyplot(st.session_state.step3_target_fig)

            if st.session_state.get('task_confidence_display', 1.0) < 0.70:
                st.warning("⚠️ Low confidence detection. You may want to override.")

            override = st.selectbox(
                "Override Detection (Optional)",
                options=["Use Auto Detection", "Classification", "Regression"]
            )
            if override != "Use Auto Detection":
                if st.button("Apply Override"):
                    pipeline.detect_task_type(override=override.lower())
                    st.success(f"✅ Task type set to: {override}")
                    st.rerun()
        else:
            st.info(f"Task already detected: {pipeline.task_type}")

        # Using on_click callback
        st.button(
            "Proceed to Preprocessing",
            key="proceed_to_preprocess",
            on_click=navigate_to_step,
            args=(4,)
        )

    # ================================================================
    # STEP 4: PREPROCESSING
    # ================================================================
    elif st.session_state.step == 4:
        st.markdown("## Step 4: Data Preprocessing")
        st.markdown("Automatic data cleaning, encoding, and scaling.")

        if st.button("Start Preprocessing", key="start_preprocessing"):
            with st.spinner("Preprocessing data..."):
                try:

                    # Run preprocessing
                    pipeline.run_preprocessing()
                    st.session_state.preprocessed = True

                    st.success("✅ Preprocessing complete!")

                    # Show preprocessing summary
                    if hasattr(pipeline.preprocessor, 'numeric_features_'):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Numeric Features", len(pipeline.preprocessor.numeric_features_))
                        with col2:
                            st.metric("Categorical Features", len(pipeline.preprocessor.categorical_features_))
                        with col3:
                            st.metric("Low Cardinality (One-Hot)", len(pipeline.preprocessor.low_card_categorical_))
                        with col4:
                            st.metric("High Cardinality (Freq Enc)", len(pipeline.preprocessor.high_card_categorical_))

                    # Show dropped ID columns if any
                    if hasattr(pipeline.preprocessor, 'id_columns_') and pipeline.preprocessor.id_columns_:
                        st.warning(f"⚠️ Dropped ID columns (>{pipeline.preprocessor.id_threshold:.0%} unique): {', '.join(pipeline.preprocessor.id_columns_)}")


                except Exception as e:
                    st.error(f"❌ Error during preprocessing: {e}")
                    st.error(traceback.format_exc())

        # Proceed button with on_click
        if st.button("Proceed to Model Training", key="proceed_to_training"):
            # Split data when proceeding
            pipeline.split_data()
            st.session_state.split_done = True
            navigate_to_step(5)
            st.rerun()

    # ================================================================
    # STEP 5: MODEL TRAINING
    # ================================================================
    elif st.session_state.step == 5:
        st.markdown("## Step 5: Model Training")
        st.markdown("Select models to train and compare.")

        # Show results if already trained
        if st.session_state.get('trained', False) and pipeline.training_results is not None:
            st.success("✅ Models already trained!")
            st.markdown("### Training Results")

            if pipeline.task_type == 'classification':
                trainer = ClassificationTrainer(pipeline.registry)
                comparison_df = trainer.compare_results(pipeline.training_results)
            else:
                trainer = RegressionTrainer(pipeline.registry)
                comparison_df = trainer.compare_results(pipeline.training_results)

            st.dataframe(comparison_df, use_container_width=True)
            st.markdown(f"### Best Model: **{pipeline.best_model_name}**")

        available_models = pipeline.registry.get_models_for_task(pipeline.task_type)

        st.markdown("### Select Models")
        select_all = st.checkbox("Select All Models", value=False)

        if select_all:
            selected_models = available_models
        else:
            selected_models = st.multiselect(
                "Choose models to train",
                options=available_models,
                default=pipeline.registry.get_default_models(pipeline.task_type)
            )

        st.markdown("### Training Options")
        run_cv = st.checkbox("Run Cross-Validation", value=True)

        if selected_models and st.button("Train Models", key="start_training"):
            with st.spinner(f"Training {len(selected_models)} models..."):
                try:
                    pipeline.train_models(
                        selected_models,
                        run_cv=run_cv,
                        show_visualization=st.session_state.show_visualizations
                    )
                    st.session_state.trained = True
                    st.success(f"✅ Trained {len(selected_models)} models!")

                    if pipeline.task_type == 'classification':
                        trainer = ClassificationTrainer(pipeline.registry)
                    else:
                        trainer = RegressionTrainer(pipeline.registry)
                    comparison = trainer.compare_results(pipeline.training_results)
                    st.markdown("### Training Results")
                    st.dataframe(comparison, use_container_width=True)
                    st.markdown(f"### Best Model: **{pipeline.best_model_name}**")

                except Exception as e:
                    st.error(f"❌ Error during training: {e}")
                    st.error(traceback.format_exc())


        if st.session_state.trained:
            if st.session_state.show_visualizations:
                if 'step5_training_fig' not in st.session_state and pipeline.training_results:
                    try:
                        if pipeline.task_type == 'classification':
                            trainer = ClassificationTrainer(pipeline.registry)
                            comparison = trainer.compare_results(pipeline.training_results)
                            fig = plot_classification_results(comparison)
                        else:
                            trainer = RegressionTrainer(pipeline.registry)
                            comparison = trainer.compare_results(pipeline.training_results)
                            fig = plot_regression_results(comparison)
                        st.session_state.step5_training_fig = fig
                    except Exception as e:
                        st.warning(f"Could not display training visualizations: {e}")
                if 'step5_training_fig' in st.session_state:
                    st.markdown("### 📊 Training Visualizations")
                    st.pyplot(st.session_state.step5_training_fig)

            col1, col2 = st.columns(2)
            with col1:
                st.button("Proceed to Tuning (Optional)", key="proceed_to_tuning",
                          on_click=navigate_to_step, args=(6,))
            with col2:
                st.button("Skip to Evaluation", key="skip_to_eval",
                          on_click=navigate_to_step, args=(7,))
        else:
            st.info("Please train models first to proceed.")


    # ================================================================
    # STEP 6: HYPERPARAMETER TUNING (OPTIONAL)
    # ================================================================
    elif st.session_state.step == 6:
        st.markdown("## Step 6: Hyperparameter Tuning (Optional)")
        st.markdown("Optimize model hyperparameters for better performance.")

        if hasattr(pipeline, 'training_results') and pipeline.training_results:
            trained_models = list(pipeline.training_results.keys())

            col1, col2 = st.columns([1, 3])
            with col1:
                select_all_tune = st.checkbox("Select All Models", value=True, key="select_all_tune")

            if select_all_tune:
                models_to_tune = trained_models
                st.info(f"✅ All {len(trained_models)} models selected for tuning")
            else:
                models_to_tune = st.multiselect(
                    "Select models to tune",
                    options=trained_models,
                    default=[pipeline.best_model_name]
                )

            col1, col2 = st.columns(2)
            with col1:
                n_iter = st.slider("Number of Iterations", 10, 50, 20)

            if models_to_tune and st.button("Start Tuning", key="start_tuning"):
                with st.spinner(f"Tuning {len(models_to_tune)} models..."):
                    try:
                        pipeline.tune_models(
                            models_to_tune,
                            n_iter=n_iter,
                            cv=Config.CV_FOLDS,
                            show_visualization=st.session_state.show_visualizations
                        )
                        st.session_state.tuned = True
                        st.success("✅ Tuning complete!")

                    except Exception as e:
                        st.error(f"❌ Error during tuning: {e}")
                        st.error(traceback.format_exc())
            else:
                if not st.session_state.tuned:
                    st.info("Select models to tune and click 'Start Tuning'")

        else:
            st.warning("No trained models found. Please train models first.")

        if st.session_state.tuned:
            tuner = HyperparameterTuner(pipeline.registry, pipeline.task_type)
            comparison = tuner.compare_results(pipeline.tuning_results)
            st.markdown("### Tuning Results")
            st.dataframe(comparison, use_container_width=True)
            st.markdown(f"### Best Tuned Model: **{pipeline.best_model_name}**")

            if st.session_state.show_visualizations:
                if 'step6_tuning_fig' not in st.session_state:
                    try:
                        if pipeline.task_type == 'classification':
                            fig = plot_tuned_classification_results(comparison)
                        else:
                            fig = plot_tuned_regression_results(comparison)
                        st.session_state.step6_tuning_fig = fig
                    except Exception as e:
                        st.warning(f"Could not display tuning visualizations: {e}")
                if 'step6_tuning_fig' in st.session_state:
                    st.markdown("### 🔧 Tuning Visualizations")
                    st.pyplot(st.session_state.step6_tuning_fig)

        st.button(
            "Proceed to Evaluation",
            key="proceed_to_eval",
            on_click=navigate_to_step,
            args=(7,)
        )

        # ================================================================
        # STEP 7: TEST EVALUATION & DEPLOYMENT
        # ================================================================
    elif st.session_state.step == 7:
        st.markdown("## Step 7: Final Evaluation & Deployment")
        st.markdown("Evaluate model on test set, save it, and make predictions on new data.")

        # Evaluation Section
        st.markdown("### 🎯 Test Set Evaluation")

        if not st.session_state.test_evaluated:
            # Show evaluate button only if not already evaluated
            if st.button("Evaluate on Test Set", key="evaluate_test"):
                with st.spinner("Evaluating on test set..."):
                    try:
                        pipeline.evaluate_test(show_visualization=st.session_state.show_visualizations)
                        st.session_state.test_evaluated = True
                        st.success("✅ Evaluation complete!")

                        # Force a rerun to show results
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error during evaluation: {e}")
                        st.error(traceback.format_exc())
        else:
            st.success("✅ Test set already evaluated!")

            # Visualization
            if st.session_state.show_visualizations:
                if 'step7_test_fig' not in st.session_state:
                    try:
                        if pipeline.task_type == 'classification':
                            fig = plot_test_classification(pipeline.test_results)
                        else:
                            fig = plot_test_regression(pipeline.test_results)
                        st.session_state.step7_test_fig = fig
                    except Exception as e:
                        st.warning(f"Could not display test visualizations: {e}")

                if 'step7_test_fig' in st.session_state:
                    st.markdown("### 📊 Test Set Visualizations")
                    st.pyplot(st.session_state.step7_test_fig)

        if st.session_state.test_evaluated and hasattr(pipeline, 'test_results'):
            st.markdown("### Test Set Results")
            st.dataframe(pipeline.test_results, use_container_width=True)

        st.markdown("---")

        # Model Saving Section
        st.markdown("### 💾 Save Model")

        if st.button("Save Best Model", key="save_model"):
            with st.spinner("Saving model..."):
                try:
                    model_path = pipeline.save_best_model()
                    st.session_state.model_saved_path = model_path
                    st.success(f"✅ Model saved successfully!")
                except Exception as e:
                    st.error(f"❌ Error saving model: {e}")

        # Download saved model - check if path exists
        if st.session_state.get('model_saved_path') and st.session_state.model_saved_path is not None:
            try:
                with open(st.session_state.model_saved_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Trained Model",
                        data=f,
                        file_name=Path(st.session_state.model_saved_path).name,
                        mime="application/octet-stream",
                        key="download_model"
                    )
            except Exception as e:
                st.error(f"Error loading model for download: {e}")

        st.markdown("---")

        # Prediction Section
        st.markdown("### 🔮 Make Predictions on New Data")
        st.markdown("Upload new data (without target column) to get predictions.")

        if st.session_state.get('model_saved_path') and st.session_state.model_saved_path is not None:
            prediction_file = st.file_uploader(
                "Upload new data for predictions",
                type=['csv', 'xlsx', 'xls'],
                help="Upload data with the same features (no target column)",
                key="prediction_upload"
            )

            if prediction_file:
                try:
                    # Load new data
                    new_data = pipeline.loader.load_from_streamlit_upload(prediction_file)
                    st.success(f"✅ Loaded {len(new_data)} rows for prediction")

                    # Show preview
                    st.markdown("**Data Preview:**")
                    st.dataframe(new_data.head(), use_container_width=True)

                    # Validate data
                    st.markdown("**Validating data...**")

                    # Check for target column
                    if pipeline.target_column in new_data.columns:
                        st.warning(f"⚠️ Target column '{pipeline.target_column}' found in data. It will be removed.")
                        new_data = new_data.drop(columns=[pipeline.target_column])

                    # Make predictions button
                    if st.button("Generate Predictions", key="make_predictions"):
                        with st.spinner("Making predictions..."):
                            try:
                                # Load model
                                model_package = joblib.load(st.session_state.model_saved_path)

                                # Use the SAVED preprocessor from the package, not the live session one
                                saved_preprocessor = model_package.get('preprocessor')
                                if saved_preprocessor is not None:
                                    new_data_processed = saved_preprocessor.transform(new_data)
                                else:
                                    # fallback for old saved models that don't have preprocessor
                                    new_data_processed = pipeline.preprocessor.transform(new_data)

                                # Make predictions - pass preprocessed data
                                predictions, probabilities = make_predictions(model_package, new_data_processed)

                                # Create results dataframe using ORIGINAL data
                                results_df = new_data.copy()
                                results_df['Prediction'] = predictions

                                # Add probabilities for classification
                                if probabilities is not None:
                                    for i in range(probabilities.shape[1]):
                                        results_df[f'Probability_Class_{i}'] = probabilities[:, i]

                                st.success(f"✅ Generated {len(predictions)} predictions!")

                                # Show results preview
                                st.markdown("### Prediction Results Preview")
                                st.dataframe(results_df.head(20), use_container_width=True)

                                # Statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Predictions", len(predictions))
                                with col2:
                                    if pipeline.task_type == 'classification':
                                        st.metric("Unique Predictions", len(np.unique(predictions)))
                                    else:
                                        st.metric("Mean Prediction", f"{np.mean(predictions):.2f}")
                                with col3:
                                    if pipeline.task_type == 'regression':
                                        st.metric("Std Prediction", f"{np.std(predictions):.2f}")

                                # Download predictions
                                st.markdown("### 📥 Download Predictions")

                                col1, col2 = st.columns(2)

                                with col1:
                                    # CSV download
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download as CSV",
                                        data=csv,
                                        file_name="predictions.csv",
                                        mime="text/csv",
                                        key="download_csv"
                                    )

                                with col2:
                                    # Excel download
                                    output = BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        results_df.to_excel(writer, index=False, sheet_name='Predictions')
                                    excel_data = output.getvalue()

                                    st.download_button(
                                        label="Download as Excel",
                                        data=excel_data,
                                        file_name="predictions.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="download_excel"
                                    )

                            except Exception as e:
                                st.error(f"❌ Error making predictions: {e}")
                                st.error(traceback.format_exc())

                except Exception as e:
                    st.error(f"❌ Error loading prediction data: {e}")
        else:
            st.info("Please save the model first before making predictions.")

        st.markdown("---")

        # Start new pipeline
        if st.button("Start New Pipeline", key="restart_pipeline"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()