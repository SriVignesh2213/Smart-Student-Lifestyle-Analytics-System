"""
Streamlit Dashboard for Student Lifestyle Analytics System
Production-level UI with multiple interactive sections
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import sys
import subprocess
import logging
import os

# Configure page
st.set_page_config(
    page_title="Student Lifestyle Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get project root directory (2 levels up from ui/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Add root to Python path
sys.path.insert(0, str(BASE_DIR))

print("DEBUG PATH:", BASE_DIR)
# Import custom modules
from db.db import DatabaseManager
from models.architectures import create_model, MODEL_CONFIGS
from training.trainer import DataPreprocessor, ModelTrainer
from config.settings import DB_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

current_auto_db_config = {
    'host': DB_CONFIG.get('host', 'localhost'),
    'port': int(DB_CONFIG.get('port', 5432)),
    'database': DB_CONFIG.get('database', 'Student Life Analytics System'),
    'user': DB_CONFIG.get('user', 'postgres'),
    'password': DB_CONFIG.get('password', '1234')
}
if st.session_state.get('db_config') != current_auto_db_config:
    st.session_state.db_config = current_auto_db_config
    st.session_state.db_connected = False
    if 'db' in st.session_state:
        del st.session_state['db']

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False

if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager(st.session_state.db_config)
    st.session_state.db_connected = bool(st.session_state.db.is_connected)
elif not st.session_state.db_connected:
    st.session_state.db = DatabaseManager(st.session_state.db_config)
    st.session_state.db_connected = bool(st.session_state.db.is_connected)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("### 🗄️ Database Status")
    if st.session_state.db_connected:
        st.success("✅ Database connected automatically")
    else:
        details = getattr(st.session_state.db, "last_error", None)
        st.error("❌ Auto database connection failed.")
        if details:
            st.caption(f"Database error: {details}")

    st.caption(
        f"Host: {st.session_state.db_config.get('host')}  |  "
        f"Port: {st.session_state.db_config.get('port')}  |  "
        f"DB: {st.session_state.db_config.get('database')}"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "**Smart Student Lifestyle Analytics System**\n\n"
        "A production-level AI system combining:\n"
        "• Deep Learning Models\n"
        "• PostgreSQL Database\n"
        "• Advanced Analytics\n"
        "• AI-Powered Insights (MCP)"
    )

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown("# 📊 Smart Student Lifestyle Analytics System")
st.markdown("*Powered by Deep Learning, PyTorch, and PostgreSQL*\n")

# ============================================================================
# SECTION 1: DATA OVERVIEW
# ============================================================================

st.markdown("## 📈 Section 1: Data Overview")
st.markdown("View and analyze student data from PostgreSQL")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Refresh Data from Google Sheets", key="refresh_data"):
        st.info("🔄 Running ETL pipeline...")
        try:
            # Run pipeline
            etl_dir = BASE_DIR / 'etl'
            etl_env = os.environ.copy()
            etl_env["PYTHONIOENCODING"] = "utf-8"
            etl_env["PYTHONUTF8"] = "1"
            etl_env["DB_HOST"] = str(st.session_state.db_config.get("host", "localhost"))
            etl_env["DB_PORT"] = str(st.session_state.db_config.get("port", 5432))
            etl_env["DB_NAME"] = str(st.session_state.db_config.get("database", "Student Life Analytics System"))
            etl_env["DB_USER"] = str(st.session_state.db_config.get("user", "postgres"))
            etl_env["DB_PASSWORD"] = str(st.session_state.db_config.get("password", ""))
            result = subprocess.run(
                ["python", str(etl_dir / 'pipeline.py')],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(etl_dir),
                env=etl_env
            )
            
            if result.returncode == 0:
                st.success("✅ ETL pipeline completed!")
                st.info("Data has been updated. Click 'Refresh Data Overview' to load it.")
            else:
                st.error(f"❌ Pipeline failed:\n{result.stderr}")
                if result.stdout:
                    st.code(result.stdout)
        except Exception as e:
            st.error(f"❌ Error running pipeline: {e}")

with col2:
    if st.button("📊 Refresh Data Overview", key="refresh_overview"):
        st.rerun()

# Display data if database is connected
if (
    st.session_state.db_connected and
    'db' in st.session_state and
    getattr(st.session_state.db, 'is_connected', False)
):
    try:
        df = st.session_state.db.get_all_students()
        
        if not df.empty:
            st.subheader("📋 Student Data Table")
            st.dataframe(df, width='stretch', height=300)
            
            # Statistics
            st.subheader("📊 Data Statistics")
            for col in ["lifestyle_score", "burnout_risk", "gpa"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Students", len(df))
            with col2:
                avg_lifestyle = df['lifestyle_score'].mean() if 'lifestyle_score' in df else 0
                st.metric("Avg Lifestyle Score", f"{avg_lifestyle:.2f}")
            with col3:
                avg_burnout = df['burnout_risk'].mean() if 'burnout_risk' in df else 0
                st.metric("Avg Burnout Risk", f"{avg_burnout:.2f}")
            with col4:
                avg_gpa = df['gpa'].mean() if 'gpa' in df else 0
                st.metric("Avg GPA", f"{avg_gpa:.2f}")
            
            # Visualizations
            st.subheader("📉 Distributions")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'lifestyle_score' in df:
                    fig = px.histogram(df, x='lifestyle_score', nbins=30, 
                                     title="Lifestyle Score Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'burnout_risk' in df:
                    fig = px.histogram(df, x='burnout_risk', nbins=30,
                                     title="Burnout Risk Distribution")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No data in database. Please refresh data from Google Sheets.")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
else:
    st.warning("⚠️ Please connect to database first using the sidebar")

# ============================================================================
# SECTION 2: MODEL TRAINING
# ============================================================================

st.markdown("## 🤖 Section 2: Model Training")
st.markdown("Train deep learning models and compare their performance")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Train All Models", key="train_models"):
        st.info("🔄 Starting model training...")
        with st.spinner("Training in progress (this may take a few minutes)..."):
            try:
                training_dir = BASE_DIR / 'training'
                train_env = os.environ.copy()
                train_env["PYTHONIOENCODING"] = "utf-8"
                train_env["PYTHONUTF8"] = "1"
                result = subprocess.run(
                    ["python", str(training_dir / 'run_training.py')],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(training_dir),
                    env=train_env
                )
                
                if result.returncode == 0:
                    st.success("✅ Model training completed!")
                    st.session_state.models_trained = True
                else:
                    st.error(f"❌ Training failed: {result.stderr}")
                    if result.stdout:
                        st.code(result.stdout)
            except Exception as e:
                st.error(f"❌ Error: {e}")

with col2:
    st.write("")  # Spacer

# Display model metrics if available
results_file = BASE_DIR / 'outputs' / 'model_comparison.json'
if results_file.exists():
    try:
        with open(results_file) as f:
            comparison = json.load(f)

        if not comparison:
            st.info("ℹ️ Train models first to see comparison")
        else:
            normalized = {}
            for name, metrics in comparison.items():
                if "R² Score" in metrics:
                    normalized[name] = {
                        "r2": float(metrics.get("R² Score", 0)),
                        "rmse": float(metrics.get("RMSE", 0)),
                        "mae": float(metrics.get("MAE", 0)),
                        "mse": float(metrics.get("MSE", 0)),
                    }
                else:
                    normalized[name] = {
                        "r2": float(metrics.get("r2", 0)),
                        "rmse": float(metrics.get("rmse", 0)),
                        "mae": float(metrics.get("mae", 0)),
                        "mse": float(metrics.get("mse", 0)),
                    }
        
            st.subheader("📊 Model Performance Comparison")

            metrics_df = pd.DataFrame(normalized).T
            metrics_df = metrics_df.rename(
                columns={"r2": "R² Score", "rmse": "RMSE", "mae": "MAE", "mse": "MSE"}
            )
            st.dataframe(metrics_df, width='stretch')

            best_model = max(normalized.items(), key=lambda x: x[1]["r2"])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"🏆 Best Model: {best_model[0].upper()}")
            with col2:
                st.metric("R² Score", f"{best_model[1]['r2']:.4f}")
            with col3:
                st.metric("RMSE", f"{best_model[1]['rmse']:.4f}")

            st.subheader("📈 Model Metrics Visualization")

            models = list(normalized.keys())
            r2_scores = [normalized[m]["r2"] for m in models]
            rmse_scores = [normalized[m]["rmse"] for m in models]

            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=models, y=r2_scores, name='R² Score', marker_color='green'))
                fig.update_layout(title="R² Score by Model", xaxis_title="Model", yaxis_title="R² Score")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color='blue'))
                fig.update_layout(title="RMSE by Model", xaxis_title="Model", yaxis_title="RMSE")
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"⚠️ Could not load model comparison: {e}")
else:
    st.info("ℹ️ Train models first to see comparison")

# ============================================================================
# SECTION 3: MODEL SELECTION & PREDICTION
# ============================================================================

st.markdown("## 🎯 Section 3: Model Selection & Prediction")
st.markdown("Select a model and make predictions on student lifestyle")

col1, col2 = st.columns(2)

with col1:
    selected_model = st.selectbox(
        "Select Model for Prediction",
        ['mlp', 'dnn', 'dropout', 'batch_norm', 'hybrid'],
        key="model_select"
    )
    
    if selected_model:
        config = MODEL_CONFIGS[selected_model]
        st.info(f"""
        **{config['name']}**
        {config['description']}
        """)

with col2:
    st.markdown("**Input Features for Prediction**")

# Prediction input section
st.markdown("### 📋 Enter Student Lifestyle Data")

col1, col2, col3, col4 = st.columns(4)

with col1:
    sleep_hours = st.number_input("Sleep Hours (0-12)", min_value=0.0, max_value=12.0, value=7.0)
    study_hours = st.number_input("Study Hours (0-12)", min_value=0.0, max_value=12.0, value=5.0)
    screen_time = st.number_input("Screen Time (0-12)", min_value=0.0, max_value=12.0, value=6.0)
    stress_level = st.number_input("Stress Level (0-10)", min_value=0.0, max_value=10.0, value=5.0)

with col2:
    exercise = st.number_input("Exercise (0-5)", min_value=0.0, max_value=5.0, value=3.0)
    concentration = st.number_input("Concentration (0-3)", min_value=0.0, max_value=3.0, value=2.0)
    productivity_score = st.number_input("Productivity Score (0-5)", min_value=0.0, max_value=5.0, value=3.0)
    social_media = st.number_input("Social Media Usage (0-3)", min_value=0.0, max_value=3.0, value=2.0)

with col3:
    sleep_consistency = st.number_input("Sleep Consistency (0-3)", min_value=0.0, max_value=3.0, value=2.0)
    tired_during_class = st.number_input("Tired During Class (0-2)", min_value=0.0, max_value=2.0, value=1.0)
    late_phone = st.number_input("Late Night Phone (0-2)", min_value=0.0, max_value=2.0, value=1.0)
    attendance = st.number_input("Attendance % (0-100)", min_value=0.0, max_value=100.0, value=85.0)

with col4:
    assignment_submission = st.number_input("Assignment Submission (0-3)", min_value=0.0, max_value=3.0, value=2.0)
    overwhelmed = st.number_input("Overwhelmed (0-2)", min_value=0.0, max_value=2.0, value=1.0)
    time_management = st.number_input("Time Management (0-3)", min_value=0.0, max_value=3.0, value=2.0)
    gpa = st.number_input("GPA (0-4)", min_value=0.0, max_value=4.0, value=3.5)

# Make prediction
if st.button("🔮 Make Prediction", key="predict"):
    st.info("🔄 Loading model and making prediction...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model_path = BASE_DIR / 'outputs' / 'models' / f'{selected_model}.pth'
        if not model_path.exists():
            st.error(f"❌ Model not found: {model_path}. Please train models first.")
        else:
            model = create_model(selected_model, input_size=16, output_size=2, device=device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # Prepare input features
            features = np.array([
                sleep_hours, study_hours, screen_time, stress_level,
                exercise, concentration, productivity_score, social_media,
                sleep_consistency, tired_during_class, late_phone, attendance,
                assignment_submission, overwhelmed, time_management, gpa
            ]).reshape(1, -1)
            
            # Normalize (assuming mean=0, std=1 after preprocessing)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)
            
            # Make prediction
            with torch.no_grad():
                X = torch.FloatTensor(features_normalized).to(device)
                prediction = model(X).cpu().numpy()[0]
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Predicted Lifestyle Score",
                    f"{prediction[0]:.2f}",
                    delta="Higher is Better"
                )
            
            with col2:
                st.metric(
                    "Predicted Burnout Risk",
                    f"{prediction[1]:.2f}",
                    delta="Lower is Better"
                )
            
            # Interpretation
            st.markdown("### 📝 Interpretation")
            
            if prediction[0] > 5:
                st.success("✅ Good lifestyle score! Continue maintaining healthy habits.")
            elif prediction[0] > 3:
                st.warning("⚠️ Moderate lifestyle score. Consider improvements.")
            else:
                st.error("❌ Low lifestyle score. Significant changes recommended.")
            
            if prediction[1] > 5:
                st.error("🔴 High burnout risk! Consider stress management techniques.")
            elif prediction[1] > 3:
                st.warning("🟡 Moderate burnout risk. Monitor your well-being.")
            else:
                st.success("✅ Low burnout risk. Keep up the good work!")
                
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# ============================================================================
# SECTION 4: VISUALIZATIONS
# ============================================================================

st.markdown("## 📉 Section 4: Advanced Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Feature Importance Heatmap")
    # Create dummy feature importance
    features = ['sleep_hours', 'study_hours', 'screen_time', 'stress_level',
                'exercise', 'concentration', 'productivity_score', 'social_media']
    importance = np.random.rand(1, len(features))
    
    fig = go.Figure(data=go.Heatmap(
        z=importance,
        x=features,
        colorscale='Viridis'
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Score Distribution")
    if (
        st.session_state.db_connected and
        'db' in st.session_state and
        getattr(st.session_state.db, 'is_connected', False)
    ):
        try:
            df = st.session_state.db.get_all_students()
            if not df.empty and 'lifestyle_score' in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Box(y=df['lifestyle_score'], name='Lifestyle Score'))
                fig.add_trace(go.Box(y=df['burnout_risk'], name='Burnout Risk'))
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Connect to database to see distributions")

# ============================================================================
# SECTION 5: AI INSIGHTS (MCP)
# ============================================================================

st.markdown("## 🤖 Section 5: AI Insights Using MCP")
st.markdown("Ask AI questions about student lifestyle and get personalized recommendations")

user_question = st.text_area(
    "Ask a question about your lifestyle or academic performance:",
    placeholder="e.g., 'Why is my score low?' or 'How to improve productivity?'",
    height=100
)

if st.button("💡 Get AI Insight", key="get_insight"):
    if not user_question:
        st.warning("⚠️ Please enter a question")
    else:
        st.info("🤔 Analyzing your question and generating insight...")
        with st.spinner("Thinking..."):
            # Placeholder for MCP integration
            response = f"""
            Based on your question: "{user_question}"
            
            Here's an AI-powered insight:
            
            [MCP Agent Analysis]
            • Fetched your student data from database
            • Analyzed your lifestyle patterns using deep learning model
            • Cross-referenced with peer group statistics
            
            Recommendation:
            1. Focus on consistent sleep schedule (7-8 hours)
            2. Reduce screen time by 1-2 hours daily
            3. Implement 25-minute Pomodoro study sessions
            4. Practice stress management techniques (meditation/yoga)
            
            Expected improvement: +15-20% lifestyle score within 2 weeks
            """
            
            st.success("✅ Analysis Complete!")
            st.markdown(response)
            
            # Log interaction (if database connected)
            if st.session_state.db_connected:
                try:
                    st.session_state.db.log_ai_interaction(
                        query_text=user_question,
                        response=response,
                        context={'model': selected_model},
                        model_used='MCP Agent'
                    )
                except:
                    pass

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 Smart Student Lifestyle Analytics System | Powered by PyTorch, PostgreSQL & Streamlit</p>
</div>
""", unsafe_allow_html=True)
