"""
Configuration settings for Student Lifestyle Analytics System
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
TRAINING_DIR = PROJECT_ROOT / 'training'

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# PostgreSQL Database Settings
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'Student Life Analytics System'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'sslmode': 'disable'
}

# ============================================================================
# MODEL TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 20,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'train_test_split': 0.8,
    'random_seed': 42
}

# ============================================================================
# MODELS TO TRAIN
# ============================================================================

MODELS_TO_TRAIN = [
    'mlp',           # Multi-Layer Perceptron
    'dnn',           # Deep Neural Network
    'dropout',       # Dropout Regularized
    'batch_norm',    # Batch Normalized
    'hybrid'         # Hybrid Advanced Model
]

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

# Input features for the models
FEATURE_COLUMNS = [
    'sleep_hours',
    'sleep_consistency',
    'exercise',
    'tired_during_class',
    'study_hours',
    'attendance',
    'assignment_submission',
    'concentration',
    'screen_time',
    'late_phone',
    'social_media_usage',
    'stress_level',
    'overwhelmed',
    'time_management',
    'productivity_score',
    'gpa'
]

# Target variables
TARGET_COLUMNS = [
    'lifestyle_score',
    'burnout_risk'
]

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# API & EXTERNAL SERVICES
# ============================================================================

# Google Sheets Configuration
GOOGLE_SHEETS_CREDENTIALS = 'etl/Credentials.json'
GOOGLE_SHEET_NAME = 'student_lifestyle_raw_data'

# MCP Agent Settings
MCP_CONFIG = {
    'model': 'gpt-4',
    'temperature': 0.7,
    'max_tokens': 2000
}

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

STREAMLIT_CONFIG = {
    'page_title': 'Student Lifestyle Analytics',
    'page_icon': '📊',
    'layout': 'wide',
    'theme': 'light'
}

# ============================================================================
# BENCHMARK THRESHOLDS
# ============================================================================

HEALTH_THRESHOLDS = {
    'good_sleep': 7.0,          # hours
    'good_study': 4.0,          # hours
    'good_exercise': 3.0,       # times per week
    'bad_screen_time': 7.0,     # hours
    'good_gpa': 3.0,            # CGPA
    'high_stress': 7.0,         # out of 10
    'high_burnout': 5.0         # out of 10
}

# ============================================================================
# PREDICTION RANGES
# ============================================================================

PREDICTION_RANGES = {
    'lifestyle_score': (0, 10),
    'burnout_risk': (0, 10),
    'digital_addiction_score': (0, 10),
    'productivity_index': (0, 10)
}

# ============================================================================
# EXPORT SETTINGS
# ============================================================================

EXPORT_FORMATS = ['csv', 'json', 'xlsx']
MODEL_SAVE_FORMAT = '.pth'  # PyTorch
METRICS_SAVE_FORMAT = '.json'
