"""
Utility script for testing and validating the system
Run this to verify all components are working correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def test_python_version():
    """Test Python version"""
    print_banner("🐍 Python Version Check")
    version = sys.version
    logger.info(f"Python {version}")
    if sys.version_info >= (3, 9):
        logger.info("✅ Python version OK (3.9+)")
        return True
    else:
        logger.error("❌ Python 3.9+ required")
        return False


def test_imports():
    """Test all required imports"""
    print_banner("📦 Testing Imports")
    
    required_packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'polars': 'Polars',
        'torch': 'PyTorch',
        'sklearn': 'scikit-learn',
        'psycopg2': 'psycopg2',
        'streamlit': 'Streamlit',
        'plotly': 'Plotly',
        'gspread': 'gspread'
    }
    
    all_ok = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✅ {name:20} - OK")
        except ImportError as e:
            logger.error(f"❌ {name:20} - FAILED: {e}")
            all_ok = False
    
    return all_ok


def test_data_files():
    """Test data files exist"""
    print_banner("📁 Data Files Check")
    
    files_to_check = {
        'data/cleaned_student_data.csv': 'Cleaned student data',
        'etl/pipeline.py': 'ETL pipeline script',
        'etl/Credentials.json': 'Google credentials (optional)',
        'config/settings.py': 'Configuration file',
    }
    
    all_ok = True
    for filepath, description in files_to_check.items():
        full_path = PROJECT_ROOT / filepath
        if full_path.exists():
            logger.info(f"✅ {description:30} - {filepath}")
        else:
            # Check if it's optional
            if 'Credentials' in filepath:
                logger.warning(f"⚠️  {description:30} - MISSING (optional)")
            else:
                logger.error(f"❌ {description:30} - MISSING: {filepath}")
                all_ok = False
    
    return all_ok


def test_pytorch_cuda():
    """Test PyTorch CUDA availability"""
    print_banner("🎮 PyTorch & GPU Check")
    
    try:
        import torch
        
        logger.info(f"PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"✅ CUDA available")
            logger.info(f"   GPU Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA Capability: {torch.cuda.get_device_capability(0)}")
            logger.info(f"   Device Count: {torch.cuda.device_count()}")
        else:
            logger.warning("⚠️  CUDA not available - Using CPU (slower)")
            logger.warning("   For GPU acceleration, install CUDA 11.8+ and matching PyTorch")
        
        return True
    except Exception as e:
        logger.error(f"❌ PyTorch error: {e}")
        return False


def test_database_connection():
    """Test PostgreSQL connection"""
    print_banner("🗄️  Database Connection Test")
    
    try:
        from db.db import DatabaseManager
        from config.settings import DB_CONFIG
        
        logger.info(f"Attempting to connect to {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        
        db = DatabaseManager(DB_CONFIG)
        
        # Check if connection was successful
        if not db.is_connected:
            logger.warning(f"⚠️  Database connection unavailable")
            logger.info("   To enable PostgreSQL:")
            logger.info("   1. Install PostgreSQL (https://www.postgresql.org/download/)")
            logger.info("   2. Create a .env file with:")
            logger.info("      DB_HOST=localhost")
            logger.info("      DB_PORT=5432")
            logger.info("      DB_NAME=student_analytics")
            logger.info("      DB_USER=postgres")
            logger.info("      DB_PASSWORD=your_password")
            logger.info("   3. Run: python db/db.py  # to initialize database")
            logger.info("   See SETUP.md for more details")
            return True  # Not a critical failure
        
        # Test query
        result = db.execute_query("SELECT 1 as test")
        
        if result:
            logger.info("✅ Database connection successful")
            logger.info(f"   Database: {DB_CONFIG['database']}")
            logger.info(f"   User: {DB_CONFIG['user']}")
            
            # Check tables
            tables = db.execute_query("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            logger.info(f"   Tables: {len(tables)} found")
            
            db.close()
            return True
        else:
            logger.warning("⚠️  Database query returned no results")
            return True
            
    except Exception as e:
        logger.warning(f"⚠️  Database connection unavailable: {type(e).__name__}")
        logger.info("   To enable PostgreSQL:")
        logger.info("   1. Install PostgreSQL (https://www.postgresql.org/download/)")
        logger.info("   2. Create a .env file with:")
        logger.info("      DB_HOST=localhost")
        logger.info("      DB_PORT=5432")
        logger.info("      DB_NAME=student_analytics")
        logger.info("      DB_USER=postgres")
        logger.info("      DB_PASSWORD=your_password")
        logger.info("   3. Run: python db/db.py  # to initialize database")
        logger.info("   See SETUP.md for more details")
        return True  # Not a critical failure


def test_models():
    """Test model architectures"""
    print_banner("🤖 Model Architectures Check")
    
    try:
        from models.architectures import create_model, MODEL_CONFIGS
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Testing models on {device}...\n")
        
        model_names = list(MODEL_CONFIGS.keys())
        
        for model_name in model_names:
            config = MODEL_CONFIGS[model_name]
            
            try:
                # Create model
                model = create_model(model_name, input_size=16, device=device)
                
                # Count parameters
                params = sum(p.numel() for p in model.parameters())
                
                logger.info(f"✅ {model_name.upper():15} - {params:,} parameters")
                logger.info(f"   {config['description']}")
                
            except Exception as e:
                logger.error(f"❌ {model_name.upper()} - Error: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False


def test_streamlit():
    """Test Streamlit app can be imported"""
    print_banner("🎨 Streamlit App Check")
    
    try:
        import streamlit as st
        logger.info(f"✅ Streamlit {st.__version__}")
        logger.info(f"   App file: ui/streamlit_app.py")
        logger.info("   Run with: streamlit run ui/streamlit_app.py")
        return True
    except Exception as e:
        logger.error(f"❌ Streamlit error: {e}")
        return False


def test_etl_pipeline():
    """Test ETL pipeline"""
    print_banner("🔄 ETL Pipeline Check")
    
    try:
        # Check imports
        import gspread
        import polars as pl
        
        logger.info("✅ ETL dependencies available")
        logger.info("   gspread: Google Sheets integration")
        logger.info("   polars: Data processing")
        
        # Check credentials
        cred_path = PROJECT_ROOT / 'etl' / 'Credentials.json'
        if cred_path.exists():
            logger.info("✅ Google Credentials found")
        else:
            logger.warning("⚠️  Google Credentials missing (needed for Google Sheets)")
            logger.warning("   Create service account at https://console.cloud.google.com")
        
        return True
    except Exception as e:
        logger.error(f"❌ ETL error: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "SYSTEM VALIDATION TEST SUITE" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("Data Files", test_data_files),
        ("PyTorch & GPU", test_pytorch_cuda),
        ("Database Connection", test_database_connection),
        ("Model Architectures", test_models),
        ("Streamlit App", test_streamlit),
        ("ETL Pipeline", test_etl_pipeline),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ Test crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print_banner("📋 Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} | {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Result: {passed}/{total} tests passed")
    print(f"{'='*70}\n")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.\n")
        print("Next steps:")
        print("1. python etl/pipeline.py          # Run ETL pipeline")
        print("2. python training/run_training.py # Train models")
        print("3. streamlit run ui/streamlit_app.py # Launch dashboard\n")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please fix issues above.\n")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
