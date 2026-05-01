"""
Main Training Pipeline
Fully Fixed & Production-Safe Version
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import polars as pl
import torch
import numpy as np

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architectures import create_model, MODEL_CONFIGS
from training.trainer import ModelTrainer, DataPreprocessor, create_data_loaders

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# PIPELINE
# =========================
class TrainingPipeline:

    def __init__(self, data_path, output_dir='outputs',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.device = device

        self.model_dir = self.output_dir / "models"
        self.metrics_dir = self.output_dir / "metrics"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🚀 Using device: {self.device}")
        logger.info(f"📁 Output directory: {self.output_dir}")

    # =========================
    def load_and_preprocess_data(self):

        logger.info(f"📂 Loading data from {self.data_path}")

        df = pl.read_csv(self.data_path)

        if df.shape[0] == 0:
            raise ValueError("❌ Dataset is empty!")

        logger.info(f"✅ Loaded {len(df)} records")

        preprocessor = DataPreprocessor()

        X, feature_names = preprocessor.prepare_features(df)
        y = preprocessor.prepare_targets(df)

        logger.info(f"✅ Features shape: {X.shape}")
        logger.info(f"✅ Targets shape: {y.shape}")

        self.feature_names = feature_names

        return X, y

    # =========================
    def train_all_models(self, X, y, epochs=100, batch_size=32):

        train_loader, val_loader = create_data_loaders(X, y, batch_size)

        model_types = ['mlp', 'dnn', 'dropout', 'batch_norm', 'hybrid']
        results = {}

        logger.info("\n" + "="*60)
        logger.info("🔄 TRAINING ALL MODELS")
        logger.info("="*60)

        for model_type in model_types:

            logger.info("\n" + "="*60)
            logger.info(f"Training {model_type.upper()}")
            logger.info("="*60)

            try:
                model = create_model(
                    model_type=model_type,
                    input_size=len(self.feature_names),
                    output_size=2,
                    device=self.device
                )

                trainer = ModelTrainer(
                    model=model,
                    model_name=model_type.upper(),
                    device=self.device
                )

                history = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    patience=20
                )

                val_loss, metrics = trainer.validate(val_loader)

                # Save model
                model_path = self.model_dir / f"{model_type}.pth"
                trainer.save_model(str(model_path))

                # Store results
                results[model_type] = {
                    "metrics": metrics,
                    "model_path": str(model_path),
                    "trained_at": datetime.now().isoformat()
                }

                logger.info(f"✅ {model_type.upper()} DONE | R2: {metrics['r2']:.4f}")

            except Exception as e:
                logger.error(f"❌ Error training {model_type}: {e}")

        return results

    # =========================
    def save_results(self, results):

        if not results:
            logger.error("❌ No models were successfully trained.")
            return

        # Save full results
        results_path = self.output_dir / "all_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"✅ Saved results to {results_path}")

        # Create comparison
        comparison = {}
        for name, data in results.items():
            m = data["metrics"]
            comparison[name] = {
                "r2": m.get("r2", 0),
                "rmse": m.get("rmse", 0),
                "mae": m.get("mae", 0),
                "mse": m.get("mse", 0),
            }

        comparison_path = self.output_dir / "model_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"✅ Saved comparison to {comparison_path}")

        self.print_summary(comparison)

    # =========================
    def print_summary(self, comparison):

        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)

        if not comparison:
            print("No results available")
            return

        print(f"{'Model':<15}{'R2':<10}{'RMSE':<10}{'MAE':<10}{'MSE':<10}")
        print("-"*70)

        for model, m in comparison.items():
            print(
                f"{model:<15}"
                f"{m['r2']:<10.4f}"
                f"{m['rmse']:<10.4f}"
                f"{m['mae']:<10.4f}"
                f"{m['mse']:<10.4f}"
            )

        best = max(comparison.items(), key=lambda x: x[1]['r2'])

        print(f"\nBest Model: {best[0].upper()} (R2: {best[1]['r2']:.4f})")
        print("="*70)

    # =========================
    def run(self):

        X, y = self.load_and_preprocess_data()

        results = self.train_all_models(X, y)

        self.save_results(results)


# =========================
# MAIN
# =========================
def main():
    BASE_DIR = Path(__file__).parent.parent

    data_path = BASE_DIR / "data" / "cleaned_student_data.csv"
    output_dir = BASE_DIR / "outputs"

    if not data_path.exists():
        logger.error("❌ Dataset not found. Run pipeline.py first.")
        return

    pipeline = TrainingPipeline(data_path, output_dir)
    pipeline.run()

    logger.info("✅ TRAINING COMPLETED")


if __name__ == "__main__":
    main()
