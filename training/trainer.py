import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, Dict, List
import polars as pl

logger = logging.getLogger(__name__)


# =========================
# MODEL TRAINER
# =========================
class ModelTrainer:

    def __init__(self, model, model_name, device='cpu',
                 learning_rate=0.001, weight_decay=1e-5):

        self.model = model.to(device)
        self.model_name = model_name
        self.device = device

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = nn.MSELoss()

        # ✅ SAFE SCHEDULER (compatible with all versions)
        try:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        except TypeError:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )

        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    # =========================
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            preds = self.model(X_batch)

            # ✅ Prevent NaN crash
            if torch.isnan(preds).any():
                logger.warning("⚠️ NaN detected in predictions. Skipping batch.")
                continue

            loss = self.criterion(preds, y_batch)

            self.optimizer.zero_grad()
            loss.backward()

            # ✅ Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(train_loader), 1)

    # =========================
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        preds_all, y_all = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.model(X_batch)

                loss = self.criterion(preds, y_batch)

                total_loss += loss.item()
                preds_all.append(preds.cpu().numpy())
                y_all.append(y_batch.cpu().numpy())

        preds_all = np.concatenate(preds_all)
        y_all = np.concatenate(y_all)

        metrics = self.compute_metrics(y_all, preds_all)
        return total_loss / max(len(val_loader), 1), metrics

    # =========================
    def train(self, train_loader, val_loader,
              epochs=100, patience=20):

        logger.info(f"🚀 Training {self.model_name}")

        best_loss = float('inf')
        counter = 0

        for epoch in range(epochs):

            train_loss = self.train_epoch(train_loader)
            val_loss, metrics = self.validate(val_loader)

            self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # ✅ Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1

            # ✅ Logging
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"R2: {metrics['r2']:.4f}"
                )

            if counter >= patience:
                logger.info(f"⏹ Early stopping at epoch {epoch+1}")
                break

        return self.history

    # =========================
    def compute_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))
        }

    # =========================
    def save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"✅ Model saved to {path}")


# =========================
# DATA PREPROCESSOR
# =========================
class DataPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.target_names = ['lifestyle_score', 'burnout_risk']

    def prepare_features(self, df: pl.DataFrame):

        feature_cols = [
            'sleep_hours', 'sleep_consistency', 'exercise', 'tired_during_class',
            'study_hours', 'attendance', 'assignment_submission', 'concentration',
            'screen_time', 'late_phone', 'social_media_usage', 'stress_level',
            'overwhelmed', 'time_management', 'productivity_score', 'gpa'
        ]

        available = [c for c in feature_cols if c in df.columns]

        features = df.select(available).to_numpy()

        if len(features) == 0:
            raise ValueError("❌ No data available after preprocessing")

        features = self.scaler.fit_transform(features)

        logger.info(f"✅ Prepared {len(available)} features")
        return features, available

    def prepare_targets(self, df):
        return df.select(self.target_names).to_numpy()


# =========================
# DATA LOADER
# =========================
def create_data_loaders(X, y, batch_size=32, split=0.8):

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    n = int(len(X) * split)

    train = torch.utils.data.TensorDataset(X[:n], y[:n])
    val = torch.utils.data.TensorDataset(X[n:], y[n:])

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val, batch_size=batch_size
    )

    logger.info(f"✅ Train: {n}, Val: {len(X)-n}")
    return train_loader, val_loader