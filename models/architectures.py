"""
Deep Learning Models for Student Lifestyle Analytics
Implements multiple neural network architectures for predicting lifestyle metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC, nn.Module):
    """Abstract base class for all models"""
    
    def __init__(self, input_size: int, output_size: int = 2):
        """
        Args:
            input_size: Number of input features
            output_size: Number of output values (lifestyle_score, burnout_risk)
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
    @abstractmethod
    def forward(self, x):
        """Forward pass - must be implemented by subclasses"""
        pass


class MLPModel(BaseModel):
    """
    Multi-Layer Perceptron (MLP)
    Simple feedforward network with 3 hidden layers
    """
    
    def __init__(self, input_size: int, output_size: int = 2, hidden_sizes: list = None):
        super().__init__(input_size, output_size)
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DNNModel(BaseModel):
    """
    Deep Neural Network (DNN)
    Deeper architecture with more layers for better feature extraction
    """
    
    def __init__(self, input_size: int, output_size: int = 2):
        super().__init__(input_size, output_size)
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class DropoutRegularizedModel(BaseModel):
    """
    Neural Network with Dropout Regularization
    Prevents overfitting through dropout layers
    """
    
    def __init__(self, input_size: int, output_size: int = 2, dropout_rate: float = 0.3):
        super().__init__(input_size, output_size)
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class BatchNormalizedModel(BaseModel):
    """
    Neural Network with Batch Normalization
    Stabilizes training and allows higher learning rates
    """
    
    def __init__(self, input_size: int, output_size: int = 2):
        super().__init__(input_size, output_size)
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class HybridModel(BaseModel):
    """
    Hybrid Model combining Batch Normalization + Dropout + Residual concepts
    Advanced architecture for superior performance
    """
    
    def __init__(self, input_size: int, output_size: int = 2, dropout_rate: float = 0.25):
        super().__init__(input_size, output_size)
        
        # Main network path
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.drop4 = nn.Dropout(dropout_rate)
        
        # Auxiliary pathway for better gradient flow
        self.aux_fc = nn.Linear(input_size, 64)
        
        # Output layer
        self.output = nn.Linear(32 + 64, output_size)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Main path
        x1 = self.relu(self.bn1(self.fc1(x)))
        x1 = self.drop1(x1)
        
        x1 = self.relu(self.bn2(self.fc2(x1)))
        x1 = self.drop2(x1)
        
        x1 = self.relu(self.bn3(self.fc3(x1)))
        x1 = self.drop3(x1)
        
        x1 = self.relu(self.bn4(self.fc4(x1)))
        x1 = self.drop4(x1)
        
        # Auxiliary path
        x2 = self.relu(self.aux_fc(x))
        
        # Concatenate paths
        x_combined = torch.cat([x1, x2], dim=1)
        
        # Output
        output = self.output(x_combined)
        return output


# Model factory function
def create_model(model_type: str, input_size: int, output_size: int = 2, device: str = 'cpu'):
    """
    Factory function to create model instances
    
    Args:
        model_type: Type of model to create
        input_size: Number of input features
        output_size: Number of output values
        device: 'cpu' or 'cuda'
        
    Returns:
        Initialized model on specified device
    """
    models = {
        'mlp': MLPModel,
        'dnn': DNNModel,
        'dropout': DropoutRegularizedModel,
        'batch_norm': BatchNormalizedModel,
        'hybrid': HybridModel
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    model_class = models[model_type.lower()]
    model = model_class(input_size, output_size)
    model = model.to(device)
    
    logger.info(f"✅ Created {model_type.upper()} model on {device}")
    return model


# Summary of model architectures
MODEL_CONFIGS = {
    'mlp': {
        'name': 'Multi-Layer Perceptron',
        'description': 'Simple 3-layer network with ReLU activations',
        'complexity': 'Low',
        'best_for': 'Quick training, baseline comparisons'
    },
    'dnn': {
        'name': 'Deep Neural Network',
        'description': 'Deep 5-layer network for complex feature learning',
        'complexity': 'High',
        'best_for': 'Non-linear relationships, high accuracy'
    },
    'dropout': {
        'name': 'Dropout Regularized Network',
        'description': 'Network with dropout layers to prevent overfitting',
        'complexity': 'Medium',
        'best_for': 'Preventing overfitting, generalization'
    },
    'batch_norm': {
        'name': 'Batch Normalized Network',
        'description': 'Network with batch normalization for stable training',
        'complexity': 'Medium',
        'best_for': 'Faster convergence, internal covariate shift reduction'
    },
    'hybrid': {
        'name': 'Hybrid Advanced Model',
        'description': 'Combines batch norm, dropout, and auxiliary pathway',
        'complexity': 'Very High',
        'best_for': 'Maximum performance, complex patterns'
    }
}
