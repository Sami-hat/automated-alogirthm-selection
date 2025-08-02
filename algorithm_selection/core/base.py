# algorithm_selection/core/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration for model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class ASPrediction:
    """Data class for AS model predictions."""
    selected_algorithms: np.ndarray
    predicted_costs: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ASEvaluation:
    """Data class for AS model evaluation results."""
    avg_cost: float
    sbs_vbs_gap: float
    metrics: Dict[str, float]
    dataset_type: str  # 'train' or 'test'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation results to dictionary."""
        return {
            'avg_cost': self.avg_cost,
            'sbs_vbs_gap': self.sbs_vbs_gap,
            'dataset_type': self.dataset_type,
            **self.metrics
        }


class BaseASModel(ABC):
    """Abstract base class for Algorithm Selection models."""
    
    def __init__(self, model_type: ModelType, name: str):
        self.model_type = model_type
        self.name = name
        self.is_fitted = False
        self._model = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseASModel':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> ASPrediction:
        """Make predictions on new instances."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.model_type.value})"