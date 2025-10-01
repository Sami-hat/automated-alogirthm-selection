import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score
import warnings
from ..core.base import BaseASModel, ModelType, ASPrediction
import logging

logger = logging.getLogger(__name__)


class SklearnASModel(BaseASModel):
    """Wrapper for scikit-learn models in the AS framework."""
    
    AVAILABLE_MODELS = {
        ModelType.CLASSIFICATION: {
            'logistic': LogisticRegression,
            'mlp': MLPClassifier,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'svm': SVC
        },
        ModelType.REGRESSION: {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'mlp': MLPRegressor,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'svm': SVR
        }
    }
    
    def __init__(self, model_type: ModelType, model_name: str, **kwargs):
        super().__init__(model_type, f"{model_name}_{model_type.value}")
        
        if model_name not in self.AVAILABLE_MODELS[model_type]:
            raise ValueError(f"Unknown model '{model_name}' for {model_type.value}")
        
        model_class = self.AVAILABLE_MODELS[model_type][model_name]
        self._model = model_class(**kwargs)
        self._training_scores = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, **kwargs) -> 'SklearnASModel':
        """Fit the model with optional validation."""
        # Prepare target variable based on model type
        if self.model_type == ModelType.CLASSIFICATION:
            y_processed = np.argmin(y, axis=1)
        else:
            y_processed = y
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self._model.fit(X, y_processed)
        
        self.is_fitted = True
        
        # Perform cross-validation if requested
        if kwargs.get('cv_scoring', False):
            cv_scores = cross_val_score(self._model, X, y_processed, cv=5)
            self._training_scores = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            logger.info(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Validate on validation set if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_pred = self.predict(X_val)
            logger.info(f"Validation completed on {X_val.shape[0]} instances")
        
        return self
    
    def predict(self, X: np.ndarray) -> ASPrediction:
        """Make predictions with confidence scores when available."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        predictions = self._model.predict(X)
        confidence_scores = None
        predicted_costs = None
        
        # Get confidence scores for classification
        if self.model_type == ModelType.CLASSIFICATION and hasattr(self._model, 'predict_proba'):
            try:
                proba = self._model.predict_proba(X)
                confidence_scores = np.max(proba, axis=1)
            except:
                pass
        
        # For regression, we have the full cost matrix
        if self.model_type == ModelType.REGRESSION:
            predicted_costs = predictions
            predictions = np.argmin(predictions, axis=1)
        
        return ASPrediction(
            selected_algorithms=predictions,
            predicted_costs=predicted_costs,
            confidence_scores=confidence_scores,
            metadata={'model_name': self.name}
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            'model_type': self.model_type.value,
            'model_name': self.name,
            'model_params': self._model.get_params()
        }
        if self._training_scores:
            params['training_scores'] = self._training_scores
        return params
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if hasattr(self._model, 'feature_importances_'):
            return self._model.feature_importances_
        elif hasattr(self._model, 'coef_'):
            return np.abs(self._model.coef_).mean(axis=0) if self._model.coef_.ndim > 1 else np.abs(self._model.coef_)
        return None