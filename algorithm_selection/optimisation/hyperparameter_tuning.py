import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
import optuna
from functools import partial
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ASHyperparameterOptimizer:
    """Advanced hyperparameter optimisation for AS models."""
    
    def __init__(self, optimization_method: str = 'optuna'):
        self.optimization_method = optimization_method
        self.study = None
        self.best_params = None
        self.optimization_history = []
    
    def optimize(self, model_class: type, model_type: str, param_space: Dict[str, Any],
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                n_trials: int = 100, cv_folds: int = 5) -> Dict[str, Any]:
        """Optimize hyperparameters using specified method."""
        
        if self.optimization_method == 'grid':
            return self._grid_search(model_class, param_space, X_train, y_train, cv_folds)
        elif self.optimization_method == 'random':
            return self._random_search(model_class, param_space, X_train, y_train, cv_folds, n_trials)
        elif self.optimization_method == 'optuna':
            return self._optuna_optimization(model_class, model_type, param_space, 
                                            X_train, y_train, X_val, y_val, n_trials)
        else:
            raise ValueError(f"Unknown optimisation method: {self.optimization_method}")
    
    def _grid_search(self, model_class: type, param_grid: Dict[str, List],
                    X: np.ndarray, y: np.ndarray, cv_folds: int) -> Dict[str, Any]:
        """Perform grid search optimisation."""
        logger.info("Starting grid search optimisation")
        
        base_model = model_class()
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {grid_search.best_score_}")
        
        return self.best_params
    
    def _random_search(self, model_class: type, param_distributions: Dict[str, Any],
                      X: np.ndarray, y: np.ndarray, cv_folds: int, n_iter: int) -> Dict[str, Any]:
        """Perform randomized search optimisation."""
        logger.info("Starting randomized search optimisation")
        
        base_model = model_class()
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X, y)
        self.best_params = random_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {random_search.best_score_}")
        
        return self.best_params
    
    def _optuna_optimization(self, model_class: type, model_type: str, param_space: Dict[str, Any],
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
        n_trials: int) -> Dict[str, Any]:
        """Perform Optuna-based Bayesian optimisation."""
        logger.info("Starting Optuna optimisation")
        
        # Create objective function
        objective = partial(self._optuna_objective, model_class=model_class, model_type=model_type,
            param_space=param_space, X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val)
        
        # Create study
        self.study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        
        # Optimize
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best value: {self.study.best_value}")
        
        return self.best_params
    
    def _optuna_objective(self, trial: optuna.Trial, model_class: type, model_type: str,
        param_space: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
        X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> float:
        """Objective function for Optuna optimisation."""
        
        # Sample hyperparameters
        params = {}
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'],
                log=param_config.get('log', False))
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
        
        # Create and train model
        model = model_class(**params)
        
        # Use validation set if provided, otherwise use cross-validation
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train)
            if model_type == 'classification':
                y_val_processed = np.argmin(y_val, axis=1) if y_val.ndim > 1 else y_val
                score = -model.score(X_val, y_val_processed)
            else:
                predictions = model.predict(X_val)
                score = mean_absolute_error(y_val, predictions)
        else:
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            score = -scores.mean()
        
        return score
    
    def get_param_importance(self) -> Optional[pd.DataFrame]:
        """Get parameter importance from Optuna study."""
        if self.study is None:
            return None
        
        importance = optuna.importance.get_param_importances(self.study)
        df = pd.DataFrame(list(importance.items()), columns=['parameter', 'importance'])
        return df.sort_values('importance', ascending=False)
    
    def plot_optimization_history(self):
        """Plot optimisation history."""
        if self.study is None:
            logger.warning("No Optuna study available for plotting")
            return
        
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            fig1 = plot_optimization_history(self.study)
            fig2 = plot_param_importances(self.study)
            
            return fig1, fig2
        except ImportError:
            logger.warning("Optuna visualisation plotly error")
            return None, None


# Define hyperparameter spaces for different models
HYPERPARAMETER_SPACES = {
    'mlp_classifier': {
        'hidden_layer_sizes': {'type': 'categorical', 'choices': [(50,), (100,), (100, 50), (100, 100), (150, 100, 50)]},
        'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'logistic']},
        'solver': {'type': 'categorical', 'choices': ['adam', 'sgd', 'lbfgs']},
        'alpha': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True},
        'learning_rate': {'type': 'categorical', 'choices': ['constant', 'invscaling', 'adaptive']},
        'learning_rate_init': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
        'max_iter': {'type': 'int', 'low': 200, 'high': 1000},
        'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128, 'auto']}
    },
    'mlp_regressor': {
        'hidden_layer_sizes': {'type': 'categorical', 'choices': [(50,), (100,), (100, 50), (100, 100), (150, 100, 50)]},
        'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'logistic']},
        'solver': {'type': 'categorical', 'choices': ['adam', 'sgd', 'lbfgs']},
        'alpha': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True},
        'learning_rate': {'type': 'categorical', 'choices': ['constant', 'invscaling', 'adaptive']},
        'learning_rate_init': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
        'max_iter': {'type': 'int', 'low': 200, 'high': 1000}
    },
    'random_forest_classifier': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'max_depth': {'type': 'int', 'low': 5, 'high': 50},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
        'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
        'bootstrap': {'type': 'categorical', 'choices': [True, False]}
    },
    'gradient_boosting_classifier': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
        'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0}
    }
}