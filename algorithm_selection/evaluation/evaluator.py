# algorithm_selection/evaluation/evaluator.py
"""Evaluation utilities for Algorithm Selection models."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_absolute_error, mean_squared_error, r2_score,
                           confusion_matrix, classification_report)
import pandas as pd
from ..core.base import ASEvaluation, ModelType, ASPrediction
import logging

logger = logging.getLogger(__name__)


class ASEvaluator:
    """Comprehensive evaluator for Algorithm Selection models."""
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_model(self, performance_data: np.ndarray, predictions: ASPrediction,
                      vbs_cost: float, sbs_cost: float, model_type: ModelType,
                      true_labels: Optional[np.ndarray] = None,
                      dataset_type: str = 'test') -> ASEvaluation:
        """Evaluate AS model performance with comprehensive metrics."""
        
        # Calculate average cost and SBS-VBS gap
        avg_cost = self._calculate_avg_cost(performance_data, predictions.selected_algorithms)
        sbs_vbs_gap = self._calculate_sbs_vbs_gap(avg_cost, vbs_cost, sbs_cost)
        
        # Calculate model-specific metrics
        if model_type == ModelType.CLASSIFICATION:
            metrics = self._evaluate_classification(predictions.selected_algorithms, true_labels)
        else:
            metrics = self._evaluate_regression(predictions.predicted_costs, performance_data)
        
        # Add algorithm selection specific metrics
        as_metrics = self._calculate_as_metrics(performance_data, predictions.selected_algorithms)
        metrics.update(as_metrics)
        
        evaluation = ASEvaluation(
            avg_cost=avg_cost,
            sbs_vbs_gap=sbs_vbs_gap,
            metrics=metrics,
            dataset_type=dataset_type
        )
        
        self.evaluation_history.append(evaluation)
        return evaluation
    
    def _calculate_avg_cost(self, performance_data: np.ndarray, selected_algorithms: np.ndarray) -> float:
        """Calculate average cost of selected algorithms."""
        n_instances = len(selected_algorithms)
        costs = performance_data[np.arange(n_instances), selected_algorithms]
        return np.mean(costs)
    
    def _calculate_sbs_vbs_gap(self, avg_cost: float, vbs_cost: float, sbs_cost: float) -> float:
        """Calculate SBS-VBS gap metric."""
        if sbs_cost == vbs_cost:
            return 0.0
        return (avg_cost - vbs_cost) / (sbs_cost - vbs_cost)
    
    def _evaluate_classification(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate classification model metrics."""
        if true_labels is None:
            raise ValueError("True labels required for classification evaluation")
        
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions) * 100,
            'precision': precision_score(true_labels, predictions, average='weighted', zero_division=0) * 100,
            'recall': recall_score(true_labels, predictions, average='weighted', zero_division=0) * 100,
            'f1_score': f1_score(true_labels, predictions, average='weighted', zero_division=0) * 100
        }
        
        # Add per-class metrics if there are few algorithms
        n_classes = len(np.unique(true_labels))
        if n_classes <= 10:
            for i in range(n_classes):
                class_mask = true_labels == i
                if np.any(class_mask):
                    class_acc = accuracy_score(true_labels[class_mask], predictions[class_mask]) * 100
                    metrics[f'accuracy_alg_{i}'] = class_acc
        
        return metrics
    
    def _evaluate_regression(self, predicted_costs: np.ndarray, true_costs: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model metrics."""
        if predicted_costs is None:
            raise ValueError("Predicted costs required for regression evaluation")
        
        metrics = {
            'mae': mean_absolute_error(true_costs, predicted_costs),
            'mse': mean_squared_error(true_costs, predicted_costs),
            'rmse': np.sqrt(mean_squared_error(true_costs, predicted_costs)),
            'r2_score': r2_score(true_costs, predicted_costs)
        }
        
        # Add normalized metrics
        true_mean = np.mean(true_costs)
        if true_mean > 0:
            metrics['normalized_mae'] = metrics['mae'] / true_mean
            metrics['normalized_rmse'] = metrics['rmse'] / true_mean
        
        return metrics
    
    def _calculate_as_metrics(self, performance_data: np.ndarray, selected_algorithms: np.ndarray) -> Dict[str, float]:
        """Calculate algorithm selection specific metrics."""
        n_instances = len(selected_algorithms)
        
        # Calculate how often we select the optimal algorithm
        optimal_algorithms = np.argmin(performance_data, axis=1)
        optimal_selection_rate = np.mean(selected_algorithms == optimal_algorithms) * 100
        
        # Calculate average rank of selected algorithms
        ranks = np.argsort(np.argsort(performance_data, axis=1), axis=1)
        selected_ranks = ranks[np.arange(n_instances), selected_algorithms]
        avg_rank = np.mean(selected_ranks) + 1  # Convert to 1-based ranking
        
        # Calculate regret (difference from optimal)
        optimal_costs = performance_data[np.arange(n_instances), optimal_algorithms]
        selected_costs = performance_data[np.arange(n_instances), selected_algorithms]
        avg_regret = np.mean(selected_costs - optimal_costs)
        
        return {
            'optimal_selection_rate': optimal_selection_rate,
            'average_rank': avg_rank,
            'average_regret': avg_regret
        }
    
    def calculate_portfolio_metrics(self, train_performance: np.ndarray, 
                                  test_performance: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate VBS and SBS metrics for the algorithm portfolio."""
        metrics = {}
        
        for name, data in [('train', train_performance), ('test', test_performance)]:
            vbs_cost = np.mean(np.min(data, axis=1))
            sbs_costs = np.mean(data, axis=0)
            sbs_cost = np.min(sbs_costs)
            sbs_index = np.argmin(sbs_costs)
            
            metrics[name] = {
                'vbs_cost': vbs_cost,
                'sbs_cost': sbs_cost,
                'sbs_algorithm': sbs_index,
                'vbs_sbs_gap': sbs_cost - vbs_cost,
                'relative_gap': (sbs_cost - vbs_cost) / vbs_cost * 100 if vbs_cost > 0 else 0
            }
        
        return metrics
    
    def generate_summary_report(self) -> pd.DataFrame:
        """Generate a summary report of all evaluations."""
        if not self.evaluation_history:
            logger.warning("No evaluations to summarize")
            return pd.DataFrame()
        
        data = []
        for eval in self.evaluation_history:
            row = eval.to_dict()
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def compare_models(self, evaluations: List[ASEvaluation]) -> pd.DataFrame:
        """Compare multiple model evaluations."""
        comparison_data = []
        
        for eval in evaluations:
            data = {
                'avg_cost': eval.avg_cost,
                'sbs_vbs_gap': eval.sbs_vbs_gap,
                'dataset': eval.dataset_type
            }
            data.update(eval.metrics)
            comparison_data.append(data)
        
        df = pd.DataFrame(comparison_data)
        return df.round(4)