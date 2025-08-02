"""Algorithm Selection Framework."""

from .core.base import ModelType, ASPrediction, ASEvaluation, BaseASModel
from .core.data_handler import DataHandler
from .models.sklearn_models import SklearnASModel
from .evaluation.evaluator import ASEvaluator
from .optimisation.hyperparameter_tuning import ASHyperparameterOptimizer
from .pipeline.experiment_runner import ExperimentRunner, ExperimentConfig
from .reporting.visualiser import ASVisualizer
from .reporting.report_generator import ReportGenerator

__version__ = "1.0.0"

__all__ = [
    "ModelType",
    "ASPrediction",
    "ASEvaluation",
    "BaseASModel",
    "DataHandler",
    "SklearnASModel",
    "ASEvaluator",
    "ASHyperparameterOptimizer",
    "ExperimentRunner",
    "ExperimentConfig",
    "ASVisualizer",
    "ReportGenerator",
]