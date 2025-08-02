# algorithm_selection/pipeline/experiment_runner.py
"""Comprehensive experiment runner for Algorithm Selection."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import pickle
from datetime import datetime
import pandas as pd
from itertools import product
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from ..core.base import ModelType
from ..core.data_handler import DataHandler
from ..models.sklearn_models import SklearnASModel
from ..evaluation.evaluator import ASEvaluator
from ..optimisation.hyperparameter_tuning import ASHyperparameterOptimizer, HYPERPARAMETER_SPACES
from ..reporting.visualiser import ASVisualizer
from ..reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration for AS experiments."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        if config_dict is None:
            config_dict = {}
        
        self.models = config_dict.get('models', {
            ModelType.CLASSIFICATION: ['logistic', 'mlp', 'random_forest'],
            ModelType.REGRESSION: ['linear', 'ridge', 'mlp', 'random_forest']
        })
        
        self.scaling_options = config_dict.get('scaling_options', ['none', 'standard', 'minmax'])
        self.n_repetitions = config_dict.get('n_repetitions', 1)
        self.use_validation_split = config_dict.get('use_validation_split', True)
        self.validation_size = config_dict.get('validation_size', 0.2)
        
        self.hyperparameter_tuning = config_dict.get('hyperparameter_tuning', {
            'enabled': True,
            'method': 'optuna',
            'n_trials': 50
        })
        
        self.parallel_execution = config_dict.get('parallel_execution', True)
        self.max_workers = config_dict.get('max_workers', None)
        self.save_models = config_dict.get('save_models', True)
        self.generate_report = config_dict.get('generate_report', True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'models': {k.value if isinstance(k, ModelType) else k: v for k, v in self.models.items()},
            'scaling_options': self.scaling_options,
            'n_repetitions': self.n_repetitions,
            'use_validation_split': self.use_validation_split,
            'validation_size': self.validation_size,
            'hyperparameter_tuning': self.hyperparameter_tuning,
            'parallel_execution': self.parallel_execution,
            'max_workers': self.max_workers,
            'save_models': self.save_models,
            'generate_report': self.generate_report
        }


class ExperimentRunner:
    """Runs comprehensive AS experiments with various configurations."""
    
    def __init__(self, data_dir: Union[str, Path], output_dir: Union[str, Path], 
                 config: Optional[ExperimentConfig] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or ExperimentConfig()
        self.data_handler = DataHandler(self.data_dir)
        self.evaluator = ASEvaluator()
        self.visualizer = ASVisualizer()
        
        self.results = []
        self.trained_models = {}
        
        # Setup experiment directory
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"experiment_{self.experiment_id}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Save configuration
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def run(self) -> Dict[str, Any]:
        """Run the complete experiment pipeline."""
        logger.info(f"Starting experiment {self.experiment_id}")
        
        try:
            # Load data
            logger.info("Loading data...")
            train_perf, train_features, test_perf, test_features = self.data_handler.load_data()
            
            # Calculate portfolio metrics
            portfolio_metrics = self.evaluator.calculate_portfolio_metrics(train_perf, test_perf)
            self._save_portfolio_metrics(portfolio_metrics)
            
            # Create validation split if requested
            if self.config.use_validation_split:
                train_features, val_features, train_perf, val_perf = self.data_handler.create_validation_split(
                    train_features, train_perf, self.config.validation_size
                )
            else:
                val_features, val_perf = None, None
            
            # Run experiments
            if self.config.parallel_execution:
                results = self._run_parallel_experiments(
                    train_perf, train_features, test_perf, test_features,
                    val_perf, val_features, portfolio_metrics
                )
            else:
                results = self._run_sequential_experiments(
                    train_perf, train_features, test_perf, test_features,
                    val_perf, val_features, portfolio_metrics
                )
            
            # Analyze results
            summary = self._analyze_results(results)
            
            # Generate visualizations
            self._generate_visualizations(results, summary)
            
            # Generate report
            if self.config.generate_report:
                report_gen = ReportGenerator(self.experiment_dir)
                report_gen.generate_comprehensive_report(
                    results, summary, portfolio_metrics, self.config.to_dict()
                )
            
            logger.info(f"Experiment {self.experiment_id} completed successfully")
            
            return {
                'experiment_id': self.experiment_id,
                'results': results,
                'summary': summary,
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _run_sequential_experiments(self, train_perf: np.ndarray, train_features: np.ndarray,
                                  test_perf: np.ndarray, test_features: np.ndarray,
                                  val_perf: Optional[np.ndarray], val_features: Optional[np.ndarray],
                                  portfolio_metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Run experiments sequentially."""
        results = []
        
        for model_type in self.config.models:
            for model_name in self.config.models[model_type]:
                for scaler_type in self.config.scaling_options:
                    for rep in range(self.config.n_repetitions):
                        result = self._run_single_experiment(
                            model_type, model_name, scaler_type, rep,
                            train_perf, train_features, test_perf, test_features,
                            val_perf, val_features, portfolio_metrics
                        )
                        results.append(result)
                        self.results.append(result)
        
        return results
    
    def _run_parallel_experiments(self, train_perf: np.ndarray, train_features: np.ndarray,
                                test_perf: np.ndarray, test_features: np.ndarray,
                                val_perf: Optional[np.ndarray], val_features: Optional[np.ndarray],
                                portfolio_metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Run experiments in parallel."""
        results = []
        
        # Create experiment configurations
        experiments = []
        for model_type in self.config.models:
            for model_name in self.config.models[model_type]:
                for scaler_type in self.config.scaling_options:
                    for rep in range(self.config.n_repetitions):
                        experiments.append((model_type, model_name, scaler_type, rep))
        
        # Run in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_experiment,
                    model_type, model_name, scaler_type, rep,
                    train_perf, train_features, test_perf, test_features,
                    val_perf, val_features, portfolio_metrics
                ): (model_type, model_name, scaler_type, rep)
                for model_type, model_name, scaler_type, rep in experiments
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    self.results.append(result)
                    logger.info(f"Completed: {futures[future]}")
                except Exception as e:
                    logger.error(f"Failed: {futures[future]} - {str(e)}")
        
        return results
    
    def _run_single_experiment(self, model_type: ModelType, model_name: str, scaler_type: str,
                             repetition: int, train_perf: np.ndarray, train_features: np.ndarray,
                             test_perf: np.ndarray, test_features: np.ndarray,
                             val_perf: Optional[np.ndarray], val_features: Optional[np.ndarray],
                             portfolio_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        
        logger.info(f"Running {model_name} ({model_type.value}) with {scaler_type} scaling, rep {repetition}")
        
        # Prepare features
        if scaler_type != 'none':
            train_feat_scaled, test_feat_scaled = self.data_handler.preprocess_features(
                train_features, test_features, scaler_type
            )
            if val_features is not None:
                _, val_feat_scaled = self.data_handler.preprocess_features(
                    train_features, val_features, scaler_type
                )
            else:
                val_feat_scaled = None
        else:
            train_feat_scaled = train_features
            test_feat_scaled = test_features
            val_feat_scaled = val_features
        
        # Hyperparameter tuning
        best_params = {}
        if self.config.hyperparameter_tuning['enabled']:
            param_space_key = f"{model_name}_{model_type.value.split('_')[0]}"
            if param_space_key in HYPERPARAMETER_SPACES:
                optimizer = ASHyperparameterOptimizer(self.config.hyperparameter_tuning['method'])
                best_params = optimizer.optimize(
                    SklearnASModel.AVAILABLE_MODELS[model_type][model_name],
                    model_type.value,
                    HYPERPARAMETER_SPACES[param_space_key],
                    train_feat_scaled,
                    train_perf if model_type == ModelType.REGRESSION else np.argmin(train_perf, axis=1),
                    val_feat_scaled,
                    val_perf if model_type == ModelType.REGRESSION else np.argmin(val_perf, axis=1) if val_perf is not None else None,
                    n_trials=self.config.hyperparameter_tuning['n_trials']
                )
        
        # Create and train model
        model = SklearnASModel(model_type, model_name, **best_params)
        
        if val_features is not None:
            model.fit(train_feat_scaled, train_perf, validation_data=(val_feat_scaled, val_perf))
        else:
            model.fit(train_feat_scaled, train_perf)
        
        # Make predictions
        train_predictions = model.predict(train_feat_scaled)
        test_predictions = model.predict(test_feat_scaled)
        
        # Evaluate
        train_eval = self.evaluator.evaluate_model(
            train_perf, train_predictions,
            portfolio_metrics['train']['vbs_cost'],
            portfolio_metrics['train']['sbs_cost'],
            model_type,
            np.argmin(train_perf, axis=1) if model_type == ModelType.CLASSIFICATION else None,
            'train'
        )
        
        test_eval = self.evaluator.evaluate_model(
            test_perf, test_predictions,
            portfolio_metrics['test']['vbs_cost'],
            portfolio_metrics['test']['sbs_cost'],
            model_type,
            np.argmin(test_perf, axis=1) if model_type == ModelType.CLASSIFICATION else None,
            'test'
        )
        
        # Save model if requested
        model_key = f"{model_name}_{model_type.value}_{scaler_type}_{repetition}"
        if self.config.save_models:
            model_path = self.experiment_dir / 'models' / f"{model_key}.pkl"
            model_path.parent.mkdir(exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        self.trained_models[model_key] = model
        
        # Compile results
        result = {
            'model_type': model_type.value,
            'model_name': model_name,
            'scaler_type': scaler_type,
            'repetition': repetition,
            'hyperparameters': best_params,
            'train_evaluation': train_eval.to_dict(),
            'test_evaluation': test_eval.to_dict(),
            'feature_importance': model.get_feature_importance()
        }
        
        return result
    
    def _save_portfolio_metrics(self, metrics: Dict[str, Dict[str, float]]):
        """Save portfolio metrics to file."""
        with open(self.experiment_dir / 'portfolio_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze experiment results and create summary statistics."""
        df_results = pd.DataFrame([
            {
                'model_type': r['model_type'],
                'model_name': r['model_name'],
                'scaler_type': r['scaler_type'],
                'repetition': r['repetition'],
                'train_avg_cost': r['train_evaluation']['avg_cost'],
                'train_sbs_vbs_gap': r['train_evaluation']['sbs_vbs_gap'],
                'test_avg_cost': r['test_evaluation']['avg_cost'],
                'test_sbs_vbs_gap': r['test_evaluation']['sbs_vbs_gap'],
                **{f"train_{k}": v for k, v in r['train_evaluation'].items() 
                   if k not in ['avg_cost', 'sbs_vbs_gap', 'dataset_type']},
                **{f"test_{k}": v for k, v in r['test_evaluation'].items() 
                   if k not in ['avg_cost', 'sbs_vbs_gap', 'dataset_type']}
            }
            for r in results
        ])
        
        # Group by model configuration and calculate statistics
        groupby_cols = ['model_type', 'model_name', 'scaler_type']
        summary_stats = df_results.groupby(groupby_cols).agg(['mean', 'std', 'min', 'max'])
        
        # Find best configurations
        best_by_test_cost = df_results.loc[df_results.groupby(groupby_cols)['test_avg_cost'].idxmin()]
        best_by_test_gap = df_results.loc[df_results.groupby(groupby_cols)['test_sbs_vbs_gap'].idxmin()]
        
        # Overall best model
        overall_best = df_results.loc[df_results['test_sbs_vbs_gap'].idxmin()]
        
        summary = {
            'summary_statistics': summary_stats.to_dict(),
            'best_by_test_cost': best_by_test_cost.to_dict('records'),
            'best_by_test_gap': best_by_test_gap.to_dict('records'),
            'overall_best': overall_best.to_dict(),
            'full_results_df': df_results
        }
        
        # Save summary
        with open(self.experiment_dir / 'summary.json', 'w') as f:
            json.dump({k: v for k, v in summary.items() if k != 'full_results_df'}, f, indent=2)
        
        df_results.to_csv(self.experiment_dir / 'full_results.csv', index=False)
        
        return summary
    
    def _generate_visualizations(self, results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """Generate comprehensive visualizations."""
        viz_dir = self.experiment_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        df = summary['full_results_df']
        
        # Performance comparison
        fig = self.visualizer.plot_model_comparison(df, 'test_avg_cost', 'Average Cost (Test)')
        fig.savefig(viz_dir / 'model_comparison_cost.png', dpi=300, bbox_inches='tight')
        
        fig = self.visualizer.plot_model_comparison(df, 'test_sbs_vbs_gap', 'SBS-VBS Gap (Test)')
        fig.savefig(viz_dir / 'model_comparison_gap.png', dpi=300, bbox_inches='tight')
        
        # Scaling impact
        fig = self.visualizer.plot_scaling_impact(df, 'test_sbs_vbs_gap')
        fig.savefig(viz_dir / 'scaling_impact.png', dpi=300, bbox_inches='tight')
        
        # Learning curves (if multiple repetitions)
        if self.config.n_repetitions > 1:
            fig = self.visualizer.plot_learning_curves(df)
            fig.savefig(viz_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
        
        # Feature importance (for best model)
        best_model_key = f"{summary['overall_best']['model_name']}_{summary['overall_best']['model_type']}_{summary['overall_best']['scaler_type']}_{summary['overall_best']['repetition']}"
        if best_model_key in self.trained_models:
            best_model = self.trained_models[best_model_key]
            importance = best_model.get_feature_importance()
            if importance is not None:
                fig = self.visualizer.plot_feature_importance(
                    importance, self.data_handler.feature_names
                )
                fig.savefig(viz_dir / 'feature_importance_best_model.png', dpi=300, bbox_inches='tight')