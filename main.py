# main.py
import argparse
import json
from pathlib import Path
import logging
import sys
from algorithm_selection.pipeline.experiment_runner import ExperimentRunner, ExperimentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config_from_file(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Run Algorithm Selection experiments')
    parser.add_argument('data_dir', type=str, help='Path to data directory')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--models', nargs='+', help='List of models to use')
    parser.add_argument('--scaling', nargs='+', default=['none', 'standard'], 
                       help='Scaling methods to use')
    parser.add_argument('--repetitions', type=int, default=1, 
                       help='Number of repetitions')
    parser.add_argument('--no-tuning', action='store_true', 
                       help='Disable hyperparameter tuning')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Run experiments in parallel')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config_dict = load_config_from_file(Path(args.config))
    else:
        config_dict = {
            'models': {
                'classification': args.models or ['logistic', 'mlp', 'random_forest'],
                'regression': args.models or ['linear', 'mlp', 'random_forest']
            },
            'scaling_options': args.scaling,
            'n_repetitions': args.repetitions,
            'hyperparameter_tuning': {
                'enabled': not args.no_tuning,
                'method': 'optuna',
                'n_trials': 50
            },
            'parallel_execution': args.parallel
        }
    
    config = ExperimentConfig(config_dict)
    
    # Run experiments
    try:
        runner = ExperimentRunner(args.data_dir, args.output_dir, config)
        results = runner.run()
        
        logger.info(f"Experiments completed successfully!")
        logger.info(f"Results saved to: {runner.experiment_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Best Model: {results['summary']['overall_best']['model_name']}")
        print(f"Model Type: {results['summary']['overall_best']['model_type']}")
        print(f"Scaling: {results['summary']['overall_best']['scaler_type']}")
        print(f"Test SBS-VBS Gap: {results['summary']['overall_best']['test_sbs_vbs_gap']:.4f}")
        print(f"Test Average Cost: {results['summary']['overall_best']['test_avg_cost']:.2f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()