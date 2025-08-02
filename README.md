# Algorithm Selection Framework

Automated Algorithm Selection (AS) using ML. This tool has been developed to experiment with different model types, enabling automated selection of the best algorithm from a portfolio on a per-instance basis. The primary metrics are used to show its efficacy: SBS-VBS gap, accuracy, and regret

    SBS represents the case where the algorithm with the lowest average cost is assumed to be our best algorithm
    VBS assumes that the truly best algorithm for a given instance is always chosen

The current set of models supported include: linear models, neural networks, random forest, gradient boosting, and SVM

## Running

```bash
# Install in development mode
pip install -e .


## Quick Start

```python
from algorithm_selection.pipeline.experiment_runner import ExperimentRunner, ExperimentConfig

# Configure experiment
config = ExperimentConfig({
    'models': {
        'classification': ['logistic', 'mlp', 'random_forest'],
        'regression': ['linear', 'mlp', 'random_forest']
    },
    'scaling_options': ['none', 'standard', 'minmax'],
    'n_repetitions': 10,
    'hyperparameter_tuning': {
        'enabled': True,
        'method': 'optuna',
        'n_trials': 50
    }
})

# Run experiment
runner = ExperimentRunner('data/', 'results/', config)
results = runner.run()
```

## Command Line Usage

```bash
# Basic usage
python main.py data/ results/

# With custom configuration
python main.py data/ results/ --config config/example_config.json

# Specify models and options
python main.py data/ results/ --models logistic mlp --scaling none standard --repetitions 5
```

## Project Structure

The code has been made as modular as possible for better readability
Should you wish add a new model, simply extend the `AVAILABLE_MODELS` dictionary in `sklearn_models.py`:

```
algorithm_selection/
    core/
        __init__.py
        base.py              # Base classes and interfaces
        data_handler.py      # Data loading and preprocessing
    models/
        __init__.py
        sklearn_models.py    # Scikit-learn model wrappers
    evaluation/
        __init__.py
        evaluator.py         # Evaluation metrics and utilities
    optimisation/
        __init__.py
        hyperparameter_tuning.py  # Hyperparameter optimisation
    pipeline/
        __init__.py
        experiment_runner.py # Main experiment pipeline
    reporting/
        __init__.py
        visualiser.py        # Visualisation utilities
        report_generator.py  # Report generation
```

The data folders have been left empty for your own datasets:

```
data/
    train/
        performance-data.txt    # Algorithm performance matrix (instances x algorithms)
        instance-features.txt   # Instance feature matrix (instances x features)
    test/
        performance-data.txt
        instance-features.txt
```

(Optional) metadata can be provided in `data/metadata.json`:

```json
{
    "feature_names": ["feature1", "feature2", ...],
    "algorithm_names": ["algorithm1", "algorithm2", ...]
}
```