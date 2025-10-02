from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import json
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """HTML and markdown reports for AS experiments"""
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, results: List[Dict[str, Any]], 
                                    summary: Dict[str, Any],
                                    portfolio_metrics: Dict[str, Dict[str, float]],
                                    config: Dict[str, Any]):
        
        # Generate markdown report
        md_report = self._generate_markdown_report(results, summary, portfolio_metrics, config)
        md_path = self.output_dir / 'experiment_report.md'
        with open(md_path, 'w') as f:
            f.write(md_report)
        
        # Generate HTML report
        html_report = self._generate_html_report(results, summary, portfolio_metrics, config)
        html_path = self.output_dir / 'experiment_report.html'
        with open(html_path, 'w') as f:
            f.write(html_report)
        
        # Generate CSV summaries
        self._generate_csv_summaries(results, summary)
        
        logger.info(f"Reports generated in {self.output_dir}")
    
    def _generate_markdown_report(self, results: List[Dict[str, Any]], 
                                summary: Dict[str, Any],
                                portfolio_metrics: Dict[str, Dict[str, float]],
                                config: Dict[str, Any]) -> str:
        
        report = f"""# Algorithm Selection Experiment Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the results of Algorithm Selection experiments comparing multiple machine learning models
for selecting the best algorithm from a portfolio on a per-instance basis.

### Key Findings

- **Best Model**: {summary['overall_best']['model_name']} ({summary['overall_best']['model_type']}) with {summary['overall_best']['scaler_type']} scaling
- **Best Test SBS-VBS Gap**: {summary['overall_best']['test_sbs_vbs_gap']:.4f}
- **Best Test Average Cost**: {summary['overall_best']['test_avg_cost']:.2f}

## Experiment Configuration

```json
{json.dumps(config, indent=2)}
```

## Portfolio Analysis

### Training Set
- **VBS Cost**: {portfolio_metrics['train']['vbs_cost']:.2f}
- **SBS Cost**: {portfolio_metrics['train']['sbs_cost']:.2f}
- **SBS Algorithm**: Algorithm {portfolio_metrics['train']['sbs_algorithm']}
- **VBS-SBS Gap**: {portfolio_metrics['train']['vbs_sbs_gap']:.2f}

### Test Set
- **VBS Cost**: {portfolio_metrics['test']['vbs_cost']:.2f}
- **SBS Cost**: {portfolio_metrics['test']['sbs_cost']:.2f}
- **SBS Algorithm**: Algorithm {portfolio_metrics['test']['sbs_algorithm']}
- **VBS-SBS Gap**: {portfolio_metrics['test']['vbs_sbs_gap']:.2f}

## Model Performance Summary

### Best Models by Test Cost
"""
        
        # Add best models table
        best_cost_df = pd.DataFrame(summary['best_by_test_cost'])
        report += "\n" + best_cost_df[['model_name', 'model_type', 'scaler_type', 'test_avg_cost', 'test_sbs_vbs_gap']].to_markdown(index=False)
        
        report += """

### Best Models by SBS-VBS Gap
"""
        
        best_gap_df = pd.DataFrame(summary['best_by_test_gap'])
        report += "\n" + best_gap_df[['model_name', 'model_type', 'scaler_type', 'test_avg_cost', 'test_sbs_vbs_gap']].to_markdown(index=False)
        
        report += """

## Detailed Results

Full results are available in `full_results.csv`.

## Visualisations

Visualisations can be found in the `visualisations/` directory:
- `model_comparison_cost.png`: Comparison of average costs across models
- `model_comparison_gap.png`: Comparison of SBS-VBS gaps across models
- `scaling_impact.png`: Impact of feature scaling on performance
- `feature_importance_best_model.png`: Feature importance for the best model (if available)

## Recommendations

Based on the experimental results:

1. **Model Selection**: The {best_model} model with {best_scaling} scaling shows the best performance
2. **Feature Scaling**: {scaling_recommendation}
3. **Future Work**: Consider ensemble methods or hybrid approaches combining the strengths of different models

""".format(
            best_model=summary['overall_best']['model_name'],
            best_scaling=summary['overall_best']['scaler_type'],
            scaling_recommendation=self._get_scaling_recommendation(summary['full_results_df'])
        )
        
        return report
    
    def _generate_html_report(self, results: List[Dict[str, Any]], 
                            summary: Dict[str, Any],
                            portfolio_metrics: Dict[str, Dict[str, float]],
                            config: Dict[str, Any]) -> str:
        """Generate HTML report"""
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Algorithm Selection Experiment Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .metric-box {
            background-color: #ecf0f1;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .best-result {
            background-color: #2ecc71;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>Algorithm Selection Experiment Report</h1>
    <p>Generated: {{ timestamp }}</p>
    
    <div class="best-result">
        <h2>Best Model</h2>
        <p><strong>{{ best_model.model_name }}</strong> ({{ best_model.model_type }}) with {{ best_model.scaler_type }} scaling</p>
        <p>Test SBS-VBS Gap: {{ "%.4f"|format(best_model.test_sbs_vbs_gap) }}</p>
        <p>Test Average Cost: {{ "%.2f"|format(best_model.test_avg_cost) }}</p>
    </div>
    
    <h2>Portfolio Metrics</h2>
    <div class="metric-box">
        <h3>Training Set</h3>
        <p>VBS Cost: {{ "%.2f"|format(portfolio_metrics.train.vbs_cost) }}</p>
        <p>SBS Cost: {{ "%.2f"|format(portfolio_metrics.train.sbs_cost) }}</p>
        <p>Gap: {{ "%.2f"|format(portfolio_metrics.train.vbs_sbs_gap) }}</p>
    </div>
    
    <div class="metric-box">
        <h3>Test Set</h3>
        <p>VBS Cost: {{ "%.2f"|format(portfolio_metrics.test.vbs_cost) }}</p>
        <p>SBS Cost: {{ "%.2f"|format(portfolio_metrics.test.sbs_cost) }}</p>
        <p>Gap: {{ "%.2f"|format(portfolio_metrics.test.vbs_sbs_gap) }}</p>
    </div>
    
    <h2>Model Comparison</h2>
    <img src="visualisations/model_comparison_gap.png" alt="Model Comparison">
    
    <h2>Configuration</h2>
    <pre>{{ config }}</pre>
    
</body>
</html>
"""
        
        template = Template(html_template)
        html_content = template.render(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            best_model=summary['overall_best'],
            portfolio_metrics=portfolio_metrics,
            config=json.dumps(config, indent=2)
        )
        
        return html_content
    
    def _generate_csv_summaries(self, results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """CSV summary files"""
        
        # Model comparison summary
        comparison_data = []
        for result in results:
            row = {
                'model_type': result['model_type'],
                'model_name': result['model_name'],
                'scaler_type': result['scaler_type'],
                'repetition': result['repetition'],
                'train_avg_cost': result['train_evaluation']['avg_cost'],
                'train_sbs_vbs_gap': result['train_evaluation']['sbs_vbs_gap'],
                'test_avg_cost': result['test_evaluation']['avg_cost'],
                'test_sbs_vbs_gap': result['test_evaluation']['sbs_vbs_gap']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        
        # Aggregated statistics
        agg_stats = comparison_df.groupby(['model_type', 'model_name', 'scaler_type']).agg({
            'test_avg_cost': ['mean', 'std', 'min', 'max'],
            'test_sbs_vbs_gap': ['mean', 'std', 'min', 'max']
        })
        agg_stats.to_csv(self.output_dir / 'aggregated_statistics.csv')
    
    def _get_scaling_recommendation(self, results_df: pd.DataFrame) -> str:
        """Scaling recommendation based on results"""
        scaling_performance = results_df.groupby('scaler_type')['test_sbs_vbs_gap'].mean()
        best_scaling = scaling_performance.idxmin()
        
        if best_scaling == 'none':
            return "Feature scaling does not appear to improve performance for this dataset"
        else:
            return f"{best_scaling.capitalize()} scaling consistently improves model performance"