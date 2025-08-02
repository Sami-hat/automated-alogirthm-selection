# algorithm_selection/reporting/visualiser.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union, Dict, Any
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

plt.style.use('seaborn-v0_8-darkgrid')


class ASVisualizer:
    """Creates visualizations for AS experiments."""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 10)
    
    def plot_model_comparison(self, results_df: pd.DataFrame, metric: str, 
                            title: Optional[str] = None) -> Figure:
        """Create a comparison plot for different models."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        plot_data = results_df.groupby(['model_type', 'model_name', 'scaler_type'])[metric].agg(['mean', 'std']).reset_index()
        
        # Create grouped bar plot
        x = np.arange(len(plot_data))
        width = 0.8
        
        bars = ax.bar(x, plot_data['mean'], width, yerr=plot_data['std'], 
                      capsize=5, alpha=0.8)
        
        # Color by model type
        model_types = plot_data['model_type'].unique()
        colors = dict(zip(model_types, self.colors[:len(model_types)]))
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            bars[i].set_color(colors[row['model_type']])
        
        # Customize plot
        ax.set_xlabel('Model Configuration')
        ax.set_ylabel(metric)
        ax.set_title(title or f'{metric} Comparison Across Models')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['model_name']}\n({row['scaler_type']})" 
                           for _, row in plot_data.iterrows()], rotation=45, ha='right')
        
        # Add legend
        patches = [mpatches.Patch(color=colors[mt], label=mt) for mt in model_types]
        ax.legend(handles=patches, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def plot_scaling_impact(self, results_df: pd.DataFrame, metric: str = 'test_sbs_vbs_gap') -> Figure:
        """Visualize the impact of feature scaling."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        scaling_impact = results_df.groupby(['model_name', 'scaler_type'])[metric].mean().reset_index()
        pivot_data = scaling_impact.pivot(index='model_name', columns='scaler_type', values=metric)
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': metric}, ax=ax)
        
        ax.set_title(f'Impact of Feature Scaling on {metric}')
        ax.set_xlabel('Scaling Method')
        ax.set_ylabel('Model')
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self, results_df: pd.DataFrame) -> Figure:
        """Plot learning curves across repetitions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        metrics = ['train_avg_cost', 'test_avg_cost', 'train_sbs_vbs_gap', 'test_sbs_vbs_gap']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            for (model_name, scaler_type), group in results_df.groupby(['model_name', 'scaler_type']):
                if len(group) > 1:  # Only plot if multiple repetitions
                    x = group['repetition'].values
                    y = group[metric].values
                    ax.plot(x, y, marker='o', label=f"{model_name} ({scaler_type})", alpha=0.7)
            
            ax.set_xlabel('Repetition')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Across Repetitions')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_algorithm_selection_heatmap(self, performance_data: np.ndarray, 
                                       predicted_algorithms: np.ndarray,
                                       instance_subset: Optional[slice] = None) -> Figure:
        """Create a heatmap showing algorithm selection decisions."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Subset data if requested
        if instance_subset is not None:
            performance_data = performance_data[instance_subset]
            predicted_algorithms = predicted_algorithms[instance_subset]
        else:
            # Limit to first 100 instances for visualization
            performance_data = performance_data[:100]
            predicted_algorithms = predicted_algorithms[:100]
        
        # Normalize performance data for better visualization
        norm_performance = (performance_data - performance_data.min(axis=1, keepdims=True)) / \
                          (performance_data.max(axis=1, keepdims=True) - performance_data.min(axis=1, keepdims=True))
        
        # Performance heatmap
        im1 = ax1.imshow(norm_performance.T, aspect='auto', cmap='RdYlGn_r')
        ax1.set_xlabel('Instance')
        ax1.set_ylabel('Algorithm')
        ax1.set_title('Algorithm Performance Heatmap (Normalized)')
        
        # Add colorbar
        cbar = plt.colorbar(im1, ax=ax1)
        cbar.set_label('Normalized Cost')
        
        # Mark selected algorithms
        for i, alg in enumerate(predicted_algorithms):
            ax1.scatter(i, alg, marker='x', color='blue', s=50)
        
        # Selection frequency
        n_algorithms = performance_data.shape[1]
        selection_freq = np.bincount(predicted_algorithms, minlength=n_algorithms)
        ax2.bar(range(n_algorithms), selection_freq)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Selection Frequency')
        ax2.set_title('Algorithm Selection Frequency')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_scores: np.ndarray, 
                              feature_names: Optional[List[str]] = None) -> Figure:
        """Plot feature importance scores."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_features = len(importance_scores)
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        # Sort by importance
        indices = np.argsort(importance_scores)[::-1]
        
        # Create bar plot
        bars = ax.bar(range(n_features), importance_scores[indices], alpha=0.8)
        
        # Color gradient
        colors = plt.cm.viridis(importance_scores[indices] / importance_scores.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance Score')
        ax.set_title('Feature Importance')
        ax.set_xticks(range(n_features))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_performance_distribution(self, results_df: pd.DataFrame) -> Figure:
        """Plot distribution of performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = [
            ('test_avg_cost', 'Test Average Cost'),
            ('test_sbs_vbs_gap', 'Test SBS-VBS Gap'),
            ('train_avg_cost', 'Train Average Cost'),
            ('train_sbs_vbs_gap', 'Train SBS-VBS Gap')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Create violin plot
            data_to_plot = [group[metric].values for name, group in results_df.groupby('model_name')]
            labels = [name for name, _ in results_df.groupby('model_name')]
            
            parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
            
            # Customize colors
            for pc, color in zip(parts['bodies'], self.colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig