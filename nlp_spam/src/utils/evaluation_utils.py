"""
Detailed evaluation utilities for NLP Spam ML Pipeline.

This module contains evaluation functions with visualizations and detailed analysis
used specifically by the evaluation scripts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, recall_score

# Ensure you have these imports available in your project structure
from pipeline.evaluator import Evaluator
# If you don't have these specific util functions, you can comment them out 
# or replace them with standard print/matplotlib calls
from utils.utils import setup_plotting, save_plot, print_results_summary 


def evaluate_model_detailed(model, X, y, groups=None, model_name="Model"):
    """
    Perform detailed evaluation with visualizations for Spam Detection.
    
    Args:
        model: Trained model (Pipeline)
        X: Feature matrix (text)
        y: Target variable (labels)
        groups: Grouping variable (optional, e.g., 'sms', 'email')
        model_name: Name for plots and reports
        
    Returns:
        Dictionary with detailed evaluation results
    """
    print(f"Detailed evaluation of {model_name}...")
    
    # 1. Make predictions
    y_pred = model.predict(X)
    
    # Get probabilities for ROC curve (if supported by model)
    if hasattr(model, "predict_proba"):
        # The second column [:, 1] corresponds to the positive class (Spam)
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = None
    
    # 2. Calculate metrics using your Evaluator class
    evaluator = Evaluator()
    metrics = evaluator.calculate_metrics(y, y_pred)
    
    # Print summary (Assuming print_results_summary exists, otherwise print manually)
    if 'print_results_summary' in globals():
        print_results_summary(metrics)
    else:
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
    
    # 3. Create visualizations
    _create_evaluation_plots(y, y_pred, y_prob, groups, model_name)
    
    # 4. Spam specific analysis (False Positives vs Negatives)
    spam_metrics = _analyze_spam_performance(y, y_pred)
    
    return {
        'metrics': metrics,
        'spam_metrics': spam_metrics,
        'predictions': y_pred
    }


def _create_evaluation_plots(y_true, y_pred, y_prob, groups, model_name):
    """Create comprehensive evaluation plots for Classification."""
    
    # Optional: setup_plotting()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Evaluation Results', fontsize=16, fontweight='bold')
    
    # --- Plot 1: Confusion Matrix (Replaces Actual vs Predicted) ---
    ax1 = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred)
    # Normalize=None to show raw counts, or 'true' for percentages
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix')
    
    # --- Plot 2: ROC Curve (Replaces Residuals Plot) ---
    ax2 = axes[0, 1]
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend(loc="lower right")
    else:
        ax2.text(0.5, 0.5, 'Probabilities not available\n(Model has no predict_proba)', 
                 ha='center', va='center')
    ax2.set_title('ROC Curve')
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Class Distribution (Replaces Histogram) ---
    ax3 = axes[1, 0]
    labels = ['Ham', 'Spam']
    true_counts = [np.sum(y_true == 0), np.sum(y_true == 1)]
    pred_counts = [np.sum(y_pred == 0), np.sum(y_pred == 1)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax3.bar(x - width/2, true_counts, width, label='Actual', alpha=0.8)
    ax3.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Count')
    ax3.set_title('Class Distribution Balance')
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # --- Plot 4: Performance by Group (SMS vs Email) ---
    ax4 = axes[1, 1]
    if groups is not None:
        _plot_performance_by_group(ax4, y_true, y_pred, groups)
    else:
        ax4.text(0.5, 0.5, 'No grouping variable provided',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance by Group')
    
    plt.tight_layout()
    # save_plot(f'{model_name.lower()}_evaluation.png')
    plt.show()


def _plot_performance_by_group(ax, y_true, y_pred, groups):
    """Plot Accuracy and Recall by group (e.g., source type)."""
    
    group_metrics = []
    group_names = []
    
    # Convert to pandas series for easier handling if necessary
    if not isinstance(groups, pd.Series):
        groups = pd.Series(groups)
        
    for group in groups.unique():
        mask = groups == group
        if mask.sum() > 0:
            g_true = y_true[mask]
            g_pred = y_pred[mask]
            
            # We care about Recall (catching spam) and Accuracy
            acc = accuracy_score(g_true, g_pred)
            # recall_score needs zero_division handling if a group has no spam
            rec = recall_score(g_true, g_pred, pos_label=1, zero_division=0)
            
            group_metrics.append({'accuracy': acc, 'recall': rec})
            group_names.append(str(group))
    
    # Plotting
    x = np.arange(len(group_names))
    width = 0.35
    
    acc_vals = [m['accuracy'] for m in group_metrics]
    rec_vals = [m['recall'] for m in group_metrics]
    
    # Bar for Accuracy
    ax.bar(x - width/2, acc_vals, width, label='Accuracy', color='skyblue')
    # Bar for Recall
    ax.bar(x + width/2, rec_vals, width, label='Recall (Spam Detection)', color='salmon')
    
    ax.set_xlabel('Group')
    ax.set_ylabel('Score (0-1)')
    ax.set_title('Performance by Domain')
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=45)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)


def _analyze_spam_performance(y_true, y_pred):
    """Analyze False Positives vs False Negatives (Critical for Spam)."""
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    # FPR: Legitimate messages lost in spam folder (User annoyance)
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # FNR: Spam messages reaching inbox (Security risk)
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print("\n--- Critical Error Analysis ---")
    print(f"False Positives (Ham labeled as Spam): {fp} ({fp_rate:.1%} of Ham)")
    print(f"False Negatives (Spam labeled as Ham): {fn} ({fn_rate:.1%} of Spam)")
    
    return {
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'confusion_matrix': cm
    }