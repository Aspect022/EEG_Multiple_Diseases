"""
Comprehensive evaluation metrics for ECG classification.

Computes 30+ metrics covering basic performance, error rates, distribution,
likelihood ratios, predictive values, and agreement metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    hamming_loss,
    log_loss,
    roc_auc_score,
    classification_report,
)
import warnings


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0 or np.isnan(denominator):
        return default
    return numerator / denominator


def compute_confusion_derived_metrics(
    cm: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Compute per-class and macro-averaged metrics from a confusion matrix.

    For multi-class, we compute One-vs-Rest metrics for each class, then average.

    Args:
        cm: Confusion matrix of shape (n_classes, n_classes).
        class_names: List of class label names.

    Returns:
        Dictionary of computed metrics.
    """
    n_classes = len(class_names)
    total = cm.sum()

    per_class = {}
    macro_metrics = {
        'TPR': [], 'TNR': [], 'PPV': [], 'NPV': [],
        'FPR': [], 'FNR': [], 'FDR': [], 'FOR': [],
        'prevalence': [], 'threat_score': [],
        'informedness': [], 'markedness': [],
        'prevalence_threshold': [],
        'LR_plus': [], 'LR_minus': [], 'DOR': [],
    }

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn

        # Basic rates
        tpr = _safe_divide(tp, tp + fn)  # Sensitivity / Recall
        tnr = _safe_divide(tn, tn + fp)  # Specificity
        ppv = _safe_divide(tp, tp + fp)  # Precision / PPV
        npv = _safe_divide(tn, tn + fn)  # Negative Predictive Value
        fpr = _safe_divide(fp, fp + tn)  # False Positive Rate
        fnr = _safe_divide(fn, fn + tp)  # False Negative Rate (Miss Rate)
        fdr = _safe_divide(fp, fp + tp)  # False Discovery Rate
        f_or = _safe_divide(fn, fn + tn) # False Omission Rate

        # Distribution metrics
        prev = _safe_divide(tp + fn, total)  # Prevalence
        threat = _safe_divide(tp, tp + fn + fp)  # Threat Score (CSI / Jaccard)
        inform = tpr + tnr - 1.0  # Bookmaker Informedness (Youden's J)
        marked = ppv + npv - 1.0  # Markedness (deltaP)

        # Prevalence threshold
        pt = _safe_divide(
            np.sqrt(tpr * fpr) - fpr,
            tpr - fpr,
            default=0.5,
        )

        # Likelihood ratios
        lr_plus = _safe_divide(tpr, fpr, default=float('inf'))
        lr_minus = _safe_divide(fnr, tnr, default=float('inf'))
        dor = _safe_divide(lr_plus, lr_minus, default=0.0)

        cls_metrics = {
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
            'TPR_sensitivity': float(tpr),
            'TNR_specificity': float(tnr),
            'PPV_precision': float(ppv),
            'NPV': float(npv),
            'FPR': float(fpr),
            'FNR': float(fnr),
            'FDR': float(fdr),
            'FOR': float(f_or),
            'prevalence': float(prev),
            'threat_score_CSI': float(threat),
            'informedness_J': float(inform),
            'markedness_MK': float(marked),
            'prevalence_threshold': float(pt),
            'LR_plus': float(lr_plus),
            'LR_minus': float(lr_minus),
            'DOR': float(dor),
        }
        per_class[class_names[i]] = cls_metrics

        # Accumulate for macro average
        for key in macro_metrics:
            macro_metrics[key].append(cls_metrics.get(key, cls_metrics.get(
                {'TPR': 'TPR_sensitivity', 'TNR': 'TNR_specificity',
                 'PPV': 'PPV_precision'}.get(key, key), 0.0
            )))

    # Compute macro averages
    macro_avg = {}
    for key, values in macro_metrics.items():
        finite_vals = [v for v in values if np.isfinite(v)]
        macro_avg[key] = float(np.mean(finite_vals)) if finite_vals else 0.0

    return {
        'per_class': per_class,
        'macro_avg': macro_avg,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for classification.

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        y_prob: Predicted probabilities, shape (N, C). Optional.
        class_names: List of class names. Auto-generated if None.

    Returns:
        Dictionary with 30+ metrics organized by category.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    unique_classes = sorted(set(np.concatenate([np.unique(y_true), np.unique(y_pred)])))
    n_classes = len(unique_classes)

    if class_names is None:
        class_names = [f"class_{i}" for i in unique_classes]

    results = {}

    # ==================================================================
    # 1. Basic Metrics
    # ==================================================================
    results['accuracy'] = float(accuracy_score(y_true, y_pred))
    results['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))

    results['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    results['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    results['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    results['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    results['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    results['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # Sensitivity = Recall (macro)
    results['sensitivity'] = results['recall_macro']

    # ==================================================================
    # 2. Confusion Matrix Derived Metrics
    # ==================================================================
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    results['confusion_matrix'] = cm.tolist()

    cm_metrics = compute_confusion_derived_metrics(cm, class_names)
    results['per_class_metrics'] = cm_metrics['per_class']

    # Macro-averaged derived metrics
    macro = cm_metrics['macro_avg']
    results['specificity'] = macro['TNR']
    results['FPR'] = macro['FPR']
    results['FNR'] = macro['FNR']
    results['FDR'] = macro['FDR']
    results['FOR'] = macro['FOR']
    results['PPV'] = macro['PPV']
    results['NPV'] = macro['NPV']
    results['prevalence'] = macro['prevalence']
    results['threat_score_CSI'] = macro['threat_score']
    results['informedness_J'] = macro['informedness']
    results['markedness_MK'] = macro['markedness']
    results['prevalence_threshold'] = macro['prevalence_threshold']

    # ==================================================================
    # 3. Likelihood Ratios
    # ==================================================================
    results['LR_plus'] = macro['LR_plus']
    results['LR_minus'] = macro['LR_minus']
    results['DOR'] = macro['DOR']

    # ==================================================================
    # 4. Agreement Metrics
    # ==================================================================
    results['MCC'] = float(matthews_corrcoef(y_true, y_pred))
    results['cohens_kappa'] = float(cohen_kappa_score(y_true, y_pred))

    # ==================================================================
    # 5. Loss-based Metrics
    # ==================================================================
    results['hamming_loss'] = float(hamming_loss(y_true, y_pred))

    if y_prob is not None:
        try:
            results['log_loss'] = float(log_loss(y_true, y_prob, labels=unique_classes))
        except Exception:
            results['log_loss'] = float('nan')

        # Entropy of predictions
        eps = 1e-12
        prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
        entropy = -np.sum(prob_clipped * np.log(prob_clipped), axis=1)
        results['mean_prediction_entropy'] = float(np.mean(entropy))
        results['std_prediction_entropy'] = float(np.std(entropy))
    else:
        results['log_loss'] = float('nan')
        results['mean_prediction_entropy'] = float('nan')
        results['std_prediction_entropy'] = float('nan')

    # ==================================================================
    # 6. AUC-ROC (if probabilities available)
    # ==================================================================
    if y_prob is not None and n_classes > 1:
        try:
            if n_classes == 2:
                results['auc_roc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                results['auc_roc'] = float(roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='macro'
                ))
        except Exception:
            results['auc_roc'] = float('nan')
    else:
        results['auc_roc'] = float('nan')

    # ==================================================================
    # 7. Summary
    # ==================================================================
    results['num_samples'] = int(len(y_true))
    results['num_classes'] = n_classes
    results['class_names'] = class_names

    return results


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dict as a human-readable table string.

    Args:
        metrics: Output of compute_all_metrics.

    Returns:
        Formatted string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("COMPREHENSIVE EVALUATION METRICS")
    lines.append("=" * 60)

    # Basic
    lines.append(f"\n{'--- Basic Metrics ---':^60}")
    for key in ['accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro',
                'sensitivity', 'specificity', 'f1_macro', 'f1_weighted']:
        val = metrics.get(key, float('nan'))
        lines.append(f"  {key:<30s} {val:>10.4f}")

    # Error rates
    lines.append(f"\n{'--- Error & Rate Metrics ---':^60}")
    for key in ['FPR', 'FNR', 'FDR', 'FOR', 'hamming_loss']:
        val = metrics.get(key, float('nan'))
        lines.append(f"  {key:<30s} {val:>10.4f}")

    # Distribution
    lines.append(f"\n{'--- Distribution Metrics ---':^60}")
    for key in ['prevalence', 'threat_score_CSI', 'informedness_J',
                'markedness_MK', 'prevalence_threshold']:
        val = metrics.get(key, float('nan'))
        lines.append(f"  {key:<30s} {val:>10.4f}")

    # Likelihood
    lines.append(f"\n{'--- Likelihood Ratios ---':^60}")
    for key in ['LR_plus', 'LR_minus', 'DOR']:
        val = metrics.get(key, float('nan'))
        lines.append(f"  {key:<30s} {val:>10.4f}")

    # Predictive values
    lines.append(f"\n{'--- Predictive Values ---':^60}")
    for key in ['PPV', 'NPV']:
        val = metrics.get(key, float('nan'))
        lines.append(f"  {key:<30s} {val:>10.4f}")

    # Agreement
    lines.append(f"\n{'--- Agreement Metrics ---':^60}")
    for key in ['MCC', 'cohens_kappa']:
        val = metrics.get(key, float('nan'))
        lines.append(f"  {key:<30s} {val:>10.4f}")

    # Probabilistic
    lines.append(f"\n{'--- Probabilistic Metrics ---':^60}")
    for key in ['auc_roc', 'log_loss', 'mean_prediction_entropy']:
        val = metrics.get(key, float('nan'))
        lines.append(f"  {key:<30s} {val:>10.4f}")

    lines.append("=" * 60)

    return "\n".join(lines)
