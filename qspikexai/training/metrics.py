import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

def compute_all_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray) -> dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: array of shape (N,)
        y_pred_logits: array of shape (N, n_classes)
    Returns:
        dict: classification performance metrics
    """
    y_pred_classes = np.argmax(y_pred_logits, axis=1)
    
    # Calculate probabilities via softmax for AUROC
    exps = np.exp(y_pred_logits - np.max(y_pred_logits, axis=1, keepdims=True))
    y_pred_probs = exps / np.sum(exps, axis=1, keepdims=True)
    
    acc = accuracy_score(y_true, y_pred_classes)
    bal_acc = balanced_accuracy_score(y_true, y_pred_classes)
    macro_f1 = f1_score(y_true, y_pred_classes, average='macro')
    weighted_f1 = f1_score(y_true, y_pred_classes, average='weighted')
    
    # Calculate AUROC
    n_classes = y_pred_logits.shape[1]
    auc_roc = 0.5
    try:
        if n_classes == 2:
            auc_roc = roc_auc_score(y_true, y_pred_probs[:, 1])
        else:
            # Multi-class OVR AUROC
            auc_roc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
    except Exception:
        # Fallback if only 1 class is present in split
        auc_roc = 0.5
        
    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'auc_roc': float(auc_roc)
    }
