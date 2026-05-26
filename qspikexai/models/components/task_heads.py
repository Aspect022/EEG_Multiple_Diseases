import torch
import torch.nn as nn

TASK_N_CLASSES = {
    'sleep_apnea':   4,
    'schizophrenia': 2,
    'mci':           2,   # Set to 3 for 3-class MCI if needed
    'depression':    2,
}

def build_task_head(task: str, fused_dim: int = 256, hidden: int = 128) -> nn.Module:
    """Factory function to build task-specific MLP classification heads."""
    if task not in TASK_N_CLASSES:
        raise ValueError(f"Unknown task: {task}. Must be one of {list(TASK_N_CLASSES.keys())}")
        
    n_classes = TASK_N_CLASSES[task]
    return nn.Sequential(
        nn.Linear(fused_dim, hidden),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, n_classes),
    )
