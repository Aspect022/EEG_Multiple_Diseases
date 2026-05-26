import torch
import numpy as np
from ..models.qspikexai_net import QSpikeXAINet

def quantum_gate_attribution(
    model: QSpikeXAINet,
    x_raw: torch.Tensor,
    x_scalogram: torch.Tensor,
    target_class: int = None,
) -> dict:
    """
    Compute attribution scores for each VQC rotation parameter.
    Gradient of the target output w.r.t. self.quantum_stream.vqc.theta.

    Returns:
        dict: theta_attribution [layers × qubits × gates (RX,RY,RZ)], layer_scores, qubit_scores, gate_type_scores
    """
    model.eval()

    # Ensure theta requires grad
    vqc_theta = model.quantum_stream.vqc.theta
    if not vqc_theta.requires_grad:
        vqc_theta.requires_grad_(True)

    # Zero existing grads
    if vqc_theta.grad is not None:
        vqc_theta.grad.zero_()

    # Forward pass
    logits = model(x_raw, x_scalogram)
    if isinstance(logits, tuple): 
        logits = logits[0]

    if target_class is None:
        target_class = logits.argmax(dim=-1)[0].item()

    # Score of interest
    score = logits[0, target_class]
    score.backward()

    # Extract theta gradient as attribution
    theta_grad = vqc_theta.grad.detach().abs().cpu().numpy()  # (3, n_qubits, 3)

    # Aggregate attribution scores
    layer_scores     = theta_grad.mean(axis=(1, 2))          # (3,)
    qubit_scores     = theta_grad.mean(axis=(0, 2))          # (n_qubits,)
    gate_names       = ['RX', 'RY', 'RZ']
    gate_type_scores = {gate_names[g]: float(theta_grad[:, :, g].mean())
                        for g in range(3)}

    return {
        'theta_attribution': theta_grad,
        'layer_scores':      layer_scores,
        'qubit_scores':      qubit_scores,
        'gate_type_scores':  gate_type_scores,
        'target_class':      target_class,
    }

def per_task_quantum_profile(model, dataloader, n_samples: int = 100, task_name: str = '') -> dict:
    """
    Compute average quantum gate attribution across multiple samples from a task.
    Allows comparing quantum activation profiles across clinical disorders.
    """
    all_theta = []
    count = 0
    for x_raw, x_scal, labels in dataloader:
        for i in range(x_raw.shape[0]):
            if count >= n_samples: 
                break
            attr = quantum_gate_attribution(
                model,
                x_raw[i:i+1].cuda() if torch.cuda.is_available() else x_raw[i:i+1],
                x_scal[i:i+1].cuda() if torch.cuda.is_available() else x_scal[i:i+1]
            )
            all_theta.append(attr['theta_attribution'])
            count += 1
        if count >= n_samples: 
            break

    if len(all_theta) == 0:
        # Fallback empty profile
        mean_theta = np.zeros((3, 4, 3), dtype=np.float32)
    else:
        mean_theta = np.stack(all_theta).mean(axis=0)
        
    return {
        'task':            task_name,
        'mean_theta':      mean_theta,
        'layer_scores':    mean_theta.mean(axis=(1, 2)),
        'qubit_scores':    mean_theta.mean(axis=(0, 2)),
        'gate_type_scores': {
            'RX': float(mean_theta[:, :, 0].mean()),
            'RY': float(mean_theta[:, :, 1].mean()),
            'RZ': float(mean_theta[:, :, 2].mean()),
        }
    }
