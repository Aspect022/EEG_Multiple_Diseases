import torch
import numpy as np

def input_saliency(model, x: torch.Tensor, target_class=None) -> np.ndarray:
    """
    Vanilla gradient saliency map.
    
    Args:
        model: any model that returns logits or (logits, gate)
        x:     (B, C, T) input raw EEG
    Returns:
        sal:   (B, C, T) absolute gradient values
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    if isinstance(logits, tuple): 
        logits = logits[0]   # Handle (logits, gate) output
        
    if target_class is None:
        target_class = logits.argmax(dim=-1)
        
    # sum of scores for target class across batch
    score = logits[torch.arange(logits.shape[0]), target_class].sum()
    
    model.zero_grad()
    score.backward()
    
    return x.grad.detach().abs().cpu().numpy()

def integrated_gradients(model, x: torch.Tensor, baseline=None, steps: int = 32) -> np.ndarray:
    """
    Integrated Gradients (Sundararajan et al. 2017) on input signal.
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
        
    grads = []
    # Linear interpolation paths between baseline and input
    for alpha in torch.linspace(0, 1, steps, device=x.device):
        xi = (baseline + alpha * (x - baseline)).clone().detach().requires_grad_(True)
        logits = model(xi)
        if isinstance(logits, tuple):
            logits = logits[0]
            
        score = logits.max(dim=-1).values.sum()
        model.zero_grad()
        score.backward()
        grads.append(xi.grad.detach())
        
    avg_grad = torch.stack(grads).mean(dim=0)
    ig = ((x - baseline) * avg_grad).abs().detach().cpu().numpy()
    return ig

FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}

def channel_band_summary(saliency_map: np.ndarray, freqs: np.ndarray) -> dict:
    """
    Summarise saliency across EEG channels and frequency bands.
    
    Args:
        saliency_map: (B, C, F, T) - CWT scalogram saliency map
        freqs:        (F,) - frequency axis values in Hz
    Returns:
        dict: channel_scores, band_scores, and time_scores
    """
    # Average across batch
    s = saliency_map.mean(axis=0)   # (C, F, T)
    channel_scores = s.mean(axis=(1, 2)).tolist()
    time_scores    = s.mean(axis=(0, 1)).tolist()
    
    band_scores    = {}
    for band, (lo, hi) in FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_scores[band] = float(s[:, mask, :].mean()) if mask.any() else 0.0
        
    return {
        'channel_scores': channel_scores,
        'band_scores':    band_scores,
        'time_scores':    time_scores,
    }
