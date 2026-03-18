"""
Hybrid Fusion Models for Multi-Modal Sleep Staging.

Combines 1D raw EEG signals with 2D scalogram representations
for improved sleep stage classification.

Three fusion strategies:
1. Early Fusion: Concatenate features before classification
2. Late Fusion: Ensemble averaging of predictions
3. Gated Fusion: Confidence-based dynamic routing

References:
    - Multi-modal fusion for EEG classification (Zhang et al., 2023)
    - Dynamic routing networks (Sabour et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


# ==========================================================================
# Early Fusion: Feature Concatenation
# ==========================================================================

class EarlyFusionNetwork(nn.Module):
    """
    Early fusion by concatenating features from 1D and 2D branches.
    
    Architecture:
        1D Branch: Raw EEG → SNN-1D → Features (128-d)
        2D Branch: Scalogram → SNN-2D → Features (512-d)
        Fusion: Concat(128+512) → FC → Classifier → 5 classes
    
    Advantages:
        - Simple and effective
        - Learns cross-modal correlations
        - Single end-to-end training
    
    Args:
        num_classes: Number of output classes (default: 5)
        dim_1d: Feature dimension from 1D branch (default: 128)
        dim_2d: Feature dimension from 2D branch (default: 512)
        fusion_dim: Hidden dimension after fusion (default: 256)
        dropout: Dropout rate (default: 0.3)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        dim_1d: int = 128,
        dim_2d: int = 512,
        fusion_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dim_1d = dim_1d
        self.dim_2d = dim_2d
        self.fusion_dim = fusion_dim
        
        # Fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(dim_1d + dim_2d, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout // 2),
            
            nn.Linear(fusion_dim // 2, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        features_1d: torch.Tensor, 
        features_2d: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with feature concatenation.
        
        Args:
            features_1d: Features from 1D branch (batch, dim_1d)
            features_2d: Features from 2D branch (batch, dim_2d)
        
        Returns:
            Logits (batch, num_classes)
        """
        # Concatenate features
        fused = torch.cat([features_1d, features_2d], dim=1)  # (B, dim_1d+dim_2d)
        
        # Classify
        logits = self.fusion_classifier(fused)
        
        return logits


# ==========================================================================
# Late Fusion: Ensemble Averaging
# ==========================================================================

class LateFusionNetwork(nn.Module):
    """
    Late fusion by ensemble averaging of predictions from 1D and 2D branches.
    
    Architecture:
        1D Branch: Raw EEG → SNN-1D → Logits (5-d)
        2D Branch: Scalogram → SNN-2D → Logits (5-d)
        Fusion: Weighted Average → Final Logits
    
    Advantages:
        - Can use pre-trained 1D and 2D models
        - Interpretable branch contributions
        - Robust to modality failure
    
    Args:
        num_classes: Number of output classes (default: 5)
        weight_1d: Weight for 1D branch (learnable if None)
        weight_2d: Weight for 2D branch (learnable if None)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        weight_1d: Optional[float] = None,
        weight_2d: Optional[float] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Learnable or fixed weights
        if weight_1d is not None and weight_2d is not None:
            # Fixed weights
            self.register_buffer('weight_1d', torch.tensor(weight_1d))
            self.register_buffer('weight_2d', torch.tensor(weight_2d))
            self.learnable_weights = False
        else:
            # Learnable weights (initialized to 0.5 each)
            self.weight_1d = nn.Parameter(torch.tensor(0.5))
            self.weight_2d = nn.Parameter(torch.tensor(0.5))
            self.learnable_weights = True
    
    def forward(
        self, 
        logits_1d: torch.Tensor, 
        logits_2d: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with weighted ensemble averaging.
        
        Args:
            logits_1d: Logits from 1D branch (batch, num_classes)
            logits_2d: Logits from 2D branch (batch, num_classes)
        
        Returns:
            Fused logits (batch, num_classes)
        """
        # Compute softmax probabilities
        probs_1d = F.softmax(logits_1d, dim=1)
        probs_2d = F.softmax(logits_2d, dim=1)
        
        # Normalize weights
        if self.learnable_weights:
            w1 = torch.sigmoid(self.weight_1d)  # Range (0, 1)
            w2 = torch.sigmoid(self.weight_2d)  # Range (0, 1)
            total = w1 + w2
            w1 = w1 / total
            w2 = w2 / total
        else:
            w1 = self.weight_1d
            w2 = self.weight_2d
        
        # Weighted average
        fused_probs = w1 * probs_1d + w2 * probs_2d
        
        # Convert back to logits (log of probabilities)
        fused_logits = torch.log(fused_probs + 1e-8)
        
        return fused_logits
    
    def get_weights(self) -> Tuple[float, float]:
        """Get current branch weights."""
        if self.learnable_weights:
            w1 = torch.sigmoid(self.weight_1d).item()
            w2 = torch.sigmoid(self.weight_2d).item()
            total = w1 + w2
            return w1 / total, w2 / total
        else:
            return self.weight_1d.item(), self.weight_2d.item()


# ==========================================================================
# Gated Fusion: Confidence-Based Dynamic Routing
# ==========================================================================

class GatedFusionNetwork(nn.Module):
    """
    Gated fusion with confidence-based dynamic routing.
    
    Architecture:
        1D Branch: Raw EEG → SNN-1D → Features + Confidence
        Gate: Confidence Score → Route to 2D if low confidence
        2D Branch: Scalogram → SNN-2D → Features (if activated)
        Fusion: Adaptive combination based on confidence
    
    The gating mechanism:
        - High confidence (>threshold): Use 1D only (fast path)
        - Low confidence (<threshold): Activate 2D branch (accurate path)
        - Medium confidence: Weighted combination
    
    This mimics expert decision-making:
        - Easy cases: Quick decision from temporal patterns
        - Hard cases: Consult spectral information for confirmation
    
    Args:
        num_classes: Number of output classes (default: 5)
        dim_1d: Feature dimension from 1D branch (default: 128)
        dim_2d: Feature dimension from 2D branch (default: 512)
        confidence_threshold: Threshold for gating (default: 0.7)
        gate_type: Type of gating ('soft', 'hard', 'adaptive')
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        dim_1d: int = 128,
        dim_2d: int = 512,
        confidence_threshold: float = 0.7,
        gate_type: str = 'adaptive',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim_1d = dim_1d
        self.dim_2d = dim_2d
        self.confidence_threshold = confidence_threshold
        self.gate_type = gate_type
        
        # Confidence estimator (from 1D features)
        self.confidence_net = nn.Sequential(
            nn.Linear(dim_1d, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Feature fusion (when both branches used)
        self.fusion_net = nn.Sequential(
            nn.Linear(dim_1d + dim_2d, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
        # 1D-only classifier
        self.classifier_1d = nn.Sequential(
            nn.Linear(dim_1d, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def estimate_confidence(self, features_1d: torch.Tensor) -> torch.Tensor:
        """
        Estimate confidence from 1D features.
        
        Args:
            features_1d: Features from 1D branch (batch, dim_1d)
        
        Returns:
            Confidence scores (batch, 1) in range [0, 1]
        """
        return self.confidence_net(features_1d)
    
    def forward(
        self, 
        features_1d: torch.Tensor, 
        features_2d: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with confidence-based gating.
        
        Args:
            features_1d: Features from 1D branch (batch, dim_1d)
            features_2d: Features from 2D branch (batch, dim_2d) or None
        
        Returns:
            Tuple of (logits, gate_info)
            - logits: (batch, num_classes)
            - gate_info: Dict with confidence, gate_decision, etc.
        """
        B = features_1d.size(0)
        device = features_1d.device
        
        # Estimate confidence
        confidence = self.estimate_confidence(features_1d)  # (B, 1)
        
        # Get 1D-only predictions
        logits_1d = self.classifier_1d(features_1d)  # (B, num_classes)
        probs_1d = F.softmax(logits_1d, dim=1)
        
        # Gate decision
        gate_info = {
            'confidence': confidence,
            'threshold': self.confidence_threshold,
        }
        
        if self.gate_type == 'hard':
            # Hard gating: binary decision
            use_2d = (confidence < self.confidence_threshold).float()  # (B, 1)
            
            if features_2d is not None and use_2d.sum() > 0:
                # Use 2D for low-confidence samples
                logits_2d = self.fusion_net(
                    torch.cat([features_1d, features_2d], dim=1)
                )
                
                # Blend based on gate
                logits = use_2d * logits_2d + (1 - use_2d) * logits_1d
            else:
                logits = logits_1d
            
            gate_info['use_2d'] = use_2d
            gate_info['gate_type'] = 'hard'
        
        elif self.gate_type == 'soft':
            # Soft gating: continuous weighting
            if features_2d is not None:
                # Always use both, but weight by confidence
                logits_2d = self.fusion_net(
                    torch.cat([features_1d, features_2d], dim=1)
                )
                
                # High confidence → trust 1D more
                # Low confidence → trust fusion more
                weight_1d = confidence
                weight_2d = 1 - confidence
                
                logits = weight_1d * logits_1d + weight_2d * logits_2d
            else:
                logits = logits_1d
            
            gate_info['weight_1d'] = confidence
            gate_info['weight_2d'] = 1 - confidence
            gate_info['gate_type'] = 'soft'
        
        elif self.gate_type == 'adaptive':
            # Adaptive gating: learn to combine
            if features_2d is not None:
                logits_2d = self.fusion_net(
                    torch.cat([features_1d, features_2d], dim=1)
                )
                
                # Adaptive weight based on confidence and uncertainty
                # High confidence: mostly 1D
                # Low confidence: mostly fusion
                adaptive_weight = confidence ** 2  # Emphasize high confidence
                logits = adaptive_weight * logits_1d + (1 - adaptive_weight) * logits_2d
            else:
                logits = logits_1d
            
            gate_info['adaptive_weight'] = confidence ** 2
            gate_info['gate_type'] = 'adaptive'
        
        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")
        
        return logits, gate_info


# ==========================================================================
# Factory Functions
# ==========================================================================

def create_early_fusion(
    num_classes: int = 5,
    dim_1d: int = 128,
    dim_2d: int = 512,
    fusion_dim: int = 256,
) -> EarlyFusionNetwork:
    """Create early fusion network."""
    return EarlyFusionNetwork(
        num_classes=num_classes,
        dim_1d=dim_1d,
        dim_2d=dim_2d,
        fusion_dim=fusion_dim,
    )


def create_late_fusion(
    num_classes: int = 5,
    weight_1d: Optional[float] = None,
    weight_2d: Optional[float] = None,
) -> LateFusionNetwork:
    """Create late fusion network."""
    return LateFusionNetwork(
        num_classes=num_classes,
        weight_1d=weight_1d,
        weight_2d=weight_2d,
    )


def create_gated_fusion(
    num_classes: int = 5,
    dim_1d: int = 128,
    dim_2d: int = 512,
    confidence_threshold: float = 0.7,
    gate_type: str = 'adaptive',
) -> GatedFusionNetwork:
    """Create gated fusion network."""
    return GatedFusionNetwork(
        num_classes=num_classes,
        dim_1d=dim_1d,
        dim_2d=dim_2d,
        confidence_threshold=confidence_threshold,
        gate_type=gate_type,
    )


# ==========================================================================
# Multi-Modal Feature Extractor Wrapper
# ==========================================================================

class MultiModalFeatureExtractor(nn.Module):
    """
    Wrapper to extract features from both 1D and 2D branches.
    
    This is used during training to get features from pre-defined
    1D and 2D models, then apply fusion.
    
    Usage:
        extractor = MultiModalFeatureExtractor(model_1d, model_2d)
        features_1d, features_2d = extractor(x_1d, x_2d)
        logits = fusion_network(features_1d, features_2d)
    """
    
    def __init__(
        self,
        model_1d: nn.Module,
        model_2d: nn.Module,
        extract_1d_from: str = 'fusion',
        extract_2d_from: str = 'avgpool',
    ):
        super().__init__()
        self.model_1d = model_1d
        self.model_2d = model_2d
        self.extract_1d_from = extract_1d_from
        self.extract_2d_from = extract_2d_from
        
        # Hooks for feature extraction
        self.features_1d = None
        self.features_2d = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to extract features."""
        # This is a simplified version - in practice, you'd need to
        # know the exact layer names in your 1D and 2D models
        pass
    
    def forward(
        self, 
        x_1d: torch.Tensor, 
        x_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from both branches.
        
        Args:
            x_1d: Raw EEG signal (batch, 6, 3000)
            x_2d: Scalogram (batch, 3, 224, 224)
        
        Returns:
            Tuple of (features_1d, features_2d)
        """
        # Forward through 1D branch
        # Note: This assumes the models have a method to return features
        # In practice, you'd modify your models to support this
        if hasattr(self.model_1d, 'extract_features'):
            features_1d = self.model_1d.extract_features(x_1d)
        else:
            # Fallback: use the full forward pass and assume it returns features
            # This won't work for classifiers - you need to modify your models
            raise NotImplementedError(
                "Model 1D must have extract_features() method or "
                "you need to modify the forward pass to return features"
            )
        
        # Forward through 2D branch
        if hasattr(self.model_2d, 'extract_features'):
            features_2d = self.model_2d.extract_features(x_2d)
        else:
            raise NotImplementedError(
                "Model 2D must have extract_features() method or "
                "you need to modify the forward pass to return features"
            )
        
        return features_1d, features_2d
