"""
Self-Supervised Pretraining for Sleep Apnea Classification.

Implements Masked Autoencoding (MAE) for EEG/scalogram pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class MAEPretrainer(nn.Module):
    """
    Masked Autoencoder (MAE) for self-supervised pretraining.
    
    Strategy:
        1. Mask random patches of input (75% masking ratio)
        2. Encoder processes only visible patches
        3. Lightweight decoder reconstructs masked patches
        4. Reconstruction loss: MSE between original and predicted
    
    After pretraining, discard decoder and use encoder for classification.
    
    Args:
        vit_name: ViT variant for encoder
        mask_ratio: Fraction of patches to mask
        decoder_depth: Number of decoder layers
        decoder_dim: Decoder hidden dimension
        patch_size: Patch size (must match ViT)
    """
    
    def __init__(
        self,
        vit_name: str = 'vit_small_patch16_224',
        mask_ratio: float = 0.75,
        decoder_depth: int = 4,
        decoder_dim: int = 256,
        patch_size: int = 16,
    ):
        super().__init__()
        
        if not HAS_TIMM:
            raise ImportError("timm is required: pip install timm")
        
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        # Encoder (ViT)
        self.encoder = timm.create_model(
            vit_name,
            pretrained=False,
            num_classes=0,
        )
        encoder_dim = self.encoder.num_features
        
        # Patch embedding info
        self.num_patches = (224 // patch_size) ** 2  # 196 for 224x224, patch16
        
        # Decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim)
        )
        
        decoder_layers = [
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=8,
                dim_feedforward=decoder_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(decoder_depth)
        ]
        self.decoder_transformer = nn.TransformerEncoder(
            nn.ModuleList(decoder_layers),
            num_layers=decoder_depth,
        )
        
        # Reconstruction head
        self.decoder_pred = nn.Linear(decoder_dim, patch_size ** 2 * 3)  # RGB output
        
        # Initialize decoder positional embedding
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
    
    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """
        Randomly mask patches.
        
        Args:
            x: (batch, num_patches, dim)
            mask_ratio: Fraction to mask
        
        Returns:
            x_masked: (batch, num_visible, dim)
            mask: (batch, num_patches) boolean
            ids_restore: (batch, num_patches) for unmasking
        """
        B, N, D = x.shape
        num_mask = int(N * mask_ratio)
        num_visible = N - num_mask
        
        # Random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep visible patches
        ids_keep = ids_shuffle[:, :num_visible]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        
        # Create binary mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_visible] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches.
        
        Args:
            x: (batch, 3, 224, 224)
        
        Returns:
            (batch, num_patches, patch_dim)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        assert H == W == 224, "Input must be 224x224"
        
        # (B, 3, H/p, p, W/p, p)
        x = x.reshape(B, C, H // p, p, W // p, p)
        # (B, 3, H/p, W/p, p, p)
        x = x.transpose(2, 3).transpose(3, 4)
        # (B, H/p * W/p, 3 * p * p)
        x = x.reshape(B, -1, C * p * p)
        
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to image.
        
        Args:
            x: (batch, num_patches, patch_dim)
        
        Returns:
            (batch, 3, 224, 224)
        """
        B = x.shape[0]
        p = self.patch_size
        C = 3
        
        # (B, H/p * W/p, 3 * p * p)
        x = x.reshape(B, H // p, W // p, C, p, p)
        # (B, 3, H/p, p, W/p, p)
        x = x.transpose(2, 3).transpose(3, 4)
        # (B, 3, H, W)
        x = x.reshape(B, C, H, W)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for MAE pretraining.
        
        Args:
            x: (batch, 3, 224, 224)
        
        Returns:
            loss: Reconstruction loss
            pred: Predicted patches (for visualization)
        """
        B = x.shape[0]
        
        # Patchify
        patches = self.patchify(x)  # (B, N, patch_dim)
        
        # Get encoder embeddings (without positional encoding first)
        # Note: timm ViT applies pos_embed internally, so we need to work around
        
        # For simplicity, use encoder's patch_embed directly
        x_embed = self.encoder.patch_embed(x)  # (B, N, dim)
        
        # Add positional encoding
        pos_embed = self.encoder.pos_embed
        x_embed = x_embed + pos_embed
        
        # Masking
        x_masked, mask, ids_restore = self.random_masking(x_embed, self.mask_ratio)
        
        # Encode visible patches
        for block in self.encoder.blocks:
            x_masked = block(x_masked)
        x_masked = self.encoder.norm(x_masked)  # (B, num_visible, dim)
        
        # Decoder
        x_dec = self.decoder_embed(x_masked)  # (B, num_visible, decoder_dim)
        
        # Append mask tokens
        mask_tokens = self.decoder_pos_embed[:, :x_masked.shape[1], :]
        x_dec = torch.cat([x_dec, mask_tokens], dim=1)
        
        # Restore order
        x_dec = torch.gather(
            x_dec, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_dec.shape[2])
        )
        
        # Add decoder positional encoding
        x_dec = x_dec + self.decoder_pos_embed
        
        # Decode
        x_dec = self.decoder_transformer(x_dec)
        
        # Predict patches
        pred = self.decoder_pred(x_dec)  # (B, N, patch_dim)
        
        # Reconstruction loss (only on masked patches)
        loss = F.mse_loss(pred[mask.bool()], patches[mask.bool()])
        
        return loss, pred
    
    def finetune_classifier(self, num_classes: int = 4) -> nn.Module:
        """
        Create classification model from pretrained encoder.
        
        Args:
            num_classes: Number of output classes
        
        Returns:
            Classification model
        """
        return ViTClassifier(self.encoder, num_classes)


class ViTClassifier(nn.Module):
    """
    ViT classifier for fine-tuning after MAE pretraining.
    """
    
    def __init__(self, encoder: nn.Module, num_classes: int = 4):
        super().__init__()
        
        self.encoder = encoder
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder.num_features),
            nn.Dropout(0.5),
            nn.Linear(encoder.num_features, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)


def create_mae_pretrainer(
    vit_name: str = 'vit_small_patch16_224',
    mask_ratio: float = 0.75,
    decoder_depth: int = 4,
    **kwargs,
) -> nn.Module:
    """
    Factory function for MAE pretrainer.
    
    Args:
        vit_name: ViT variant
        mask_ratio: Masking ratio
        decoder_depth: Decoder layers
    
    Returns:
        MAEPretrainer model
    """
    return MAEPretrainer(
        vit_name=vit_name,
        mask_ratio=mask_ratio,
        decoder_depth=decoder_depth,
        **kwargs,
    )


if __name__ == '__main__':
    # Test MAE pretrainer
    model = create_mae_pretrainer()
    x = torch.randn(4, 3, 224, 224)
    
    loss, pred = model(x)
    print(f"Input: {x.shape}")
    print(f"Reconstruction loss: {loss.item():.4f}")
    print(f"Prediction shape: {pred.shape}")
    
    # Test fine-tuning classifier
    classifier = model.finetune_classifier(num_classes=4)
    y = classifier(x)
    print(f"Classifier output: {y.shape}")
