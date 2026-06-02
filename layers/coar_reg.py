import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

class C_RegNet(nn.Module):
    """
    Revised transformation matrix prediction network
    Handles variable input sizes using global average pooling
    """
    def __init__(self, in_channels=64):
        super().__init__()
        # Feature reduction before pooling
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(in_channels * 2, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU()
        )
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Transformation predictor
        self.transform_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
        
        # Initialize transformation matrix parameters
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize transformation matrix prediction layer"""
        # Initialize as identity transformation
        self.transform_predictor[-1].weight.data.zero_()
        self.transform_predictor[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, ir_feat, vis_feat):
        """
        Forward pass
        :param ir_feat: IR features [B, C, H, W]
        :param vis_feat: Visible features [B, C, H, W]
        :return: Transformed IR features, transformation matrix
        """
        # Concatenate features along channel dimension
        fused_feat = torch.cat([ir_feat, vis_feat], dim=1)
        
        # Reduce feature dimensions
        reduced_feat = self.feature_reduction(fused_feat)
        
        # Global average pooling
        pooled_feat = self.pool(reduced_feat)
        
        # Flatten for linear layers
        flattened_feat = pooled_feat.view(pooled_feat.size(0), -1)
        
        # Predict transformation parameters
        transform_params = self.transform_predictor(flattened_feat)
        
        # Build affine transformation matrix
        transform_matrix = transform_params.view(-1, 2, 3)

        # Apply transformation to original IR features
        c_reg_ir = apply_transform(ir_feat, transform_matrix, mode='bilinear')
        
        return c_reg_ir, transform_matrix

def apply_transform(image, transform_matrix, mode='bilinear'):
    """
    Apply affine transformation to input features
    :param image: Input features [B, C, H, W]
    :param transform_matrix: Transformation matrix [B, 2, 3]
    :return: Transformed features [B, C, H, W]
    """
    batch_size, channels, height, width = image.size()
    
    # Create grid in normalized coordinates
    grid = F.affine_grid(
        transform_matrix, 
        [batch_size, channels, height, width],
        align_corners=False
    )
    
    # Sample transformed features
    transformed_image = F.grid_sample(
        image, 
        grid, 
        mode=mode, 
        padding_mode='border', 
        align_corners=False
    )
    
    return transformed_image