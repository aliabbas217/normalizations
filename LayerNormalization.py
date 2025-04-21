import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Layer Normalization layer.
        This layer normalizes the input across the last dimension.
        Args:
            normalized_shape (int or tuple): Shape of the input to normalize.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """
        Forward pass for Layer Normalization.
        Args:
            x (torch.Tensor): Input tensor to normalize.
        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Calculate dimensions to normalize over
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        # Compute mean and variance
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_hat + self.beta
        return out