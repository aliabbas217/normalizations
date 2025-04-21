import torch
import torch.nn as nn

class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        """
        Group Normalization layer.
        This layer normalizes the input across groups of channels.
        Args:
            num_groups (int): Number of groups to divide the channels into.
            num_channels (int): Number of channels in the input.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        assert num_channels % num_groups == 0, \
            f"num_channels {num_channels} must be divisible by num_groups {num_groups}"
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        """
        Forward pass for Group Normalization.
        Args:
            x (torch.Tensor): Input tensor to normalize.
        Returns:
            torch.Tensor: Normalized tensor.
        """
        N, C = x.shape[0], x.shape[1]
        G = self.num_groups
        
        # Check input dimensions
        assert C == self.num_channels, \
            f"Expected {self.num_channels} channels but got {C} channels"
        
        # Reshape input into groups
        x = x.view(N, G, C // G, *x.shape[2:])  # (N, G, C//G, ...)
        
        # Compute mean and variance over group and spatial dimensions
        dims = tuple(range(2, x.dim()))  # (2, 3, ...) for (N, G, C//G, ...)
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back to original form
        x_norm = x_norm.view(N, C, *x.shape[3:])  # Original shape except channels
        
        # Apply scale and shift
        gamma = self.gamma.view(1, C, *([1]*(x.dim()-2)))  # Match input dimensions
        beta = self.beta.view(1, C, *([1]*(x.dim()-2)))
        
        return gamma * x_norm + beta