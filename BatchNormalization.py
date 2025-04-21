import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        """
        Batch Normalization layer.
        Args:
            num_features (int): Number of features (channels) in the input.
            momentum (float): Momentum for the running mean and variance.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics (buffers)
        # we used buffers at another place, do you remember it?
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        """
        Forward pass for batch normalization.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) for 2D data
                              or (N, C) for 1D data.
        Returns:
            torch.Tensor: Normalized output tensor.
        """
        if self.training:
            # Compute batch mean and variance
            # x is of shape (N, C, H, W) for 2D data
            # or (N, C) for 1D data
            # dim=0 means we are computing the mean and variance across the batch dimension
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            # Use running statistics during evaluation
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        # Normalize
        x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_hat + self.beta
        
        return out