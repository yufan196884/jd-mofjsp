import torch
import numpy as np

class MakespanLRScheduler:
    """Learning rate scheduler that can work with both makespan and reward metrics"""
    
    def __init__(self, optimizer, factor=0.9, patience=10, min_lr=1e-6, verbose=False):
        """
        Args:
            optimizer: The optimizer to adjust learning rate for
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement after which LR will be reduced
            min_lr: Lower bound on the learning rate
            verbose: If True, prints a message when lr is reduced
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        # Initialize tracking variables
        self.best_score = None
        self.counter = 0
        
    def step(self, metric_value):
        """
        Update scheduler state based on a performance metric
        
        Args:
            metric_value: The value to compare against the best (can be makespan or negative reward)
                          Lower values are considered better
        """
        # Initialize best_score on first call
        if self.best_score is None:
            self.best_score = metric_value
            return
        
        # Check if metric has improved
        if metric_value < self.best_score:
            # Metric improved, reset counter and update best
            self.best_score = metric_value
            self.counter = 0
        else:
            # Metric did not improve, increment counter
            self.counter += 1
            
            # Reduce learning rate if patience is exceeded
            if self.counter >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                    
                    if self.verbose:
                        print(f'Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}')
                
                # Reset counter
                self.counter = 0
    
    def get_lr(self):
        """Return current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_best_makespan(self):
        """Returns best makespan observed so far"""
        return self.best_makespan
    
    def reset(self):
        """Reset the best makespan and counter"""
        self.best_makespan = float('inf')
        self.counter = 0


def explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes fraction of variance that y_pred explains about y_true.
    Returns 1 - Var[y_true-y_pred] / Var[y_true]

    Interpretation:
        ev=1  =>  perfect prediction
        ev=0  =>  might as well have predicted zero
        ev<0  =>  worse than just predicting zero

    Args:
        y_true: the expected values (targets)
        y_pred: the predictions

    Returns:
        Explained variance as a scalar float between -inf and 1
    """
    # Handle dimension issues - ensure both are 1D
    if y_true.dim() > 1:
        y_true = y_true.reshape(-1)
    if y_pred.dim() > 1:
        y_pred = y_pred.reshape(-1)
    
    # Compute variance of y_true
    var_y = torch.var(y_true)
    
    # Handle edge case when variance is 0
    if var_y < 1e-10:
        return torch.nan
    
    # Compute explained variance
    diff_var = torch.var(y_true - y_pred)
    
    # Ensure we return a Python float, not a tensor
    return float(1 - diff_var / var_y)

def generate_distractor_preferences(Preferences, mean=0, std=0.1):
    """
    generate a distractor preference vector based on a input preference vectors and a normal distribution with specified mean and std

    Preferences: tensor of shape (batch_size, 2) where each row is a preference vector for makespan and energy
    """
    # Sample noise from a normal distribution
    noise = torch.normal(mean=mean, std=std, size=Preferences.shape, device=Preferences.device)
    
    # Add noise to the original preferences
    distractor_prefs = Preferences + noise
    
    # Ensure the distractor preferences are still valid (non-negative and sum to 1)
    distractor_prefs = torch.clamp(distractor_prefs, min=0.0)
    row_sums = distractor_prefs.sum(dim=1, keepdim=True)
    distractor_prefs = distractor_prefs / row_sums
    
    return distractor_prefs
