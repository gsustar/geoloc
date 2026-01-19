import torch
from torch.optim.lr_scheduler import LRScheduler

class WarmupCosineAnnealingLR(LRScheduler):
    """
    Warmup + Cosine Annealing Learning Rate Scheduler using PyTorch's SequentialLR
    
    This scheduler works per-step but takes epoch-based arguments for convenience.
    It automatically converts epochs to steps based on steps_per_epoch.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total number of training epochs
        steps_per_epoch: Number of optimizer steps per epoch
        warmup_start_factor: Starting learning rate factor for warmup (default: 0.01)
        eta_min: Minimum learning rate for cosine annealing (default: 0)
        last_epoch: The index of last step (default: -1)
    """
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, steps_per_epoch,
                 warmup_start_factor=0.01, eta_min=0, last_epoch=-1):
        warmup_steps = warmup_epochs * steps_per_epoch
        max_steps = max_epochs * steps_per_epoch
        
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer=optimizer, 
                    start_factor=warmup_start_factor,
                    total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=max_steps - warmup_steps,
                    eta_min=eta_min
                )
            ],
            milestones=[warmup_steps]
        )
        super().__init__(optimizer, last_epoch)
    
    def step(self, epoch=None):
        self.scheduler.step()
    
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
    
    def get_last_lr(self):
        return self.scheduler.get_last_lr()