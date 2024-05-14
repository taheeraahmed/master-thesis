from models import ModelConfig
import torch


def set_scheduler(model_config: ModelConfig, optimizer: torch.optim.Optimizer) -> dict:
    model = model_config.model
    scheduler_arg = model_config.scheduler_arg
    if scheduler_arg == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10)
    elif scheduler_arg == 'cycliclr':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0.001, max_lr=0.1)
    elif scheduler_arg == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1)
    elif scheduler_arg == "reduceonplateu":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10)
    return scheduler
