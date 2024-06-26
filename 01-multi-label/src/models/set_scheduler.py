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
            optimizer, base_lr=model_config.learning_rate, max_lr=0.1)
    elif scheduler_arg == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1)
    elif scheduler_arg == "reduceonplateu":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10// 2, mode='min',
            threshold=0.0001, min_lr=0, verbose=True)
    elif scheduler_arg == 'custom':
        def lr_foo(epoch):
            if epoch < model_config.warm_up_step:
                # warm up lr
                lr_scale = 0.1 ** (model_config.warm_up_step - epoch)
            else:
                lr_scale = 0.95 ** epoch

            return lr_scale

        scheduler = torch.optim.LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10)
    return scheduler
