from utils import get_df, set_up, str_to_bool
from models import set_model, ModelConfig, set_criterion
from multi_label import train_and_evaluate_model
import argparse
import sys
import random
import numpy as np

np.random.seed(0)
random.seed(0)

def train(args):
    file_manager = set_up(args)

    file_manager.logger.info('Set-up is completed')

    if args.fast_dev_run:
        file_manager.logger.warning('Fast dev run is enabled')

    model_config = ModelConfig(
        model_arg=args.model,
        loss_arg=args.loss,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_mode=args.eval_mode,
        experiment_name=args.experiment_name,
        add_transforms=args.add_transforms,
        optimizer_arg=args.optimizer,
        scheduler_arg=args.scheduler,
        num_cores=args.num_cores,
        test_time_augmentation=args.test_time_augmentation,
        fast_dev_run=args.fast_dev_run,
        checkpoint_path=args.checkpoint_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        normalization=args.normalization
    )

    labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia"
    ]

    model_config.num_labels = len(labels)
    model_config.labels = labels

    model_config.model, model_config.img_size = set_model(
        model_config.model_arg,
        model_config.num_labels,
        model_config.labels
    )

    model_file = f"{file_manager.output_folder}/model-architecture.txt"
    with open(model_file, 'w') as f:
        f.write(str(model_config.model.__repr__()))

    model_config.criterion = set_criterion(model_config)

    file_manager.logger.info(f'{model_config.__str__()}')
    file_manager.logger.info(f'{file_manager.__str__()}')

    train_and_evaluate_model(
        model_config=model_config,
        file_manager=file_manager,
    )

    done_file = f"{file_manager.output_folder}/✅.txt"
    with open(done_file, 'w') as f:
        f.write("done!!")


if __name__ == "__main__":
    model_choices = ['swin', 'vit', 'resnet50', 'alexnet',
                     'densenet121', 'efficientnet', 'chexnet']
    loss_choices = ['mlsm', 'wmlsm', 'bce',
                    'wbce', 'focal', 'wfocal', 'asl', 'twoway']
    optimizer_choice = ['adam', 'sgd', 'adamw']
    scheduler_choice = ['cosineannealinglr',
                        'cycliclr', 'step', 'reduceonplateu', 'custom']
    normalization_choices = ['imagenet', 'none', 'chestx-ray']

    parser = argparse.ArgumentParser(
        description="Arguments for training with pytorch")
    parser.add_argument('-en', '--experiment_name',
                        help="Name of folder output files will be added, also name of run", required=False, default='./output/')
    parser.add_argument(
        "-it", "--idun_time", help="The duration of job set on IDUN", default=None, required=False)
    parser.add_argument("-t", "--eval_mode", help="Test mode?",
                        required=False, default=True)
    parser.add_argument("-m", "--model", choices=model_choices,
                        help="Model to run", required=True)
    parser.add_argument("-e", "--num_epochs",
                        help="Number of epochs", type=int, default=15)
    parser.add_argument("-bs", "--batch_size",
                        help="Batch size", type=int, default=8)
    parser.add_argument("-lr", "--learning_rate",
                        help="Learning rate", type=float, default=0.01)
    parser.add_argument("-l", "--loss", choices=loss_choices,
                        help="Type of loss function used", default="wce")
    parser.add_argument("-a", "--add_transforms",
                        help="Add transforms", default=False, required=False)
    parser.add_argument("-o", "--optimizer", choices=optimizer_choice,
                        help="Type of optimizer to use", default="adamw")
    parser.add_argument("-s", "--scheduler", help="Type of scheduler to use",
                        default="cosineannealinglr", choices=scheduler_choice)
    parser.add_argument("-c", "--num_cores",
                        help="Number of cores to use", default=4, type=int)
    parser.add_argument("-tta", "--test_time_augmentation",
                        help="Test time augmentation", default=False, required=True)
    parser.add_argument("-fdr", "--fast_dev_run",
                        help="Fast dev run", default=False, required=False)
    parser.add_argument("-ckpt", "--checkpoint_path",
                        help="Checkpoint path file of model you want to load", default=None, required=False)
    parser.add_argument("-agb", "--accumulate_grad_batches",
                        help="Accumulate gradient batches", default=1, required=False, type=int)
    parser.add_argument("-norm", "--normalization", choices=normalization_choices,
                        help="Type of normalization to be used", default="imagenet")


    args = parser.parse_args()
    args.eval_mode = str_to_bool(args.eval_mode)
    args.add_transforms = str_to_bool(args.add_transforms)
    args.fast_dev_run = str_to_bool(args.fast_dev_run)
    args.test_time_augmentation = str_to_bool(args.test_time_augmentation)

    train(args)
