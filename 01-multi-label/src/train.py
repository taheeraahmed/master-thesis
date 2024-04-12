from utils import get_df, set_up, str_to_bool
from models import set_model, ModelConfig, set_criterion
from multi_label import train_and_evaluate_model
import argparse
import sys


def train(args):
    file_manager = set_up(args)

    file_manager.logger.info('Set-up is completed')

    model_config = ModelConfig(
        model_arg=args.model,
        loss_arg=args.loss,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_mode=args.test_mode,
        experiment_name=args.experiment_name,
        add_transforms=args.add_transforms
    )

    train_df, val_df, test_df, labels, class_weights = get_df(
        file_manager=file_manager, 
    )

    model_config.num_labels = len(labels)
    model_config.labels = labels
    model_config.model, model_config.img_size = set_model(
        model_config.model_arg, 
        model_config.num_labels, 
    )

    model_file =f"{file_manager.output_folder}/model-architecture.txt"
    with open(model_file, 'w') as f:
        f.write(str(model_config.model.__repr__()))

    model_config.criterion = set_criterion(model_config, class_weights)

    file_manager.logger.info(f'{model_config.__str__()}')
    file_manager.logger.info(f'{file_manager.__str__()}')

    train_and_evaluate_model(
            model_config=model_config,
            file_manager=file_manager,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
    )

    done_file = f"{file_manager.output_folder}/✅.txt"
    with open(done_file, 'w') as f:
        f.write("done!!")

    file_manager.logger.info('✅ Training is done')


if __name__ == "__main__":
    model_choices = ['swin', 'vit', 'resnet50', 'resnet34' "alexnet", "densenet121"]
    loss_choices = ['mlsm','wmlsm', 'bce', 'wbce', 'focal', 'wfocal']

    parser = argparse.ArgumentParser(
        description="Arguments for training with pytorch")
    parser.add_argument('-en', '--experiment_name',
                        help="Name of folder output files will be added, also name of run", required=False, default='./output/')
    parser.add_argument(
        "-it", "--idun_time", help="The duration of job set on IDUN", default=None, required=False)
    parser.add_argument("-t", "--test_mode", help="Test mode?",
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

    args = parser.parse_args()
    args.test_mode = str_to_bool(args.test_mode)
    args.add_transforms = str_to_bool(args.add_transforms)
    train(args)
