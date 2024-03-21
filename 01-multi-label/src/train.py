from utils import set_up, str_to_bool, ModelConfig
from models import densenet121, swin, vit
import argparse
import sys


def train(args):
    file_manager = set_up(args)

    model_config = ModelConfig(
        model=args.model,
        loss=args.loss,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_mode=args.test_mode,
        experiment_name=args.experiment_name
    )

    file_manager.logger.info(f'{model_config.__str__()}')
    file_manager.logger.info(f'{file_manager.__str__()}')

    if model_config.model == 'densenet':
        densenet121(
            model_config=model_config,
            file_manager=file_manager,
        )
    elif model_config.model == 'swin':
        swin(
            model_config=model_config,
            file_manager=file_manager,
        )
    elif model_config.model == 'vit':
        vit(
            model_config=model_config,
            file_manager=file_manager,
        )
    else:
        file_manager.logger.error('Invalid model argument')
        sys.exit(1)

    file_manager.logger.info('Training is done')


if __name__ == "__main__":
    model_choices = ['densenet', 'swin', 'vit']
    loss_choices = ['bce_logits', 'multi_label_soft_margin', 'weigthed_focal_loss']

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

    args = parser.parse_args()
    args.test_mode = str_to_bool(args.test_mode)
    train(args)
