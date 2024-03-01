import pyfiglet
import logging
import os
from datetime import datetime
from utils.create_dir import create_directory_if_not_exists
from utils.check_gpu import check_gpu
from datetime import datetime, timedelta
import time

class ModelConfig():
    def __init__(self, model, loss, num_epochs, batch_size, learning_rate):
        self.model = model
        self.loss = loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def __str__(self):
        return f'model: {self.model}, loss: {self.loss}, num_epochs: {self.num_epochs}, batch_size: {self.batch_size}, learning_rate: {self.learning_rate}'

    def __repr__(self):
        return f'model: {self.model}, loss: {self.loss}, num_epochs: {self.num_epochs}, batch_size: {self.batch_size}, learning_rate: {self.learning_rate}'

    def __eq__(self, other):
        return self.model == other.model and self.loss == other.loss and self.num_epochs == other.num_epochs and self.batch_size == other.batch_size and self.learning_rate == other.learning_rate

def set_up(args):
    start_time = time.time()

    idun_time = args.idun_time
    output_folder = 'output/'+args.output_folder

    result = pyfiglet.figlet_format("Master-thesis B)", font="slant")
    print(result)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))

    LOG_DIR = output_folder

    create_directory_if_not_exists(LOG_DIR)
    create_directory_if_not_exists(f'{output_folder}/model_checkpoints')

    LOG_FILE = f"{LOG_DIR}/log_file.txt"

    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(LOG_FILE),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    logger.info(f'Running: {args.model}')
    logger.info(f'Root directory of project: {project_root}')
    check_gpu(logger)
    logger.info('Set-up completed')

    # Calculate at what time IDUN job is done
    try:
        hours, minutes, seconds = map(int, idun_time.split(":"))
        now = datetime.fromtimestamp(start_time)
        idun_datetime_done = now + \
            timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except:
        logger.info('Didnt get IDUN time')

    return logger, idun_datetime_done, output_folder


def calculate_idun_time_left(epoch, num_epochs, epoch_duration, idun_datetime_done, logger):
    remaining_epochs = num_epochs - (epoch + 1)
    estimated_remaining_time = epoch_duration * remaining_epochs
    # Calculate the estimated completion time
    estimated_completion_time = datetime.now(
    ) + timedelta(seconds=estimated_remaining_time)

    logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")
    logger.info(
        f"Estimated time remaining: {estimated_remaining_time:.2f} seconds")
    logger.info(
        f"Estimated completion time: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        if idun_datetime_done > estimated_completion_time:
            time_diff = idun_datetime_done - estimated_completion_time
            logger.info(f"There is enough time allocated: {time_diff}")
        else:
            time_diff = estimated_completion_time - idun_datetime_done
            logger.warning(
                f"There might not be enough time allocated on IDUN: {time_diff}")
    except:
        logger.info('Dont have IDUN time')


def str_to_bool(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert {value} to boolean.")
