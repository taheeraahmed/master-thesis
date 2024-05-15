import pyfiglet
import logging
import os
from datetime import datetime
from utils.create_dir import create_directory_if_not_exists
from utils.check_gpu import check_gpu
from datetime import datetime, timedelta
import time


class FileManager():
    def __init__(self, experiment_name, idun_datetime_done):
        self.experiment_name = experiment_name
        self.idun_datetime_done = idun_datetime_done

        self.root = '/cluster/home/taheeraa/code/master-thesis/01-multi-label/output'
        self.output_folder = f'{self.root}/{experiment_name}'

        self.model_ckpts_folder = f'{self.output_folder}/model_checkpoints'
        self.image_folder = f'{self.output_folder}/images'

        self.logger = self._set_up_logger()
        create_directory_if_not_exists(self.model_ckpts_folder)
        self.data_path = '/cluster/home/taheeraa/datasets/chestxray-14'
        self.images_path = self.data_path + '/images'
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))

    def __str__(self):
        table_str = (
            f"🗃️ File Manager Configuration 🗃️\n"
            f"🌲 Root folder: {self.root:<25}\n"
            f"📁 Output Folder: {self.output_folder:<25}\n"
            f"🗂️ Model Checkpoints: {self.model_ckpts_folder:<25}\n"
            f"🖼️ Image Folder: {self.image_folder:<25}\n"
            f"💾 Data Path: {self.data_path:<25}\n"
            f"🕒 IDUN Done Time: {self.idun_datetime_done:<25}\n"
        )
        return table_str

    def __repr__(self):
        return f"FileManager(output_folder={self.output_folder}, logger={self.logger}, idun_datetime_done={self.idun_datetime_done}, model_ckpts_folder={self.model_ckpts_folder}, image_folder={self.image_folder}, data_path={self.data_path})"

    def __eq__(self, other):
        return self.output_folder == other.output_folder

    def _set_up_logger(self):
        create_directory_if_not_exists(self.output_folder)
        LOG_FILE = f"{self.output_folder}/log_file.txt"

        logging.basicConfig(level=logging.INFO,
                            format='[%(levelname)s] %(asctime)s - %(message)s',
                            handlers=[
                                logging.FileHandler(LOG_FILE),
                                logging.StreamHandler()
                            ])
        return logging.getLogger()


def set_up(args):
    start_time = time.time()

    idun_time = args.idun_time
    experiment_name = args.experiment_name

    result = pyfiglet.figlet_format("master-thesis", font="slant")
    print(result)

    # Calculate at what time IDUN job is done
    hours, minutes, seconds = map(int, idun_time.split(":"))
    now = datetime.fromtimestamp(start_time)
    idun_datetime_done = now + \
        timedelta(hours=hours, minutes=minutes, seconds=seconds)

    file_manager = FileManager(
        experiment_name=experiment_name, idun_datetime_done=idun_datetime_done)

    return file_manager


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
