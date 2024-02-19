import pyfiglet
import os
import logging

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_up():
    LOG_DIR = "output"
    result = pyfiglet.figlet_format("Master-thesis B)", font="slant")
    print(result)
    create_directory_if_not_exists(LOG_DIR)

    LOG_FILE = f"{LOG_DIR}/log_file.txt"

    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(LOG_FILE),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    logger.info('Set-up completed')

    return logger
