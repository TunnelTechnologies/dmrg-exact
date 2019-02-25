import logging
import yaml

from datetime import datetime
from random import Random
from time import time


def get_logger():
    full_code, just_word = code_name()

    # application level logging
    logger = logging.getLogger('dmrg-{}'.format(just_word))
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger, full_code


def read_config(config_path, logger=None):
    with open(config_path, "r") as f:
        config_data = f.read()
        params = yaml.load(config_data)
        if logger:
            logger.info("loaded config from {}".format(config_path))
            logger.info(config_data)
    return params


def random_word():
    with open("training_data/words.txt", "r") as f:
        words = [line.strip() for line in f]
    rng = Random(time())
    return rng.choice(words)


def code_name():
    word = random_word()
    now = datetime.now().replace(microsecond=0).isoformat()
    return now + '-' + word, word