"""
This file provides functions to generate scheduling problem instances.

Using this file requires a data_generation config. For example, it is necessary to specify
the type of the scheduling problem.
"""
# OS imports
import random
import argparse

# Config and data handling imports
from src.utils.file_handler.config_handler import ConfigHandler
from src.utils.file_handler.data_handler import DataHandler

# Functional imports
import copy
import numpy as np
from typing import List
from src.data_generator.task import Task
from src.data_generator.sp_factory import SPFactory

from tqdm import tqdm

# constants
SEED = 0


def generate_instances_from_config(config: dict, print_info: bool = False) -> List[List[Task]]:
    """
    Generates a list of raw scheduling instances according to the console

    :param config: Data_generation config
    :param print_info: True if the created instances should be output to the console

    :return: List of raw scheduling problem instances

    """
    # Generate instances
    instances = SPFactory.generate_instances(**config)

    

    if print_info:
        print(f"Setups: {len(instances)}")

    return instances

def main(config_file_name=None, external_config=None):
    # get config
    current_config: dict = ConfigHandler.get_config(config_file_name, external_config, check_against_schema=False)

    # set seeds
    seed = current_config.get('seed', SEED)
    np.random.seed(seed)
    random.seed(seed)

    num_instances = current_config.get('num_instances', 100)
    batch_size = current_config.get('batch_size', 1)

    num_batches = int(np.ceil(num_instances / batch_size))
    current_config['num_instances'] = batch_size  # override num_instances for batching

    DataHandler.reset_instances_data_file(current_config)

    for _ in tqdm(range(num_batches), desc="Generating batches"):
        generated_instances = generate_instances_from_config(current_config)
        # Write resulting instance data to file
        if current_config.get('write_to_file', False):
            DataHandler.save_instances_data_file(current_config, generated_instances)


def get_parser_args():
    """Get arguments from command line."""
    # Arguments for function
    parser = argparse.ArgumentParser(description='Instance generation for scheduling optimization')
    parser.add_argument('-fp', '--config_file_path', type=str, required=True,
                        help='Path to config file you want to use for training')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # get config_file from terminal input
    parse_args = get_parser_args()
    config_file_path = parse_args.config_file_path

    main(config_file_name=config_file_path)
