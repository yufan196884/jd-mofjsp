"""
This file provides functions to train an agent on a scheduling-problem environment.
By default, the trained model will be evaluated on the test data after training,
by running the test_model_and_heuristic function from test.py.

Using this file requires a training config. For example, you have to specify the algorithm used for the training.

There are several constants, which you can change to adapt the training process:
"""
# OS imports
import argparse
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
import torch

# Functional imports
from src.agents import intermediate_test
from src.environments.environment_loader import EnvironmentLoader
from src.data_generator.task import Task
from src.utils.logger import Logger

from src.agents.train_test_utility_functions import get_agent_class_from_config, load_config, load_data
from src.agents.test_RL import final_evaluation

from torchrl.envs import ParallelEnv


def training(config: dict, data_train: List[List[Task]], data_val: List[List[Task]], logger: Logger) -> None:
    """
    Handles the actual training process.
    Including creating the environment, agent and intermediate_test object. Then the agent learning process is started

    :param config: Training config
    :param data_train: Dataset with instances to be used for the training
    :param data_val: Dataset with instances to be used for the evaluation
    :param logger: Logger object used for the whole training process, including evaluation and testing

    :return: None

    """
    # create Environment
    if config.get('parallel_environments',1) > 1:



        from src.environments.environment_loader import ENVIRONMENT_MAPPER_DICT, DEFAULT_ENVIRONMENT_NAME
        env_name = config.get("environment", DEFAULT_ENVIRONMENT_NAME)
        env = ParallelEnv(config.get('parallel_environments',1), lambda: ENVIRONMENT_MAPPER_DICT[env_name]['class'](config, data_train))
        # check_env_specs(env)
        _ = env.reset()
    else:
        env, _ = EnvironmentLoader.load(config, data=data_train, check_env_agent_compatibility=False)
        
    # create Agent model
    agent = get_agent_class_from_config(config)(env=env, config=config, logger=logger)

    if config.get("resume_from_checkpoint"):
        ckpt_path = config["resume_from_checkpoint"]
        data = torch.load(ckpt_path, map_location="cpu")
        agent.mlp_network.load_state_dict(data["model_params"])
        print(f"Resumed actor weights from {ckpt_path}")

    # create IntermediateTest class to save new optimum model every <n_test_steps> steps
    inter_test = intermediate_test.IntermediateTest(env_config=config,
                                                    n_test_steps=config.get('intermediate_test_interval'),
                                                    data=data_val, logger=logger)

    # Actual "learning" or "training" phase
    agent.learn(total_instances=config['total_instances'], total_timesteps=config['total_timesteps'],
                intermediate_test=inter_test)


def main(config_file_name: dict = None, external_config: dict = None) -> None:
    """
    Main function to train an agent in a scheduling-problem environment.

    :param config_file_name: path to the training config you want to use for training
        (relative path from config/ folder)
    :param external_config: dictionary that can be passed to overwrite the config file elements

    :return: None
    """

    # get config and data
    config = load_config(config_file_name, None)

    # Update config with external_config values if keys match
    if external_config:
        for key, value in external_config.items():
            if value is not None and key in config:
                config[key] = value
            elif value is not None and key not in config:
                config[key] = value

    data = load_data(config)

    # create logger and update config
    logger = Logger(config=config)
    config = logger.config

    # Random seed for numpy as given by config
    np.random.seed(config['seed'])

    # train/test/validation data split
    split_random_seed = config['seed'] if not config.get('overwrite_split_seed', False) else 1111
    train_data, test_data = train_test_split(
        data, train_size=config.get('train_test_split'), random_state=split_random_seed)
    test_data, val_data = train_test_split(
        test_data, train_size=config.get('test_validation_split'), random_state=split_random_seed)
    
    # training
    training(config=config, data_train=train_data, data_val=val_data, logger=logger)

    # evaluate results
    final_evaluation(config=config, test_data=test_data, logger=logger) ###FIXME train data


def get_perser_args():
    """Get arguments from command line"""
    # Arguments for function
    parser = argparse.ArgumentParser(description='Train Agent in Production Scheduling Environment')

    parser.add_argument('-fp', '--config_file_path', type=str, required=True,
                        help='Path to config file you want to use for training')

    parser.add_argument('-s', '--seed', type=int, required=False, default=None,
                        help='Seed of run')
    
    parser.add_argument('-wbn', '--wandb_project', type=str, required=False, default=None,
                        help='Name of the run in weights and biases')
    
    parser.add_argument('-cd', '--config_description', type=str, required=False, default=None,
                        help='Config description to distinguish between different runs')
    parser.add_argument('-mn', '--saved_model_name', type=str, required=False, default=None,
                        help='Config description to distinguish between different runs')
    parser.add_argument('-en', '--experiment_name', type=str, required=False, default=None,
                        help='Name of the experiment for wandb to track')

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # get config_file from terminal input
    parse_args = get_perser_args()
    config_file_path = parse_args.config_file_path

    external_config = vars(parse_args)

    main(config_file_name=config_file_path, external_config=external_config)
