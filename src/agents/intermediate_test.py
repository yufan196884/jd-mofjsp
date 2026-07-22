"""
This file provides the IntermediateTest class which is used to run an intermediate test on the current model policy.
If the recent model has the best result it is saved as the new current optimum
"""
from typing import List
import numpy as np
import copy
from src.agents.test_RL import test_model
from src.utils.logger import Logger
from src.data_generator.task import Task
from src.utils.file_handler.model_handler import ModelHandler
from src.agents.metaheuristic.nsga2 import NSGA2Scheduler


class IntermediateTest:
    """
    This object is used to run an intermediate test on the current model policy.
    If the recent model has the best result it is saved as the new current optimum

    :param env_config: Config used to initialize the environment for training
    :param n_test_steps: Number of environment steps between intermediate tests
    :param data: Dataset with instances to be used for the intermediate test
    :param logger: Logger object

    """
    def __init__(self, env_config: dict, n_test_steps: int, data: List[List[Task]], logger: Logger):
        # callback and model
        self.n_test_steps = n_test_steps  # steps between callbacks
        self.last_time_trigger = 0
        self.optimum_rwd = 0  # best reward
        self.results = []  # mean rewards fom all tests during training
        self.file = env_config.get('saved_model_name')
        self.logger = logger
        # test/Env parameter
        self.env_config = env_config
        self.data_test = data

        self.Policy_step_counter = 0

        # Precompute nadir for all instances
        self.nadir_refs = {}
        for i, instance in enumerate(data):
            nsga_tmp = NSGA2Scheduler(instance)
            self.nadir_refs[i] = nsga_tmp.compute_nadir_reference(buffer=1.5)
        print(f"Nadir precomputed for {len(self.nadir_refs)} instances.")

    def on_step(self, num_timesteps, model):
        """
        Run intermediate test
        
        Args:
            num_timesteps: Current number of timesteps
            model: RL model to test
            
        Returns:
            Mean reward from evaluation
        """
        print('#'*5)
        print('#'*5, f'INTERMEDIATE TEST after step {num_timesteps}')
        print('#' * 5)

        # Test current model with greedy evaluation
        evaluation_results = test_model(
            self.env_config, 
            self.data_test, 
            model=model, 
            logger=self.logger,
            plot=False, 
            log_episode=False,
            nadir_refs = self.nadir_refs,
            greedy=True
        )
        # Change name back in wandb config
        self.env_config.update({"saved_model_name": self.file})

        # Extract metrics
        makespan_mean = evaluation_results['mean_makespan']
        total_costs = evaluation_results['mean_total_costs']
        instance_wall_time = evaluation_results['time_mean']
        hypervolume = evaluation_results['hypervolume_mean']
        spacing = evaluation_results['spacing_mean']
        mean_reward = evaluation_results['mean_reward']



        # Log results
        self.logger.record(
            {
                'interm_test/mean_makespan': makespan_mean,
                'interm_test/mean_total_costs': total_costs,
                'interm_test/hypervolume': hypervolume,
                'interm_test/spacing': spacing,
                'interm_test/instance_wall_time': instance_wall_time,
                'interm_test/mean_reward': mean_reward,
            }
        )
        self.logger.dump()

        latest_file = ModelHandler.get_best_model_path(self.env_config, latest=True)
        model.save(latest_file)

        # If first test or reward >= current optimum, reset optimum and save model
        if self.last_time_trigger == 0 or hypervolume >= self.optimum_rwd:
            self.optimum_rwd = hypervolume
            best_model_save_path = ModelHandler.get_best_model_path(self.env_config)
            model.save(best_model_save_path)
        
        self.last_time_trigger = num_timesteps
        
        


        # Return mean reward instead of makespan
        return hypervolume