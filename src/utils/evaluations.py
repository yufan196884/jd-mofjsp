"""
This file provides the EvaluationHandler class, which supports the evaluation of a model testing.
"""
import numpy as np
import torch
import copy
from tensordict.tensordict import TensorDict

class EvaluationHandler:
    """
    This class keeps track of several environment parameters (e.g. makespan, tardiness) during a model testing.
    After a testing, you can call the evaluate_test function
    to compute evaluation metrics across all collected test episodes (e.g. mean, standard deviation).

    You can adapt evaluate_test to compute different or more metrics.
    """
    def __init__(self, collect_tardiness: bool = False) -> None:
        #  test parameters. Valid for a complete test loop by one agent. Including multiple episodes
        self.rewards = []
        self.makespan = []  
        self.energy_consumed = []
        self.total_costs = []
        self.actions_list = []
        self.tasks_list = []
        self.time_taken = []
        self.hypervolumes = []
        self.spacings = []

    def record_environment_episode(self, env, total_reward, time_taken, index, GT_instance=False) -> None:
        """
        Stores all necessary environment parameters from the recent episode

        :param env: Non reset environment object, whose parameters should be recorded
        :param total_reward: Total reward of the episode

        :return: None

        """
        # append data from test run to list
        energy_consumed, total_costs = env.consumed_energy_and_costs()
        if not GT_instance:
            self.makespan.append(env.get_makespan())
            self.energy_consumed.append(energy_consumed.item())
            self.total_costs.append(total_costs.item())
            self.rewards.append(total_reward)
            self.tasks_list.append(env.tasks)
            self.actions_list.append(env.action_history)
            self.time_taken.append(time_taken)
                    
        else:
            max_makespan = 0
            for task in env.tasks:
                if task.finished > max_makespan:
                    max_makespan = task.finished
            self.rewards.append(total_reward)
            self.makespan.append(max_makespan)

    def update_episode_with_solver(self, env) -> None:

        for task in env.tasks:
            env.ends_of_machine_occupancies = torch.zeros(task._n_machines)
            for machine_id in range(task._n_machines):
                env.ends_of_machine_occupancies[machine_id] = max([task.finished for task in env.tasks if task.selected_machine == machine_id], default=0)
            env.action_history = []

        return env

    def evaluate_test(self) -> dict:
        """
        Gets all test_parameter and computes all relevant statistical data for plots and prints

        :return: Dictionary with all specified evaluation metrics

        """
        # Only consider rewards from successful indices
        evaluation_results = {}
        
        # Only consider successful indices for makespan calculation
        evaluation_results['mean_makespan'] = np.mean(self.makespan)
        evaluation_results['std_makespan'] = np.std(self.makespan)

        evaluation_results['mean_total_costs'] = np.mean(self.total_costs)
        evaluation_results['std_total_costs'] = np.std(self.total_costs)

        evaluation_results['time_mean'] = np.mean(self.time_taken)
        evaluation_results['time_std'] = np.std(self.time_taken)

        evaluation_results['hypervolume_mean'] = np.mean(self.hypervolumes)
        evaluation_results['hypervolume_std'] = np.std(self.hypervolumes)

        evaluation_results['spacing_mean'] = np.mean(self.spacings)
        evaluation_results['spacing_std'] = np.std(self.spacings)

        evaluation_results['mean_reward'] = np.mean(self.rewards)
        evaluation_results['reward_std'] = np.std(self.rewards)

        return evaluation_results

    @classmethod
    def add_solver_gap_to_results(cls, results: dict) -> dict:

        if 'solver' in results:

            hypervolume_solver = results['solver']['hypervolume_mean'] 

            for method in results:
                hypervolume_method = results[method]['hypervolume_mean']
                gap = (hypervolume_solver - hypervolume_method) / hypervolume_solver
                results[method]['gap_to_solver'] = gap

        
        return results