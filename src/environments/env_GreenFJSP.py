from typing import Optional
from src.data_generator.task import Task
from src.environments.Samplers import IndexDrivenSampler, PreferenceSampler
from typing import List, Tuple, Any
from torchrl.envs import EnvBase
import torch
import copy
import numpy as np
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
)

from src.environments.get_observations import create_observation_spec, get_observation as _get_observation
from src.visuals_generator import choose_gantt_chart_plotter
from src.data_generator.sp_factory import SPFactory
from src.agents.train_test_utility_functions import load_config

from src.agents.heuristic.heuristic_agent import HeuristicSelectionAgent

from src.visuals_generator.GreenFJSP_Functions import plot_solar_info, plot_energy_prices, plot_energy_per_timestep, plot_cost_per_timestep, plot_cumulative_energy, plot_cumulative_cost

COMPARE_PRECISION = 1e-6 # used for floating point comparisons

class UnbatchedGreenFJSPEnv(EnvBase):

    """
    This environment evaluates GreenFJSP instances. 
    The logic is that at every available time step (the earliest possible time step an operation can be scheduled), 
    the agent executes an operation on a machine at that time step.
    """

    name = "GreenFJSPEnv"

    def __init__(self,
                 config,
                 data: List[List[Task]],
                 sampler = None,
                 device: str = 'cpu',
                 batch_size: torch.Size = None,
                 run_type_checks: bool = True,
                 allow_done_after_reset: bool = False,
                 instance_index: Optional[int] = None,
                 **kwargs):
    
        self.On_fly_data_generation = config.get('On_fly_train_data', False) if len(data) > 1 else False
        if not self.On_fly_data_generation:
            self.data = data
            if config.get("model_type") == "MLP":
                data_config = config
            else:
                data_config = None
            self.get_max_operations(data, data_config)
            # Handle sampler - either use provided sampler or create a new one
            if sampler is not None:
                self.sampler = sampler
            else:
                sampling_mode = config.get('sampling_mode', 'train')
                self.sampler = IndexDrivenSampler(data, mode=sampling_mode, config=config)
            # Store instance index if specified
            self.fixed_instance_index = instance_index
            del data
        else:
            data_config = load_config(config.get('data_generation_yaml', {}))
            data_config["num_instances"] = 1
            self.data_config = data_config
            self.max_num_jobs = data_config.get('jobs_config', {}).get('max', 50)
            self.max_num_operations = data_config.get('task_config', {}).get('max', 50) * self.max_num_jobs
            self.max_num_machines = data_config.get('machines_config', {}).get('max', 25)
            del data


        super().__init__(
            device=device,
            batch_size=batch_size,
            run_type_checks=run_type_checks,
            allow_done_after_reset=allow_done_after_reset,
        )

        self.batch_size = [1]
        self.config = config
        self.sp = config['sp_type']
        self.environment_max_step_size = config.get('environment_max_step_size', 1)
        self.num_steps_max = config.get('num_steps_max', 1000)
        self.reward_strategy = config.get('reward_strategy', 'dense_makespan_reward')
        self.reward_scale_makespan = config.get('reward_scale_makespan', 1.0)
        self.reward_scale_energy = config.get('reward_scale_energy', 1.0)
        self.fixed_instance_index = instance_index
        energy_cfg = config.get('Energy_info', {})
        self.solar_max = energy_cfg.get('solar_peak_power', 186.0)
        self.price_max = energy_cfg.get('price_max_cap', 60.0)
        self.forecast_horizon = energy_cfg.get('forecast_horizon', 500)

        self.preference_sampler = PreferenceSampler(history_limit=config.get('preference_history_limit', 1000))

        

        self.action_mapping = {
            0: "LPT",
            1: "MTR",
            2: "LTR",
            3: "EDT",
            4: "MWKR",
            5: "energy",
            6: "MEM",
            7: "SW",

            8: "FI-LPT",
            9: "FI-MTR",
            10: "FI-LTR",
            11: "FI-EDT",
            12: "FI-MWKR",
            13: "FI-energy",
            14: "FI-MEM",
            15: "FI-SW",

        }
        self.heuristic_agent = HeuristicSelectionAgent()

        self._make_spec()

    def action_heuristic_mapper(self, action):

        heuristic = self.action_mapping[action]
        action_task_machine = self.heuristic_agent(self, heuristic)
        return action_task_machine

    def _make_spec(self) -> None:
        """
        This method defines the action space, done specification, and observation
        specification for the reinforcement learning environment. The action space
        is bounded by the total number of job-machine combinations, while the
        observation specification is created based on the model type specified
        in the configuration.
        Sets the following instance attributes:
            action_spec (BoundedTensorSpec): Defines the valid action space with
                shape (1,), dtype int64, ranging from 0 to len(self.action_mapping)-1.
            done_spec (BoundedTensorSpec): Boolean specification for episode termination
                with shape (1,) and values 0 or 1.
            observation_spec (TensorSpec): Created by create_observation_spec() based on
                the model_type from config (defaults to 'original').
        Returns:
            None
        """
        if self.config.get("heuristic_action_space", False):
            high_end = len(self.action_mapping)
        else:
            high_end = (self.max_num_jobs * self.max_num_machines)

        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=high_end
        )

        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,2), dtype=torch.float32)
        self.done_spec = BoundedTensorSpec(shape=(1,), dtype=torch.bool, low=0, high=1)

        # Use the function from get_observations.py to create the observation spec
        self.observation_spec = create_observation_spec(self, observation_type = self.config.get('model_type', 'original'))

    def get_observation(self,use_heuristic=False) -> TensorDict:
        """
        Get the current observation from the environment.

        Returns:
            TensorDict: The current state observation as a tensor dictionary.
        """
        return _get_observation(self, use_heuristic)

    def get_max_operations(self, data, from_config=None):
        """
        Calculate maximum number of operations, machines, and jobs across all instances
        """
        max_num_operations = 0
        max_num_machines = 0
        max_num_jobs = 0

        if from_config is not None:
            data_config = load_config(from_config["data_generation_yaml"], {})
            max_num_jobs = data_config.get('jobs_config', {}).get('max', 50)
            max_num_operations = data_config.get('task_config', {}).get('max', 50) * max_num_jobs
            max_num_machines = data_config.get('machines_config', {}).get('max', 25)
            self.data_config = data_config
            self.max_num_operations = max_num_operations
            self.max_num_machines = max_num_machines
            self.max_num_jobs = max_num_jobs
            return

        for instance in data:
            max_num_operations = max(max_num_operations, len(instance))
            job_indices = set()
            
            for task in instance:
                max_num_machines = max(max_num_machines, task._n_machines)
                job_indices.add(task.job_index)
            
            max_num_jobs = max(max_num_jobs, len(job_indices))
            
        self.max_num_operations = max_num_operations
        self.max_num_machines = max_num_machines
        self.max_num_jobs = max_num_jobs

    def get_instance_info(self,instance) -> Tuple[int, int, int]:
        """
        Retrieves information about the instance size and configuration from an instance sample.

        Returns:
            Tuple[int, int, int]: A tuple containing:
                - num_jobs (int): Total number of jobs (max job_index + 1)
                - num_tasks (int): Total number of tasks in the instance
                - num_machines (int): Number of machines available

        Note:
            The number of machines is extracted from the first task in the instance.
        """
        num_jobs = 0
        for i,task in enumerate(instance):
            if i==0:
                num_machines = task._n_machines
            num_jobs = task.job_index if task.job_index > num_jobs else num_jobs
        return num_jobs + 1, len(instance), num_machines

    def reset(self, tensordict: Optional[TensorDict] = None, batch_size=None, lambda_weights=None) -> TensorDict:
        """Reset function to call at the beginning of each episode"""
        if batch_size is None:
            batch_size = self.batch_size if tensordict is None else tensordict.batch_size
        if tensordict is None or tensordict.is_empty():
            tensordict = TensorDict(
                {
                },
                batch_size=batch_size,
                device=self.device
            )
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.to(tensordict.device)
        self.lambda_weights = lambda_weights
        obs = super().reset(tensordict, batch_size=batch_size)
        
        return obs
    
    def _reset(self, td: TensorDict, batch_size=None) -> TensorDict:
        """
        Reset the environment to initial state and return observation TensorDict.
        Args:
            td: TensorDict containing reset parameters (optional)
            batch_size: Batch size for reset (optional)
        Returns:
            TensorDict with initial observation, reward, done flag, and metrics
        """

        if td is None:
            device = self.device
        else:
            device = td.device

        if not self.On_fly_data_generation:
            # Use fixed index if provided, otherwise sample from the sampler
            if self.fixed_instance_index is not None:
                data_index = self.sampler.get_instance(self.fixed_instance_index)
            else:
                data_index = self.sampler.get_instance()
            instance_complete = self.data[data_index]
            instance = instance_complete['tasks']
            machine_info = instance_complete["machines"]
            solar_info = instance_complete["solar_profile"]
            energy_prices = instance_complete["energy_prices"]
            


        else:
            instance_complete = SPFactory.generate_instances(**self.data_config)[0]
            instance = instance_complete['tasks']
            machine_info = instance_complete["machines"]
            solar_info = instance_complete["solar_profile"]
            energy_prices = instance_complete["energy_prices"]

        self.tasks = instance
        self.machine_info = machine_info
        self.solar_info = solar_info
        self.energy_prices = energy_prices
        self.instance_start_offset = instance_complete.get('start_offset', 0)
        self.perform_reset()

        td = self.get_observation()
        td.update(
            {
                "reward": torch.zeros((1,2), dtype=torch.float32),  # Shape [1, 2]
                "Reward_Makespan": torch.tensor(0.0, dtype=torch.float32).unsqueeze(0),  # Shape [1, 1]
                "Reward_Energy": torch.tensor(0.0, dtype=torch.float32).unsqueeze(0),  # Shape [1, 1]
                "done": torch.tensor(False, dtype=torch.bool).unsqueeze(0),  # Shape [1, 1]
                "makespan": torch.tensor(0, dtype=torch.float32).unsqueeze(0),
                "Energy_Consumed": torch.tensor(0, dtype=torch.float32).unsqueeze(0),
                "Total_Costs": torch.tensor(0, dtype=torch.float32).unsqueeze(0),
                "Preference": self.lambda_weights.unsqueeze(0) if type(self.lambda_weights) == torch.Tensor else torch.tensor(self.lambda_weights, dtype=torch.float32).unsqueeze(0)
            }
        )

        return td
    
    def perform_reset(self):

        self.num_operations = len(self.tasks)

        self.num_jobs, self.num_tasks, self.num_machines = self.get_instance_info(self.tasks)

        self.makespan = 0
        self.energy_cost = 0.0
        self.num_steps = 0

        self.current_time_step = 0

        # Find min and max runtimes across all tasks for normalization
        min_runtime = float('inf')
        max_runtime = float('-inf')
        min_energy_consumption = float('inf')
        max_energy_consumption = float('-inf')

        for task in self.tasks:
            for runtime in task.runtimes:
                if runtime > 0:
                    min_runtime = min(min_runtime, runtime)
                    max_runtime = max(max_runtime, runtime)
            for energy in task.machine_running_energy:
                if energy > 0:
                    min_energy_consumption = min(min_energy_consumption, energy)
                    max_energy_consumption = max(max_energy_consumption, energy)

        min_runtime = 0
        self.runtime_min = 0
        self.runtime_max = max_runtime
        self.energy_min = 0
        self.energy_max = max_energy_consumption

        self.action_history = []
        self.task_index_history = []

        # Reset task attributes
        self.num_tasks_per_job = torch.zeros(self.num_jobs, dtype=int)
        for task in self.tasks:
            self.num_tasks_per_job[task.job_index] += 1
            task.done = False
            task.finished = float('inf')
            task.started = float('inf')

        # Important machine and operation information reset
        self.task_job_mapping = {(task.job_index, task.task_index): i for i, task in enumerate(self.tasks)}
        self.tasks_left_per_job = torch.ones(self.num_jobs, dtype=int)          # binary mask for tasks left: 1 if task(s) left, 0 if not
        self.is_job_available = torch.zeros(self.num_jobs, dtype=int)           # binary mask for available jobs: 1 if job is available, 0 if not
        self.job_completion_time = torch.zeros(self.num_jobs, dtype=int)        # tracks the end time of the latest completed task for each job
        self.job_task_state = torch.zeros(self.num_jobs, dtype=int)       # current task index for each job
        self.machine_availability = torch.zeros(self.num_machines, dtype=int)   # current time for each machine: when it will be available next
        self.tasks_per_job = torch.bincount(torch.tensor([task.job_index for task in self.tasks]), minlength=self.num_jobs)
        self.ends_of_machine_occupancies= torch.zeros(self.num_machines, dtype=int)
        self.previous_task_machine: List[Task] = [None]*self.num_machines


        # Set Lambda weights for multi-objective optimization
        if self.lambda_weights is None:
            mode = self.config.get('Objective', {}).get('mode', 'multi')
            if mode == 'single_makespan':
                self.lambda_weights = torch.tensor([1.0, 0.0]) # makespan only
            elif mode == 'single_energy':
                self.lambda_weights = torch.tensor([0.0, 1.0])
            else:
                new_vector = self.preference_sampler.sample_uniform_preference()
                self.preference_sampler.add_to_history(new_vector)
                self.lambda_weights = torch.tensor(new_vector, dtype=torch.float32)

        # Compute baseline using the energy-aware heuristic
        self._compute_heuristic_baseline()


    def _compute_heuristic_baseline(self):
        baseline_env = copy.deepcopy(self)
        heuristic_agent = HeuristicSelectionAgent()
        
        done = False
        max_steps = self.num_tasks * 2
        steps = 0
        
        while not done and steps < max_steps:
            action = heuristic_agent(baseline_env, 'energy')
            job_idx = action // baseline_env.num_machines
            machine_idx = action % baseline_env.num_machines
            operation_index = baseline_env.job_task_state[job_idx].item()
            task = baseline_env.tasks[baseline_env.task_job_mapping[(job_idx, operation_index)]]
            
            start_time = int(max(
                baseline_env.machine_availability[machine_idx],
                baseline_env.job_completion_time[job_idx]
            ))
            end_time = start_time + task.runtimes[machine_idx]
            task.started = start_time
            task.finished = end_time
            task.selected_machine = machine_idx
            task.done = True
            baseline_env.job_task_state[job_idx] += 1
            baseline_env.machine_availability[machine_idx] = end_time
            baseline_env.job_completion_time[job_idx] = end_time
            baseline_env.tasks_left_per_job[job_idx] = int(
                baseline_env.job_task_state[job_idx] != baseline_env.tasks_per_job[job_idx]
            )
            done = baseline_env.check_done()
            if not done:
                baseline_env.get_new_time_stamp()
            steps += 1
    
        self.baseline_makespan = max(float(baseline_env.get_makespan().item()), 1.0)
        _, baseline_costs = baseline_env.consumed_energy_and_costs()
        self.baseline_energy_cost = max(float(baseline_costs.item()), 1.0)

    def step(self, td: TensorDict, heuristic=False) -> TensorDict:
        self._assert_tensordict_shape(td)
        next_preset = td.get("next", None)

        if heuristic:
            self.heuristic = True
        else:
            self.heuristic = False
        next_tensordict = self._step(
            td.clone(), heuristic=heuristic
        )  # NOTE: we clone to avoid recursion error
        next_tensordict = self._step_proc_data(next_tensordict)
        if next_preset is not None:
            next_tensordict.update(next_preset.exclude(*next_tensordict.keys(True, True)))
        td.set("next", next_tensordict)
        return td

    def _step(self, td: TensorDict, heuristic=False) -> TensorDict:
        """
        Execute one step of the environment's dynamics.
        """
        # get action
        action = td["action"].item()
        self.action_history.append(action)

        if not heuristic and self.config.get("heuristic_action_space", False):
            action = self.action_heuristic_mapper(action) 

        # get the indexes, task and runtime
        job_index = action // self.num_machines
        machine_index = action % self.num_machines
        operation_index = self.job_task_state[job_index].item()
        task = self.tasks[self.task_job_mapping[(job_index, operation_index)]]
        runtime = task.runtimes[machine_index]
        self.task_index_history.append(self.task_job_mapping[(job_index, operation_index)])
        assert runtime > 0, f"Task {task} has no runtime on machine {machine_index}!"
        assert not task.done, f"Task {task} is already done!"
        
        # compute start and end time
        start_time = max(                                               # consider the maximum of the following:
            [self.machine_availability[machine_index]] +                # 1. when the machine is available
            [self.job_completion_time[job_index]]                       # 2. when the last task of the job was completed
        )

        start_time = int(start_time)
        end_time = int(start_time) + runtime

        # update task attributes
        task.started = start_time
        task.finished = end_time
        task.selected_machine = machine_index
        task.done = True

        # update environment attributes
        self.ends_of_machine_occupancies[machine_index] = end_time
        self.job_task_state[job_index] += 1
        self.machine_availability[machine_index] = end_time
        self.tasks_left_per_job[job_index] = int(self.job_task_state[job_index] != self.tasks_per_job[job_index])
        self.job_completion_time[job_index] = end_time

        # update stuff
        done = self.check_done()
        # Update time step if not done
        if not done:
            self.get_new_time_stamp()

        energy_consumed, total_costs = self.consumed_energy_and_costs()
        #energy_consumed = torch.tensor(0.0, dtype=torch.float32)
        #total_costs = torch.tensor(0.0, dtype=torch.float32)

        td = self.get_observation(use_heuristic=heuristic)
        reward = self.compute_reward()
        makespan = self.get_makespan().float()
        td.update(
            {
                "reward": torch.tensor([reward]).float(),
                "Reward_Makespan": torch.tensor([reward[0]]).float(),  # Shape [1, 1]
                "Reward_Energy": torch.tensor([reward[1]]).float(),  # Shape [1, 1]
                "done": torch.tensor([done], dtype=torch.bool).unsqueeze(0),
                "makespan": torch.tensor([makespan]),
                "Energy_Consumed": energy_consumed.unsqueeze(0),
                "Total_Costs": total_costs.unsqueeze(0),
                "Preference": self.lambda_weights.unsqueeze(0) if type(self.lambda_weights) == torch.Tensor else torch.tensor(self.lambda_weights, dtype=torch.float32).unsqueeze(0)
           }
        )

        return td

    def get_job_machine_action_mask(self, ts=False):
        """
        Generate a boolean mask indicating which job-machine combinations are valid scheduling actions.
        This method creates a mask tensor where each element corresponds to a potential action
        (scheduling a specific job on a specific machine). An element is True if the action is
        valid (i.e., the job can be scheduled on that machine), and False otherwise.
        A job-machine combination is considered valid if:
        1. The job still has tasks remaining to be scheduled
        2. The current task of the job can be processed on the specified machine
        3. All dependencies of the current task have been completed
        Returns:
            torch.Tensor: A 1D boolean tensor of shape (num_jobs * num_machines,) where True 
                         indicates a valid scheduling action and False indicates an invalid one.
                         The tensor is indexed such that action index = job_index * num_machines + machine_index.
        """

        # Create a mask tensor initialized with 0s/False (cannot schedule)
        mask = torch.zeros((self.num_jobs * self.num_machines), dtype=torch.bool)
        
        # Iterate through all tasks and check if they can be scheduled
        for possible_action in range(self.num_jobs * self.num_machines):
            job_index = possible_action // self.num_machines
            machine_index = possible_action % self.num_machines

            # check if job is available
            if self.tasks_left_per_job[job_index] == False:
                continue

            # check if task can be scheduled on the machine
            operation_index = self.job_task_state[job_index].item()
            task = self.tasks[self.task_job_mapping[(job_index, operation_index)]]
            if machine_index not in task.machines:
                continue

            if ts:
                # check if machine and job is available at current time
                if self.machine_availability[machine_index] > self.current_time_step:
                    continue
                if task.task_index > 0:
                    prev_task_idx = self.task_job_mapping[(job_index, operation_index - 1)]
                    prev_task = self.tasks[prev_task_idx]
                    if not prev_task.done or (prev_task.finished > self.current_time_step):
                        continue

            mask[possible_action] = True  # Set to True (can schedule) if all dependencies are done
        return mask

    def get_new_time_stamp(self):
        """
        Find the next time stamp at which a job-machine pair becomes available and update the current time step.
        """
        min_time = float('inf')
        for m in range(self.num_machines):
            m_end_time = self.ends_of_machine_occupancies[m].item()
            for j in range(self.num_jobs):
                task_idx = self.task_job_mapping.get((j, self.job_task_state[j].item()), None)
                if task_idx is not None:
                    if not self.tasks[task_idx].done:
                        if self.tasks[task_idx].runtimes[m] > 0:
                            # The job-machine pair (j, m) can be scheduled after max(machine availability, job availability)
                            job_machine_available_time = max(m_end_time, self.job_completion_time[j].item())
                            min_time = min(min_time, job_machine_available_time)

        if min_time < float('inf'):
            self.current_time_step = min_time

    def compute_reward(self) -> Any:
        """
        Calculates the reward that will later be returned to the agent. Uses the self.reward_strategy string to
        discriminate between different reward strategies. Default is 'dense_reward'.

        :return: Reward

        """
        if self.reward_strategy == 'dense_makespan_reward':
            # dense reward for makespan optimization according to https://arxiv.org/pdf/2010.12367.pdf
            reward = float(self.makespan - self.get_makespan())
            self.makespan = self.get_makespan()
        
        elif self.reward_strategy == 'dense_makespan_energy_reward':
            new_makespan = self.get_makespan()
            r_makespan = float(self.makespan - new_makespan)   
            self.makespan = new_makespan

            # lambda_weights[0] -> Makespan, lambda_weights[1] -> Energy
            w = self.lambda_weights          # Tensor, shape (2,), summe = 1
            if float(w[1]) > 0:     # for makespan only this will be skipped
                _, new_energy_cost = self.consumed_energy_and_costs()
                new_energy_cost = float(new_energy_cost)
                r_energy = float(self.energy_cost - new_energy_cost) 
                self.energy_cost = new_energy_cost
            else:
                r_energy = 0.0

            r_makespan_norm = r_makespan * self.reward_scale_makespan
            r_energy_norm   = r_energy   * self.reward_scale_energy


            reward = [r_makespan_norm, r_energy_norm]

        elif self.reward_strategy == 'sparse_makespan_energy_reward':
            makespan = self.get_makespan()
            self.makespan = makespan
            if self.check_done():
                w = self.lambda_weights          # Tensor, shape (2,), summe = 1
                if float(w[1]) > 0:
                    _, energy_cost = self.consumed_energy_and_costs()
                    energy_cost = float(energy_cost)
                    reward = -  float(w[0]) * (makespan / self.baseline_makespan) - float(w[1]) * (energy_cost / self.baseline_energy_cost)
                else:
                    reward = - (makespan / self.baseline_makespan) 
            else:
                reward = 0.0

        else:
            raise NotImplementedError(f'The reward strategy {self.reward_strategy} has not been implemented.')

        # reward *= self.reward_scale

        return reward

    def check_done(self) -> bool:
        """
        Check if all jobs are done

        :return: True if all jobs are done, else False

        """
        sum_done = sum([task.done for task in self.tasks])
        return sum_done == len(self.tasks) or self.num_steps == self.num_steps_max

    def get_makespan(self) -> torch.Tensor:
        """
        Returns the current makespan (the time the latest of all scheduled tasks finishes)
        """
        try:
            #return torch.max(self.machine_availability)
            return torch.max(torch.tensor([int(task.finished) for task in self.tasks if task.done], dtype=torch.int))
        except:
            return torch.tensor(0, dtype=torch.float32)


    def consumed_energy_and_costs(self) -> torch.Tensor:
        makespan = int(self.get_makespan().item())
        
        # Standby Energy as Tensor (Shape: [num_machines, 1])
        standby_energies = torch.tensor([m.standby_energy for m in self.machine_info], device=self.device).unsqueeze(1)
        
        # Grid with Standby Energy 
        # Shape: [num_machines, makespan]
        machine_energy_grid = standby_energies.expand(-1, makespan).clone()
        
        for task in self.tasks:
            if task.done:
                m_idx = int(task.selected_machine)
                start = int(task.started)
                end = int(task.finished)
                
                active_energy = task.machine_running_energy[m_idx]
                
                if start < makespan and end > start:
                    # Clamp end to makespan to avoid index errors if makespan estimation was off
                    safe_end = min(end, makespan) 
                    machine_energy_grid[m_idx, start:safe_end] = active_energy

        total_energy_per_timestep = machine_energy_grid.sum(dim=0)

        solar_production = self.solar_info[:makespan]  
        energy_prices = self.energy_prices[:makespan]

        # Net Energy: (Consumption - Solar). Negative Values 0 (ReLu).
        net_energy_per_timestep = torch.relu(total_energy_per_timestep - solar_production)
        
        # Energy per time step in kWh instead Industrial Minute
        energy_per_timestep_kwh = net_energy_per_timestep / 100.0
        costs_per_timestep = energy_per_timestep_kwh * energy_prices # energy_prices in ct/kWh; cost in ct
        cost_per_timestep_euro = costs_per_timestep / 100.0

        net_energy_from_grid = energy_per_timestep_kwh.sum()
        total_costs = cost_per_timestep_euro.sum()

        return net_energy_from_grid.float(), total_costs.float()

    def _set_seed(self, seed: Optional[int]):
        """Set the seed for the environment"""
        rng = torch.manual_seed(seed)
        self.rng = rng

    def render(self, mode='human'):
        """
        Render the current state of the environment
        """
        plotter = choose_gantt_chart_plotter(self.config)
        
        if mode == 'human':
            # Show gantt chart
            plotter.get_gantt_chart_image(self.tasks, show_image=True, return_image=False)
            
        elif mode == 'image':
            # Get gantt chart image
            gantt_chart = plotter.get_gantt_chart_image(self.tasks)
            
            
            # Return both visualizations
            return gantt_chart

        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported")
        
    def get_solar_plot(self, show_plot=False, quality_dpi=150):
        """
        Generate solar profile plot.

        :param show_plot: Whether to display the plot
        :param quality_dpi: DPI quality of the saved figure
        :return: PIL.Image of the plot
        """
        return plot_solar_info(self.solar_info, show_plot=show_plot, quality_dpi=quality_dpi)

    def get_energy_price_plot(self, show_plot=False, quality_dpi=150):
        """
        Generate energy price plot.

        :param show_plot: Whether to display the plot
        :param quality_dpi: DPI quality of the saved figure
        :return: PIL.Image of the plot
        """
        return plot_energy_prices(self.energy_prices, show_plot=show_plot, quality_dpi=quality_dpi)
    
    def get_energy_consumption_plot(self, show_plot=False, quality_dpi=150, makespan=None):
        """
        Generate energy consumption per time step plot.

        :param show_plot: Whether to display the plot
        :param quality_dpi: DPI quality of the saved figure
        :param makespan: Total makespan (if None, will be calculated from tasks)
        :return: PIL.Image of the plot
        """
        return plot_energy_per_timestep(self.tasks, self.machine_info, self.energy_prices, show_plot=show_plot, quality_dpi=quality_dpi, makespan=makespan)
        
    def get_cost_per_timestep(self, show_plot=False, quality_dpi=150, makespan=None):
        """
        Generate cost per time step plot.

        :param show_plot: Whether to display the plot
        :param quality_dpi: DPI quality of the saved figure
        :param makespan: Total makespan (if None, will be calculated from tasks)
        :return: PIL.Image of the plot
        """
        return plot_cost_per_timestep(self.tasks, self.machine_info, self.energy_prices, self.solar_info, show_plot=show_plot, quality_dpi=quality_dpi, makespan=makespan)
    
    def get_cumulative_energy_plot(self, show_plot=False, quality_dpi=150, makespan=None):
        """
        Generate cumulative energy consumption plot.

        :param show_plot: Whether to display the plot
        :param quality_dpi: DPI quality of the saved figure
        :param makespan: Total makespan (if None, will be calculated from tasks)
        :return: PIL.Image of the plot
        """
        return plot_cumulative_energy(self.tasks, self.machine_info, self.energy_prices, show_plot=show_plot, quality_dpi=quality_dpi, makespan=makespan) 
    
    def get_cumulative_cost_plot(self, show_plot=False, quality_dpi=150, makespan=None):
        """
        Generate cumulative cost plot.

        :param show_plot: Whether to display the plot
        :param quality_dpi: DPI quality of the saved figure
        :param makespan: Total makespan (if None, will be calculated from tasks)
        :return: PIL.Image of the plot
        """
        return plot_cumulative_cost(self.tasks, self.machine_info, self.energy_prices, self.solar_info, show_plot=show_plot, quality_dpi=quality_dpi, makespan=makespan)