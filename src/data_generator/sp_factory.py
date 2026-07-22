"""
Helper function for the instance generation in instance_factory.py.
"""
# Standard package import
from enum import Enum
from typing import List, Tuple
import itertools
import random
import copy

# Library import
import numpy as np

# Functional internal import
from src.data_generator.task import Task
from src.data_generator.Machine import Machine
from concurrent.futures import ProcessPoolExecutor
import functools
from src.data_generator.GreenFJSP_Utils.functions.energy_manager import EnergyManager

class SP(Enum):
    green_fjsp = "_generate_instance_green_fjsp"

    @classmethod
    def is_sp_type_implemented(cls, sp_type: str = "") -> bool:
        return True if sp_type in cls.str_list_of_sp_types_implemented() else False

    @classmethod
    def str_list_of_sp_types_implemented(cls) -> List[str]:
        return [name for name, _ in cls.__members__.items()]

class SPFactory:

    # Cache EnergyManager: {month -> EnergyManager}
    _energy_manager_cache: dict = {}

    @classmethod
    def _get_energy_manager(cls, month: int) -> EnergyManager:
        """
        Returns a cached EnergyManager for the given month.
        Creates a new one only if not yet cached.
        """
        if month not in cls._energy_manager_cache:
            cls._energy_manager_cache[month] = EnergyManager(month=month)
        return cls._energy_manager_cache[month]

    @classmethod
    def generate_instances(cls, num_jobs: int = 2, num_tasks: int = 2, num_machines: int = 2, num_tools: int = 2,
                        num_instances: int = 2, runtimes: List[int] = None, sp_type: str = "jssp",
                        print_info: bool = False, **kwargs) -> List[List[Task]]:
        """
        Creates a list of instances with random values in the range of the input parameters

        :param num_jobs: number of jobs generated in an instance
        :param num_tasks: number of tasks per job generated in an instance
        :param num_machines: number of machines available
        :param num_tools: number of tools available
        :param num_instances: number of instances which are to be generated
        :param runtimes: list of possible runtimes for tasks
        :param sp_type: Scheduling problem type (e.g. "jssp")
        :param print_info: if True additional info printed to console

        :return: List of list of Task instances which together form an instance

        """

        # Default initialization of mutable parameters - code safety.
        if runtimes is None:
            runtimes = [4, 6]

        # Check if implemented SP type is provided and get matching generate instance function.
        assert SP.is_sp_type_implemented(sp_type), \
            f"{sp_type} is not valid, you have to provide a valid sp type: {SP.str_list_of_sp_types_implemented()}\n"
        generate_instance_function = getattr(cls, SP[sp_type].value)

        # Collect arguments for generate instance function
        current_kwargs = locals().copy()
        # Remove class argument from collection
        current_kwargs.pop('cls', None)
        # Remove additional kwargs from collection to pass them individually
        current_kwargs.pop('kwargs', None)

        instances = []
        # Create and collect n instances in parallel

        for _ in range(num_instances):
            instance = generate_instance_function(**current_kwargs, **kwargs)
            instances.append(instance)

        return instances
    

    @classmethod
    def compute_and_set_hashes(cls, instances: List[List[Task]]):
        for instance in instances:
            instance_hash = hash(tuple(instance))
            # set hash attributes of each task
            for task in instance:
                task.instance_hash = instance_hash

    @classmethod
    def _generate_instance_green_fjsp(cls, num_jobs: int, num_tasks: int,
                                  num_machines: int, num_tools: int, **kwargs) -> List[Task]:
        """
        Generates a flex4green instance

        :param task_combinations: List with all possible tasks
        :param num_jobs: number of jobs generated in an instance
        :param num_tasks: number of tasks per job generated in an instance
        :param num_machines: number of machines available
        :param num_tools: number of tools available
        :param kwargs: Unused

        :return: flex4green instance (List of tasks)

        """

        # get parameters
        runtime_min, runtime_max = kwargs['runtime_range']
        
        # create instance
        # Parameters (edit sigma to adjust spread)

        jobs_config = kwargs.get('jobs_config', {})
        task_config = kwargs.get('task_config', {})
        machine_config = kwargs.get('machines_config', {})

        machine_running_energy_config = kwargs.get('Machine_Running_Consumption', {})
        machine_Standby_energy_config = kwargs.get('Machine_Standby_Consumption', {})
        Solar_Energy_config = kwargs.get('Solar_Power_Profile', {})
        Energy_info_config = kwargs.get('Energy_info', {})

        num_jobs = get_gaussian_sample(jobs_config,size=1)

        instance_max_machine = get_gaussian_sample(machine_config, size=1)


        def get_random_machines(multi_machine_prob, num_machines):
            if random.random() < multi_machine_prob:
                num_machines_task = np.random.randint(1, num_machines+1)
                machines = np.random.choice(np.arange(num_machines), num_machines_task, replace=False)
            else:
                machines = [np.random.randint(0, num_machines)]
            return machines

        # get parameters
        runtime_min, runtime_max = kwargs['runtime_range']
        multi_machine_prob = kwargs['multi_machine_prob']
        
        # create instance
        tasks_per_job = [1] if num_jobs == 1 else get_gaussian_sample(task_config, size=num_jobs)

        job_depends_on = [[]for j in range(num_jobs)]
        machines = [[get_random_machines(multi_machine_prob, instance_max_machine) for _ in range(tasks_per_job[j])] for j in range(num_jobs)]
        chosen_machines = [[np.random.choice(task) for task in tasks] for tasks in machines]
        durations = []
        for j in range(num_jobs):
            job_durations = []
            for t in range(tasks_per_job[j]):
                task_durations = []
                for m in range(instance_max_machine):
                    if m in machines[j][t]:
                        duration = np.random.randint(runtime_min, runtime_max + 1)
                    else:
                        duration = 0
                    task_durations.append(duration)
                job_durations.append(task_durations)
            durations.append(job_durations)

        # simulate scheduling to gain a gannt chart
        all_tasks_dict = {}                             # dictionary to store all tasks. key: (job, task), value: task object
        tasks_left_per_job = np.ones(num_jobs, dtype=int)          # binary mask for tasks left: 1 if task(s) left, 0 if not
        is_job_available = np.zeros(num_jobs, dtype=int)           # binary mask for available jobs: 1 if job is available, 0 if not
        current_task_per_job = np.zeros(num_jobs, dtype=int)       # current task index for each job

        # loop until all tasks are done
        while np.sum(tasks_left_per_job) > 0:

            # update job availability: check if all dependencies are done
            is_job_available = [all(tasks_left_per_job[dep] == 0 for dep in job_depends_on[j]) for j in range(num_jobs)]
            # update job availability: check if there are tasks left
            is_job_available = is_job_available * tasks_left_per_job

            # choose an available job randomly
            job = int(np.random.choice(np.arange(num_jobs)[is_job_available == 1]))

            # get info
            task = int(current_task_per_job[job])
            machine = chosen_machines[job][task]
            duration = durations[job][task]

            Machine_Costs = []
            for d in duration:
                if d > 0:
                    Sampled_Consumption = get_gaussian_sample(machine_running_energy_config, size=1)
                    Costs = machine_running_energy_config.get('scaling_factor', 1) * Sampled_Consumption
                    Machine_Costs.append(Costs)
                else:
                    Machine_Costs.append(-1)

            # create task object
            task_obj = Task(
                job_index=job,
                task_index=task,
                machines=machines[job][task],
                done=False,
                runtime=duration[machine],
                _n_machines=instance_max_machine,
            )
            all_tasks_dict[(job, task)] = task_obj

            # Added runtime for each machine
            task_obj.runtimes = duration

            # Added energy consumption for each machine
            task_obj.machine_running_energy = Machine_Costs

            # update "environment"
            current_task_per_job[job] += 1
            tasks_left_per_job[job] = int(current_task_per_job[job] != tasks_per_job[job])

        # convert dict to list
        instance = [all_tasks_dict[key] for key in sorted(all_tasks_dict.keys())]

        instance = compute_and_set_hash(instance)

        # Generate Standyby energy for each machine
        Machines = []
        for m in range(instance_max_machine):
            standby_energy = get_gaussian_sample(machine_Standby_energy_config, size=1)
            standby_energy = machine_Standby_energy_config.get('scaling_factor', 1) * standby_energy
            machine_obj = Machine(standby_energy=standby_energy)
            Machines.append(machine_obj)

        # Generate Solar Power Profile
        #energy_manager = EnergyManager(month = Solar_Energy_config.get('target_month', 6))
        month = Solar_Energy_config.get('target_month', 6)
        energy_manager = cls._get_energy_manager(month)
        solar_segments = []
        price_segments = []

        # Generate Profiles for x days (x is specified in "config_GreenFJSP.yaml")
        loading_ahead = Energy_info_config.get('loading_days', 5)
        for _ in range(int(loading_ahead)):
            df_solar = energy_manager.get_solar_energy()
            df_energy_prices = energy_manager.get_energy_prices()
            solar_segments.append(energy_manager.convert_minutes_to_industrial_minutes(df_solar))
            price_segments.append(energy_manager.convert_minutes_to_industrial_minutes(df_energy_prices))


        df_solar_extended = np.concatenate(solar_segments)
        df_energy_prices_extended = np.concatenate(price_segments)

        # Generate Offset to unsure that instances start at different time
        start_offset = np.random.randint(0, 2400)

        Instance_complete = {
            'tasks': instance,
            'machines': Machines,
            'solar_profile': df_solar_extended[start_offset:],
            'energy_prices': df_energy_prices_extended[start_offset:],
            'start_offset' : start_offset
        }

        return Instance_complete

def get_random_machines(multi_machine_prob, num_machines):
    if random.random() < multi_machine_prob:
        num_machines_task = np.random.randint(1, num_machines+1)
        machines = np.random.choice(np.arange(num_machines), num_machines_task, replace=False)
    else:
        machines = [np.random.randint(0, num_machines)]
    return machines

def get_gaussian_sample(config, size=1):
    """
    Generates a sample from a truncated Gaussian distribution.
    """
    
    num_variable = config.get('mean', 10)
    min_num = config.get('min', 1)
    max_num = config.get('max', 10)
    sigma = config.get('sigma', 3)
    min_probability = config.get('min_probability', 0.05)
    max_probability = config.get('max_probability', 0.1)

    mu = num_variable-min_num if num_variable > min_num else min_num       # center of gaussian
    
    xs = np.arange(1, max_num + 1)           # integers 1..40
    p = np.zeros_like(xs, dtype=float)

    # endpoints
    p[0] = min_probability
    p[-1] = max_probability

    # gaussian weights for 2..39
    middle = xs[1:-1]
    gauss = np.exp(-0.5 * ((middle - mu) / sigma) ** 2)
    # normalize gaussian to sum to 0.8
    gauss = gauss * ((1-p[0]-p[-1]) / gauss.sum())
    p[1:-1] = gauss

    # get parameters
    if size == 1:
        sample = int(np.random.choice(xs, p=p)) if num_variable > 1 else 1
        if sample < min_num:
            sample = min_num
        if sample > max_num:
            sample = max_num
    else:
        sample = np.random.choice(xs, p=p, size=size) if num_variable > 1 else 1
        sample = np.clip(sample, min_num, max_num)
        sample = sample.tolist()

    return sample

def compute_and_set_hash(instance: List[Task]):
    instance_hash = hash(tuple(instance))
    # set hash attributes of each task
    for task in instance:
        task.instance_hash = instance_hash
    return instance