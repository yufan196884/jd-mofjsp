"""
This module provides the following scheduling heuristics as function:

- EDD: earliest due date
- SPT: shortest processing time first
- MTR: most tasks remaining
- LTR: least tasks remaining
- Random: random action

You can implement additional heuristics in this file by specifying a function that takes a list of tasks and an action
mask and returns the index of the job to be scheduled next.

If you want to call your heuristic via the HeuristicSelectionAgent or edit an existing shortcut,
adapt/extend the task_selection dict attribute of the HeuristicSelectionAgent class.

:Example:

Add a heuristic that returns zeros (this is not a practical example!)
1. Define the according function

.. code-block:: python

    def return_0_heuristic(tasks: List[Task], action_mask: np.array) -> int:
        return 0

2. Add the function to the task_selection dict within the HeuristicSelectionAgent class:

.. code-block:: python

    self.task_selections = {
        'rand': random_task,
        'EDD': edd,
        'SPT': spt,
        'MTR': mtr,
        'LTR': ltr,
        'ZERO': return_0_heuristic
    }

"""
import numpy as np
from typing import List

from src.data_generator.task import Task


def get_active_task_dict(tasks: List[Task]) -> dict:
    """
    Helper function to determining the next unfinished task to be processed for each job

    :param tasks: List of task objects, so one instance

    :return: Dictionary containing the next tasks to be processed for each job

    Would be an empty dictionary if all tasks were completed

    """
    active_job_task_dict = {}
    for task_i, task in enumerate(tasks):
        if not task.done and task.job_index not in active_job_task_dict.keys():
            active_job_task_dict[task.job_index] = task_i
    return active_job_task_dict


def random_operation_machine_pair(env) -> int:
    """
    Returns a random task

    :param tasks: Not needed
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    
    action_mask = env.get_job_machine_action_mask().squeeze()

    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]
    
    a = np.random.choice(available_actions)

    return a

def least_processing_time(env) -> int:
    """
    Returns the task with the least processing time

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    action_mask = env.get_job_machine_action_mask().squeeze()

    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]

    least_processing_time = np.inf
    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        # If no candidate found or operation can't run on this machine, 
        # this is an invalid action but we'll handle that later

        task = env.tasks[operation_index]
        processing_time = task.runtimes[machine_idx]
        if processing_time < least_processing_time:
            least_processing_time = processing_time
            min_index = action


    return min_index

def most_tasks_remaining(env) -> int:
    """
    Returns the task with the most remaining tasks

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    action_mask = env.get_job_machine_action_mask().squeeze()
    max_remaining_tasks = -1
    least_processing_time = np.inf

    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]

    tasks_per_job = env.num_tasks_per_job
    tasks_per_jobs_done = env.job_task_state
    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        # If no candidate found or operation can't run on this machine, 
        # this is an invalid action but we'll handle that later
        task = env.tasks[operation_index]
        job_index = task.job_index
        num_remaining_tasks = tasks_per_job[job_index] - tasks_per_jobs_done[job_index]
        if num_remaining_tasks > max_remaining_tasks:
            max_remaining_tasks = num_remaining_tasks
            least_processing_time = task.runtimes[machine_idx]
            max_index = action
        elif num_remaining_tasks == max_remaining_tasks:
            if task.runtimes[machine_idx] < least_processing_time:
                least_processing_time = task.runtimes[machine_idx]
                max_index = action

    return max_index

def least_tasks_remaining(env) -> int:
    """
    Returns the task with the least remaining tasks

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    action_mask = env.get_job_machine_action_mask().squeeze()
    min_index = -1
    min_remaining_tasks = np.inf
    least_processing_time = np.inf
    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]

    tasks_per_job = env.num_tasks_per_job
    tasks_per_jobs_done = env.job_task_state
    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        # If no candidate found or operation can't run on this machine, 
        # this is an invalid action but we'll handle that later
        task = env.tasks[operation_index]
        job_index = task.job_index
        num_remaining_tasks = tasks_per_job[job_index] - tasks_per_jobs_done[job_index]
        if num_remaining_tasks < min_remaining_tasks:
            min_remaining_tasks = num_remaining_tasks
            least_processing_time = task.runtimes[machine_idx]
            min_index = action
        elif num_remaining_tasks == min_remaining_tasks:
            if task.runtimes[machine_idx] < least_processing_time:
                least_processing_time = task.runtimes[machine_idx]
                min_index = action

    return min_index

def least_setup_time(env) -> int:
    """
    Returns the task with the least setup time. Only works on an environment with setup times defined.

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """

    # Find indices where action_mask is 0 (available actions)
    action_mask = env.get_job_machine_action_mask().squeeze()

    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]
    least_processing_time = np.inf
    least_setup_time = np.inf

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]
    
    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_index = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        task = env.tasks[operation_index]
        for prev_machine_index, prev_task in enumerate(env.previous_task_machine):
            if machine_index != prev_machine_index:
                continue
            if prev_task == None:
                least_setup_time = 0
                if task.runtimes[machine_index] < least_processing_time:
                    least_processing_time = task.runtimes[machine_index]
                    least_setup_time_index = action
            else:
                sdst_key = (prev_task.job_index, prev_task.task_index,
                        task.job_index, task.task_index,
                        machine_index)
                setup_time = prev_task.SDST[sdst_key]
                if setup_time < least_setup_time:
                    least_setup_time = setup_time
                    least_setup_time_index = action
                    least_processing_time = task.runtimes[machine_index]
                elif setup_time == least_setup_time:
                    if task.runtimes[machine_index] < least_processing_time:
                        least_processing_time = task.runtimes[machine_index]
                        least_setup_time_index = action

    return least_setup_time_index

def earliest_done_time(env) -> int:
    """
    Returns the task with the earliest done time, i.e. the task that is done the earliest on a particular machine.
    """

    action_mask = env.get_job_machine_action_mask().squeeze()
    earliest_done_time = np.inf
    least_processing_time = np.inf

    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]

    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        # If no candidate found or operation can't run on this machine, 
        # this is an invalid action but we'll handle that later

        task = env.tasks[operation_index]
        end_machine_time = env.ends_of_machine_occupancies[machine_idx]
        if hasattr(task, 'SDST'):
            if env.previous_task_machine[machine_idx] is None:
                setup_time = 0
            else:
                prev_task = env.previous_task_machine[machine_idx]
                sdst_key = (prev_task.job_index, prev_task.task_index,
                    task.job_index, task.task_index,
                    machine_idx)
                # Use get with a default value of 0
                setup_time = prev_task.SDST.get(sdst_key) # Access SDST from the previous task
        else:
            setup_time = 0
        done_time = end_machine_time + setup_time + task.runtimes[machine_idx]
        if done_time < earliest_done_time:
            earliest_done_time = done_time
            min_index = action

    return min_index

def most_work_remaining(env) -> int:
    """
    Returns the task with the most work remaining, i.e. the task whose job has the highest average sum of processing times
    for all remaining tasks.
    """

    action_mask = env.get_job_machine_action_mask().squeeze()
    max_work_remaining = -1
    least_processing_time = np.inf

    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]

    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        # If no candidate found or operation can't run on this machine, 
        # this is an invalid action but we'll handle that later
        task = env.tasks[operation_index]
        job_index = task.job_index
        work_remaining = sum([sum([runtime for runtime in t.runtimes if runtime > 0]) / len([runtime for runtime in t.runtimes if runtime > 0])
                      for t in env.tasks 
                      if t.job_index == job_index and not t.done])
        if work_remaining > max_work_remaining:
            max_work_remaining = work_remaining
            least_processing_time = task.runtimes[machine_idx]
            max_index = action
        elif work_remaining == max_work_remaining:
            if task.runtimes[machine_idx] < least_processing_time:
                least_processing_time = task.runtimes[machine_idx]
                max_index = action

    return max_index

def energy_aware(env) -> int:
    action_mask = env.get_job_machine_action_mask().squeeze()
    available_actions = np.where(action_mask == 1)[0]
    if len(available_actions) == 1:
        return available_actions[0]

    best_action = available_actions[0]
    best_cost = float('inf')
    current_t = int(env.current_time_step)
    price_len = len(env.energy_prices)

    for action in available_actions:
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines

        
        task = None
        for t in env.tasks:
            if t.job_index == job_idx and not t.done:
                task = t
                break
        if task is None:
            continue

        p_load = task.machine_running_energy[machine_idx]
        if p_load <= 0:  
            continue

        runtime = int(task.runtimes[machine_idx])
        if runtime <= 0:
            continue

        start_time = max(
            int(env.machine_availability[machine_idx]),
            int(env.job_completion_time[job_idx]),
            current_t
        )
        end_time = start_time + runtime

        safe_end = min(end_time, price_len)
        if safe_end <= start_time:
            monetary_cost = 0.0
        else:
            solar_slice = np.array(env.solar_info[start_time:safe_end])
            price_slice = np.array(env.energy_prices[start_time:safe_end])
            net_energy = np.maximum(0.0, p_load - solar_slice)
            monetary_cost = float(np.sum(net_energy * price_slice)) / 10000.0  # ct → €

        if monetary_cost < best_cost:
            best_cost = monetary_cost
            best_action = action

    return best_action

def minimum_energy_machine(env) -> int:
    """
    """
    action_mask = env.get_job_machine_action_mask().squeeze()
    available_actions = np.where(action_mask == 1)[0]
 
    if len(available_actions) == 1:
        return available_actions[0]
 
    best_action = available_actions[0]
    best_energy = float('inf')
    best_runtime = float('inf')
 
    for action in available_actions:
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
 
        task = None
        for t in env.tasks:
            if t.job_index == job_idx and not t.done:
                task = t
                break
        if task is None:
            continue
 
        runtime = int(task.runtimes[machine_idx])
        if runtime <= 0:
            continue
 
        p_load = task.machine_running_energy[machine_idx]
        total_energy = p_load * runtime  # Wh (or equivalent unit)
 
        if total_energy < best_energy or (
            total_energy == best_energy and runtime < best_runtime
        ):
            best_energy = total_energy
            best_runtime = runtime
            best_action = action
 
    return best_action

def solar_window(env) -> int:
    """
    """
    action_mask = env.get_job_machine_action_mask().squeeze()
    available_actions = np.where(action_mask == 1)[0]
 
    if len(available_actions) == 1:
        return available_actions[0]
 
    best_action = available_actions[0]
    best_coverage = -float('inf')
    best_energy = float('inf')
    current_t = int(env.current_time_step)
    solar_len = len(env.solar_info)
 
    for action in available_actions:
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
 
        task = None
        for t in env.tasks:
            if t.job_index == job_idx and not t.done:
                task = t
                break
        if task is None:
            continue
 
        runtime = int(task.runtimes[machine_idx])
        if runtime <= 0:
            continue
 
        p_load = task.machine_running_energy[machine_idx]
        if p_load <= 0:
            continue
 
        start_time = max(
            int(env.machine_availability[machine_idx]),
            int(env.job_completion_time[job_idx]),
            current_t
        )
        end_time = start_time + runtime
        safe_end = min(end_time, solar_len)
 
        if safe_end <= start_time:
            solar_coverage = 0.0
        else:
            solar_slice = np.array(env.solar_info[start_time:safe_end])
            # Solarabsorption: min(p_load, solar) pro Zeitschritt summiert
            absorbed = np.sum(np.minimum(p_load, solar_slice))
            # Normiert durch den Gesamtenergiebedarf der Operation
            total_demand = p_load * runtime
            solar_coverage = absorbed / total_demand if total_demand > 0 else 0.0
 
        total_energy = p_load * runtime
        if solar_coverage > best_coverage or (
            solar_coverage == best_coverage and total_energy < best_energy
        ):
            best_coverage = solar_coverage
            best_energy = total_energy
            best_action = action
 
    return best_action

def shortest_processing_time(env) -> int:
    """
    """
    action_mask = env.get_job_machine_action_mask().squeeze()
    available_actions = np.where(action_mask == 1)[0]
 
    if len(available_actions) == 1:
        return available_actions[0]
 
    best_action = available_actions[0]
    best_remaining_work = float('inf')
    best_current_runtime = float('inf')
 
    for action in available_actions:
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
 
        task = None
        for t in env.tasks:
            if t.job_index == job_idx and not t.done:
                task = t
                break
        if task is None:
            continue
 
        # Mean remaining runtime over all remaining operations of the job
        remaining_tasks = [
            t for t in env.tasks
            if t.job_index == job_idx and not t.done
        ]
        if not remaining_tasks:
            continue
 
        avg_remaining = np.mean([
            np.mean([r for r in t.runtimes if r > 0]) if any(r > 0 for r in t.runtimes) else 0.0
            for t in remaining_tasks
        ])
        current_runtime = int(task.runtimes[machine_idx])
 
        if avg_remaining < best_remaining_work or (
            avg_remaining == best_remaining_work and current_runtime < best_current_runtime
        ):
            best_remaining_work = avg_remaining
            best_current_runtime = current_runtime
            best_action = action
 
    return best_action

def least_processing_time_ts(env) -> int:
    """
    Returns the task with the least processing time

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    action_mask = env.get_job_machine_action_mask(ts=True).squeeze()

    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]

    least_processing_time = np.inf
    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        # If no candidate found or operation can't run on this machine, 
        # this is an invalid action but we'll handle that later

        task = env.tasks[operation_index]
        processing_time = task.runtimes[machine_idx]
        if processing_time < least_processing_time:
            least_processing_time = processing_time
            min_index = action


    return min_index

def most_tasks_remaining_ts(env) -> int:
    """
    Returns the task with the most remaining tasks

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    action_mask = env.get_job_machine_action_mask(ts=True).squeeze()
    max_remaining_tasks = -1
    least_processing_time = np.inf

    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]

    tasks_per_job = env.num_tasks_per_job
    tasks_per_jobs_done = env.job_task_state
    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        # If no candidate found or operation can't run on this machine, 
        # this is an invalid action but we'll handle that later
        task = env.tasks[operation_index]
        job_index = task.job_index
        num_remaining_tasks = tasks_per_job[job_index] - tasks_per_jobs_done[job_index]
        if num_remaining_tasks > max_remaining_tasks:
            max_remaining_tasks = num_remaining_tasks
            least_processing_time = task.runtimes[machine_idx]
            max_index = action
        elif num_remaining_tasks == max_remaining_tasks:
            if task.runtimes[machine_idx] < least_processing_time:
                least_processing_time = task.runtimes[machine_idx]
                max_index = action

    return max_index

def least_tasks_remaining_ts(env) -> int:
    """
    Returns the task with the least remaining tasks

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    action_mask = env.get_job_machine_action_mask(ts=True).squeeze()
    min_index = -1
    min_remaining_tasks = np.inf
    least_processing_time = np.inf
    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]

    tasks_per_job = env.num_tasks_per_job
    tasks_per_jobs_done = env.job_task_state
    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        # If no candidate found or operation can't run on this machine, 
        # this is an invalid action but we'll handle that later
        task = env.tasks[operation_index]
        job_index = task.job_index
        num_remaining_tasks = tasks_per_job[job_index] - tasks_per_jobs_done[job_index]
        if num_remaining_tasks < min_remaining_tasks:
            min_remaining_tasks = num_remaining_tasks
            least_processing_time = task.runtimes[machine_idx]
            min_index = action
        elif num_remaining_tasks == min_remaining_tasks:
            if task.runtimes[machine_idx] < least_processing_time:
                least_processing_time = task.runtimes[machine_idx]
                min_index = action

    return min_index

def least_setup_time_ts(env) -> int:
    """
    Returns the task with the least setup time. Only works on an environment with setup times defined.

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """

    # Find indices where action_mask is 0 (available actions)
    action_mask = env.get_job_machine_action_mask(ts=True).squeeze()

    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]
    least_processing_time = np.inf
    least_setup_time = np.inf

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]
    
    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_index = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        task = env.tasks[operation_index]
        for prev_machine_index, prev_task in enumerate(env.previous_task_machine):
            if machine_index != prev_machine_index:
                continue
            if prev_task == None:
                least_setup_time = 0
                if task.runtimes[machine_index] < least_processing_time:
                    least_processing_time = task.runtimes[machine_index]
                    least_setup_time_index = action
            else:
                sdst_key = (prev_task.job_index, prev_task.task_index,
                        task.job_index, task.task_index,
                        machine_index)
                setup_time = prev_task.SDST[sdst_key]
                if setup_time < least_setup_time:
                    least_setup_time = setup_time
                    least_setup_time_index = action
                    least_processing_time = task.runtimes[machine_index]
                elif setup_time == least_setup_time:
                    if task.runtimes[machine_index] < least_processing_time:
                        least_processing_time = task.runtimes[machine_index]
                        least_setup_time_index = action

    return least_setup_time_index

def earliest_done_time_ts(env) -> int:
    """
    Returns the task with the earliest done time, i.e. the task that is done the earliest on a particular machine.
    """

    action_mask = env.get_job_machine_action_mask(ts=True).squeeze()
    earliest_done_time = np.inf
    least_processing_time = np.inf

    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]

    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        # If no candidate found or operation can't run on this machine, 
        # this is an invalid action but we'll handle that later

        task = env.tasks[operation_index]
        end_machine_time = env.ends_of_machine_occupancies[machine_idx]
        if hasattr(task, 'SDST'):
            if env.previous_task_machine[machine_idx] is None:
                setup_time = 0
            else:
                prev_task = env.previous_task_machine[machine_idx]
                sdst_key = (prev_task.job_index, prev_task.task_index,
                    task.job_index, task.task_index,
                    machine_idx)
                # Use get with a default value of 0
                setup_time = prev_task.SDST.get(sdst_key) # Access SDST from the previous task
        else:
            setup_time = 0
        done_time = end_machine_time + setup_time + task.runtimes[machine_idx]
        if done_time < earliest_done_time:
            earliest_done_time = done_time
            min_index = action

    return min_index

def most_work_remaining_ts(env) -> int:
    """
    Returns the task with the most work remaining, i.e. the task whose job has the highest average sum of processing times
    for all remaining tasks.
    """

    action_mask = env.get_job_machine_action_mask(ts=True).squeeze()
    max_work_remaining = -1
    least_processing_time = np.inf

    # Find indices where action_mask is 0 (available actions)
    available_actions = np.where(action_mask == 1)[0]

    if len(available_actions) == 1 and available_actions[0] == len(action_mask) - 1:
        return available_actions[0]

    for action in available_actions:

        # Extract job_idx and machine_idx from the action
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
        # Find the next schedulable operation for this job
        operation_index = None
        # Get all tasks for this job sorted by task_index
        job_tasks = [(idx, task) for idx, task in enumerate(env.tasks) 
                    if task.job_index == job_idx]
        job_tasks = sorted(job_tasks, key=lambda x: x[1].task_index)
        # Find the first non-done task
        for idx, task in job_tasks:
            if not task.done:
                operation_index = idx
                break
        # If no candidate found or operation can't run on this machine, 
        # this is an invalid action but we'll handle that later
        task = env.tasks[operation_index]
        job_index = task.job_index
        work_remaining = sum([sum([runtime for runtime in t.runtimes if runtime > 0]) / len([runtime for runtime in t.runtimes if runtime > 0])
                      for t in env.tasks 
                      if t.job_index == job_index and not t.done])
        if work_remaining > max_work_remaining:
            max_work_remaining = work_remaining
            least_processing_time = task.runtimes[machine_idx]
            max_index = action
        elif work_remaining == max_work_remaining:
            if task.runtimes[machine_idx] < least_processing_time:
                least_processing_time = task.runtimes[machine_idx]
                max_index = action

    return max_index

def energy_aware_ts(env) -> int:
    action_mask = env.get_job_machine_action_mask(ts=True).squeeze()
    available_actions = np.where(action_mask == 1)[0]
    if len(available_actions) == 1:
        return available_actions[0]

    best_action = available_actions[0]
    best_cost = float('inf')
    current_t = int(env.current_time_step)
    price_len = len(env.energy_prices)

    for action in available_actions:
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines

        
        task = None
        for t in env.tasks:
            if t.job_index == job_idx and not t.done:
                task = t
                break
        if task is None:
            continue

        p_load = task.machine_running_energy[machine_idx]
        if p_load <= 0:  
            continue

        runtime = int(task.runtimes[machine_idx])
        if runtime <= 0:
            continue

        start_time = max(
            int(env.machine_availability[machine_idx]),
            int(env.job_completion_time[job_idx]),
            current_t
        )
        end_time = start_time + runtime

        safe_end = min(end_time, price_len)
        if safe_end <= start_time:
            monetary_cost = 0.0
        else:
            solar_slice = np.array(env.solar_info[start_time:safe_end])
            price_slice = np.array(env.energy_prices[start_time:safe_end])
            net_energy = np.maximum(0.0, p_load - solar_slice)
            monetary_cost = float(np.sum(net_energy * price_slice)) / 10000.0  # ct → €

        if monetary_cost < best_cost:
            best_cost = monetary_cost
            best_action = action

    return best_action

def minimum_energy_machine_ts(env) -> int:
    """
    """
    action_mask = env.get_job_machine_action_mask(ts=True).squeeze()
    available_actions = np.where(action_mask == 1)[0]
 
    if len(available_actions) == 1:
        return available_actions[0]
 
    best_action = available_actions[0]
    best_energy = float('inf')
    best_runtime = float('inf')
 
    for action in available_actions:
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
 
        task = None
        for t in env.tasks:
            if t.job_index == job_idx and not t.done:
                task = t
                break
        if task is None:
            continue
 
        runtime = int(task.runtimes[machine_idx])
        if runtime <= 0:
            continue
 
        p_load = task.machine_running_energy[machine_idx]
        total_energy = p_load * runtime  # Wh (or equivalent unit)
 
        if total_energy < best_energy or (
            total_energy == best_energy and runtime < best_runtime
        ):
            best_energy = total_energy
            best_runtime = runtime
            best_action = action
 
    return best_action

def solar_window_ts(env) -> int:
    """
    """
    action_mask = env.get_job_machine_action_mask(ts=True).squeeze()
    available_actions = np.where(action_mask == 1)[0]
 
    if len(available_actions) == 1:
        return available_actions[0]
 
    best_action = available_actions[0]
    best_coverage = -float('inf')
    best_energy = float('inf')
    current_t = int(env.current_time_step)
    solar_len = len(env.solar_info)
 
    for action in available_actions:
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
 
        task = None
        for t in env.tasks:
            if t.job_index == job_idx and not t.done:
                task = t
                break
        if task is None:
            continue
 
        runtime = int(task.runtimes[machine_idx])
        if runtime <= 0:
            continue
 
        p_load = task.machine_running_energy[machine_idx]
        if p_load <= 0:
            continue
 
        start_time = max(
            int(env.machine_availability[machine_idx]),
            int(env.job_completion_time[job_idx]),
            current_t
        )
        end_time = start_time + runtime
        safe_end = min(end_time, solar_len)
 
        if safe_end <= start_time:
            solar_coverage = 0.0
        else:
            solar_slice = np.array(env.solar_info[start_time:safe_end])
            # Solarabsorption: min(p_load, solar) pro Zeitschritt summiert
            absorbed = np.sum(np.minimum(p_load, solar_slice))
            # Normiert durch den Gesamtenergiebedarf der Operation
            total_demand = p_load * runtime
            solar_coverage = absorbed / total_demand if total_demand > 0 else 0.0
 
        total_energy = p_load * runtime
        if solar_coverage > best_coverage or (
            solar_coverage == best_coverage and total_energy < best_energy
        ):
            best_coverage = solar_coverage
            best_energy = total_energy
            best_action = action
 
    return best_action

def shortest_processing_time_ts(env) -> int:
    """
    """
    action_mask = env.get_job_machine_action_mask(ts=True).squeeze()
    available_actions = np.where(action_mask == 1)[0]
 
    if len(available_actions) == 1:
        return available_actions[0]
 
    best_action = available_actions[0]
    best_remaining_work = float('inf')
    best_current_runtime = float('inf')
 
    for action in available_actions:
        job_idx = action // env.num_machines
        machine_idx = action % env.num_machines
 
        task = None
        for t in env.tasks:
            if t.job_index == job_idx and not t.done:
                task = t
                break
        if task is None:
            continue
 
        # Mean remaining runtime over all remaining operations of the job
        remaining_tasks = [
            t for t in env.tasks
            if t.job_index == job_idx and not t.done
        ]
        if not remaining_tasks:
            continue
 
        avg_remaining = np.mean([
            np.mean([r for r in t.runtimes if r > 0]) if any(r > 0 for r in t.runtimes) else 0.0
            for t in remaining_tasks
        ])
        current_runtime = int(task.runtimes[machine_idx])
 
        if avg_remaining < best_remaining_work or (
            avg_remaining == best_remaining_work and current_runtime < best_current_runtime
        ):
            best_remaining_work = avg_remaining
            best_current_runtime = current_runtime
            best_action = action
 
    return best_action


class HeuristicSelectionAgent:
    """
    This class can be used to get the next task according to the heuristic passed as string abbreviation (e.g. EDD).
    If you want to edit a shortcut, or add one for your custom heuristic, adapt/extend the task_selection dict.

    :Example:

    .. code-block:: python

        def my_custom_heuristic():
            ...<function body>...

    or

    .. code-block:: python

        self.task_selections = {
            'rand': random_task,
            'XYZ': my_custom_heuristic
            }

    """

    def __init__(self) -> None:

        super().__init__()
        # Map heuristic ids to corresponding function
        self.task_selections = {
            'rand': random_operation_machine_pair,
            'LPT': least_processing_time,
            'MTR': most_tasks_remaining,
            'LTR': least_tasks_remaining,
            'LST': least_setup_time,
            'EDT': earliest_done_time,
            'MWKR': most_work_remaining,
            'energy': energy_aware,
            'MEM': minimum_energy_machine,
            'SW':  solar_window,

            'FI-LPT': least_processing_time_ts,
            'FI-MTR': most_tasks_remaining_ts,
            'FI-LTR': least_tasks_remaining_ts,
            'FI-LST': least_setup_time_ts,
            'FI-EDT': earliest_done_time_ts,
            'FI-MWKR': most_work_remaining_ts,
            'FI-energy': energy_aware_ts,
            'FI-MEM': minimum_energy_machine_ts,
            'FI-SW':  solar_window_ts,
        }

    def __call__(self, env, task_selection: str) -> int:
        """
        Selects the next heuristic function according to the heuristic passed as string abbreviation
        and the assignment in the task_selections dictionary

        :param tasks: List of task objects, so one instance
        :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic
        :param task_selection: Heuristic string abbreviation (e.g. EDD)

        :return: Index of the job selected according to the heuristic

        """
        choose_task = self.task_selections[task_selection]

        chosen_task = choose_task(env)

        return chosen_task
