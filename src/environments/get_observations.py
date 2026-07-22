import torch
from tensordict.tensordict import TensorDict
import numpy as np


from torchrl.data import (
    CompositeSpec,
    UnboundedDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

def create_observation_spec(env, observation_type='MLP'):
    """Create observation specifications based on environment configuration"""
    
    if observation_type == 'MLP':
        return create_MLP_observation_spec(env)
    else:
        raise ValueError(f"Unknown observation type: {observation_type}")

def create_MLP_observation_spec(env):
    horizon = env.forecast_horizon
    input_dim = env.max_num_jobs + \
                env.max_num_jobs * env.max_num_machines + \
                env.max_num_jobs * env.max_num_machines + \
                env.max_num_jobs * env.max_num_machines + \
                env.max_num_machines + \
                env.max_num_operations + \
                env.max_num_operations + \
                (horizon * 2) + \
                2 # for Multiple objectives #
    
    base_dict = {
        "input": UnboundedContinuousTensorSpec(
            shape=(1, input_dim), # change size here # for 8x8x5: 291, 9x9x6: 377 for 15x15x10: 1064, 25x25x10: 2649
            dtype=torch.float32
        ),
        "mask": UnboundedDiscreteTensorSpec(
            shape=(1, env.max_num_jobs*env.max_num_machines),
            dtype=torch.bool
        ),
        "Preference": UnboundedContinuousTensorSpec(
            shape=(1,2,),
            dtype=torch.float32
        ),
        "makespan": UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32
        ),
        "Energy_Consumed": UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32
        ),
        "Total_Costs": UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32
        ),
        "Reward_Makespan": UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32
        ),
        "Reward_Energy": UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32
        ),
    }

    return CompositeSpec(
        base_dict,
        reward=env.reward_spec,
        shape=env.batch_size
    )

def get_MLP_observation(env) -> TensorDict:
    """Returns original observation format"""

    time_divisor = env.runtime_max

    # 1. tasks left per job
    num_tasks_per_job = np.zeros(env.num_jobs, dtype=np.int32)
    for task in env.tasks:
        num_tasks_per_job[task.job_index] += 1
    num_tasks_tensor = torch.tensor(num_tasks_per_job, dtype=torch.float32, device=env.device)
    tasks_left_per_job = (env.job_task_state.clone().float() / num_tasks_tensor).to(device=env.device)

    # 2. available machines per next task in job
    available_machines_per_job = torch.zeros(env.num_jobs * env.max_num_machines, dtype=torch.float32, device=env.device)
    for job_id in range(env.num_jobs):
        if (job_id, env.job_task_state[job_id].item()) not in env.task_job_mapping:
            continue
        next_task_in_job = env.tasks[env.task_job_mapping[(job_id, env.job_task_state[job_id].item())]]
        for machine_index in next_task_in_job.machines:
            available_machines_per_job[job_id * env.max_num_machines + machine_index] = 1.0

    # 3. runtimes per next task in job
    runtimes_per_next_task = torch.zeros(env.num_jobs * env.max_num_machines, dtype=torch.float32, device=env.device)
    for job_id in range(env.num_jobs):
        if (job_id, env.job_task_state[job_id].item()) not in env.task_job_mapping:
            continue
        next_task_in_job = env.tasks[env.task_job_mapping[(job_id, env.job_task_state[job_id].item())]]
        for machine_index in next_task_in_job.machines:
            runtimes_per_next_task[job_id * env.max_num_machines + machine_index] = next_task_in_job.runtimes[machine_index] / time_divisor

    # 4. energy consumption per next task in job
    energy_consumption_per_next_task = torch.ones(env.num_jobs * env.max_num_machines, dtype=torch.float32, device=env.device)*-1.0
    for job_id in range(env.num_jobs):
        if (job_id, env.job_task_state[job_id].item()) not in env.task_job_mapping:
            continue
        next_task_in_job = env.tasks[env.task_job_mapping[(job_id, env.job_task_state[job_id].item())]]
        for machine_index in next_task_in_job.machines:
            energy_consumption_per_next_task[job_id * env.max_num_machines + machine_index] = next_task_in_job.machine_running_energy[machine_index] / env.energy_max

    # 5. Standby energy consumption for each machine
    standby_energy_consumption = torch.ones(env.max_num_machines, dtype=torch.float32, device=env.device)*-1.0
    for machine_id in range(env.max_num_machines):
        machine = env.machine_info[machine_id]
        standby_energy = machine.standby_energy
        standby_energy_consumption[machine_id] = standby_energy / env.energy_max

    # 6. Done Flag and Start Time for each Task
    done_flags = torch.zeros(env.max_num_operations, dtype=torch.float32, device=env.device)
    start_times = torch.zeros(env.max_num_operations, dtype=torch.float32, device=env.device)
    for op_idx in range(env.num_operations):
        task = env.tasks[op_idx]
        if task.done:
            done_flags[op_idx] = 1.0
        start_times[op_idx] = task.started if task.started != float('inf') else 0.0
    #normalize start times through mean and std of start times of array
    mean_start_time = torch.mean(start_times)
    std_start_time = torch.std(start_times) if torch.std(start_times) > 0 else 1.0
    start_times = (start_times - mean_start_time) / std_start_time

    # 7. Solarprofile and Energy price "forecasting"
    t = int(env.current_time_step)
    horizon = env.forecast_horizon
    solar_window = env.solar_info[t : t + horizon]
    price_window = env.energy_prices[t : t + horizon]
    solar_norm =  torch.tensor(solar_window, dtype=torch.float32, device=env.device) / env.solar_max
    price_norm = torch.tensor(price_window, dtype=torch.float32, device=env.device) / env.price_max

    lambdas_tensor = torch.as_tensor(env.lambda_weights, dtype=torch.float32, device=env.device)

    # flatten and concatenate features
    data = torch.cat(
        (
            tasks_left_per_job, # max_num_jobs
            available_machines_per_job, # max_num_jobs * max_num_machines
            runtimes_per_next_task, # max_num_jobs * max_num_machines
            energy_consumption_per_next_task, # max_num_jobs * max_num_machines
            standby_energy_consumption, #  max_num_machines
            done_flags, # max_num_operations
            start_times, # max_num_operations
            solar_norm,
            price_norm,
            lambdas_tensor
        ),
        dim=0,
    )

    # Build mask sized to max_num_jobs * max_num_machines (padded)
    raw_mask = env.get_job_machine_action_mask()  # shape: (num_jobs * num_machines,)
    full_mask = torch.zeros(env.max_num_jobs * env.max_num_machines, dtype=torch.bool, device=env.device)
    full_mask[:raw_mask.shape[0]] = raw_mask  # place valid entries at correct positions


    # Create a TensorDict with the observation
    td = TensorDict(
        {
            "input": data.unsqueeze(0),  # change observation here
            "mask": full_mask.unsqueeze(0),
        },
        batch_size=env.batch_size,
        device=env.device
    )
    
    return td

def get_observation(env, use_heuristic=False) -> TensorDict:
    """Returns observation based on the configured format"""
    observation_type = env.config.get('model_type', 'DANIEL')
    
    if observation_type == 'MLP':
        return get_MLP_observation(env)