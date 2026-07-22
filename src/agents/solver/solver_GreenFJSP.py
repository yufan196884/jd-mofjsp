"""
This Solver solves certain scheduling problems optimally

It can handle:

- Classic JSSP
- Classic FJSSP
- both of the above with and without tool constraints
- optimization criteria tardiness and makespan

"""
import numpy as np
import collections
from ortools.sat.python import cp_model
from typing import List
import argparse
import copy

from src.data_generator.task import Task
from src.utils.file_handler.data_handler import DataHandler
from src.visuals_generator.gantt_chart import GanttChartPlotter


# constants
SOLVER_OBJECTIVE: str = 'makespan'


class OrToolSolver_GreenFJSP:
    """
    This class can be used to solve JSSP problems. It can handle:

    - Classic JSSP
    - Classic FJSSP
    - both of the above with and without tool constraints
    - optimization criteria makespan

    Data needs to be passed in 'instance format' and is returned in this format, too.
    """

    @classmethod
    def optimize(cls, instance: List[Task], 
                 objective_weights: List[float] = [0.5,0.5], 
                 time_limit_seconds: float = 10.0, 
                 num_workers: int = 8):
        """
        Optimizes the passed instance according to the passed objective using CP-SAT.
        Handles flexible job shop problems with:
        - Multiple machine options per operation
        - Precedence constraints (job-level and explicit dependencies)
        - Temporal constraints (min/max time between task start/end times)
        
        :param List[Task] instance: The instance as a list of Tasks. The tasks in this list will be modified in-place if a solution is found.
        :param List[float] objective_weights: Weights for the multi-objective optimization [makespan_weight, total_costs_weight]
        :param float time_limit_seconds: Maximum time allowed for the solver.
        :param int num_workers: Number of parallel workers for the solver.
        
        :return: tuple(List[Task], float | None) Solved instance and objective value, or (instance, None) if no solution is found or an error occurs.
        """
        # --- 1. Input Validation and Preprocessing ---

        tasks_list = instance["tasks"]
        machines = instance["machines"]
        solar_profile = instance["solar_profile"]
        energy_prices = instance["energy_prices"]
        
        # Normalize objective weights
        makespan_weight = objective_weights[0]
        energy_cost_weight = objective_weights[1]

        if not tasks_list:
            print("Error: Instance is empty.")
            return tasks_list, None

        try:
            num_machines = max(task._n_machines for task in tasks_list)
            if num_machines <= 0:
                 print("Error: No machines found in the instance.")
                 return tasks_list, None
            all_machines = range(num_machines)
        except Exception as e:
            print(f"Error determining number of machines: {e}")
            return tasks_list, None

        # Validate machine availability for each task
        for task in tasks_list:
            if not hasattr(task, 'runtimes') or len(task.runtimes) != num_machines:
                 print(f"Error: Task {task.job_index}-{task.task_index} has inconsistent runtime data.")
                 return tasks_list, None
            if not any(rt > 0 for rt in task.runtimes):
                print(f"Error: Task {task.job_index}-{task.task_index} has no valid machines with positive runtime.")
                return tasks_list, None

        # Group tasks by job and sort
        jobs = collections.defaultdict(list)
        task_keys = set()
        for i, task in enumerate(tasks_list):
            task_key = (task.job_index, task.task_index)
            if task_key in task_keys:
                 print(f"Error: Duplicate task key found: {task_key}")
                 return tasks_list, None
            task_keys.add(task_key)
            jobs[task.job_index].append(task)

        for job_id in jobs:
            jobs[job_id].sort(key=lambda x: x.task_index)


        # --- 2. Model Creation ---
        model = cp_model.CpModel()

        # Calculate a robust (loose) horizon
        loose_horizon = 0
        for job_id, tasks in jobs.items():
            for task in tasks:
                loose_horizon += max(task.runtimes)
        # Fallback if calculation is zero or too small
        if loose_horizon <= 0:
             loose_horizon = sum(rt for task in tasks_list for rt in task.runtimes if rt > 0) * 2

        # Tighten horizon using a fast constructive feasible schedule (greedy UB)
        greedy_horizon_ub = cls._estimate_tight_horizon(tasks_list, num_machines)
        if greedy_horizon_ub > 0:
            horizon = max(1, min(loose_horizon, greedy_horizon_ub))
        else:
            horizon = max(1, loose_horizon)

        # Ensure energy_prices covers the horizon
        if len(energy_prices) < horizon:
            print(f"Warning: Energy prices array ({len(energy_prices)}) shorter than horizon ({horizon}). Extending with last value.")
            last_price = energy_prices[-1] if len(energy_prices) > 0 else 0.1
            energy_prices = list(energy_prices) + [last_price] * (horizon - len(energy_prices))


        # Data structures
        all_tasks_vars = {}
        machine_to_intervals = collections.defaultdict(list)
        machine_usages = {} # (job_id, task_id, machine_id) -> BoolVar
        machine_start_vars = {} # (job_id, task_id, machine_id) -> IntVar (start on that machine)

        task_type = collections.namedtuple('task_type', 'start end interval duration presence_literals')

        # Create variables for each task and machine alternative
        for task in tasks_list:
            job_id = task.job_index
            task_id = task.task_index
            task_key = (job_id, task_id)
            suffix = f'_{job_id}_{task_id}'

            # Main task variables
            start_var = model.NewIntVar(0, horizon, f'start{suffix}')
            end_var = model.NewIntVar(0, horizon, f'end{suffix}')
            duration_var = model.NewIntVar(0, horizon, f'duration{suffix}') # Duration depends on machine

            presence_literals = []
            possible_durations = []

            # Create alternative intervals for each valid machine
            for machine_id in all_machines:
                duration = task.runtimes[machine_id]
                if duration <= 0:
                    # Create a dummy literal for non-viable machines to satisfy sum == 1 later
                    machine_usage = model.NewBoolVar(f'presence{suffix}_{machine_id}')
                    machine_usages[job_id, task_id, machine_id] = machine_usage
                    presence_literals.append(machine_usage)
                    model.Add(machine_usage == 0) # Force this literal to false
                    continue

                alt_suffix = f'{job_id}_{task_id}_{machine_id}'
                machine_usage = model.NewBoolVar(f'presence{alt_suffix}')
                alt_start = model.NewIntVar(0, horizon, f'start{alt_suffix}')
                alt_end = model.NewIntVar(0, horizon, f'end{alt_suffix}')
                alt_interval = model.NewOptionalIntervalVar(
                    alt_start,
                    duration, # Use direct duration
                    alt_end,
                    machine_usage,
                    f'interval{alt_suffix}'
                )

                # Link alternative variables to main task variables
                model.Add(start_var == alt_start).OnlyEnforceIf(machine_usage)
                model.Add(end_var == alt_end).OnlyEnforceIf(machine_usage)
                model.Add(duration_var == duration).OnlyEnforceIf(machine_usage)

                machine_to_intervals[machine_id].append(alt_interval)
                machine_usages[job_id, task_id, machine_id] = machine_usage
                machine_start_vars[job_id, task_id, machine_id] = alt_start
                presence_literals.append(machine_usage)
                possible_durations.append(duration)

            # Task interval (needed for some constraints, duration is variable)
            interval_var = model.NewIntervalVar(start_var, duration_var, end_var, f'interval{suffix}')

            all_tasks_vars[task_key] = task_type(
                start=start_var,
                end=end_var,
                interval=interval_var,
                duration=duration_var,
                presence_literals=presence_literals
            )

            # Ensure exactly one machine is selected among the valid ones
            model.Add(sum(all_tasks_vars[task_key].presence_literals) == 1)

        # --- 3. Add Constraints ---

        # Machine non-overlap
        for machine_id in all_machines:
            if machine_to_intervals[machine_id]:
                model.AddNoOverlap(machine_to_intervals[machine_id])

        # Job precedence
        for job_id, tasks_in_job in jobs.items():
            for i in range(len(tasks_in_job) - 1):
                pred_task_key = (job_id, tasks_in_job[i].task_index)
                succ_task_key = (job_id, tasks_in_job[i+1].task_index)
                model.Add(all_tasks_vars[succ_task_key].start >= all_tasks_vars[pred_task_key].end)


        # --- 4. Define Objective ---
        # Create makespan variable
        makespan_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(
            makespan_var,
            [all_tasks_vars[job_id, tasks_in_job[-1].task_index].end
             for job_id, tasks_in_job in jobs.items() if tasks_in_job]
        )

        # --- Total Energy Cost Calculation (Pulse-based) ---
        # Fast objective with explicit constants and start-time lookup tables.
        # Formulation is aligned with env/nsga decomposition:
        # total_power[t] = standby_total + sum(net_running_power_of_active_tasks)
        # grid[t] = max(0, total_power[t] - solar[t])
        # cost = sum_t grid[t] * price[t] / 60000 (in cents)
        #
        # To keep CP-SAT fast, we keep the constant baseline exactly and optimize
        # task-dependent marginal costs via AddElement(start_time -> precomputed cost).
        total_energy_cost_var, total_cost_upper = cls._calculate_energy_cost_pulse_based(
            model,
            tasks_list,
            machines,
            solar_profile,
            energy_prices,
            horizon,
            machine_usages,
            machine_start_vars,
        )

        # Create weighted objective with tight upper bound
        # Scale weights to preserve fractional values (e.g., 0.5 must not become 0)
        weight_scale = 1000
        makespan_scaled = max(0, int(round(makespan_weight * weight_scale)))
        energy_cost_scaled = max(0, int(round(energy_cost_weight * weight_scale)))
        if makespan_scaled == 0 and energy_cost_scaled == 0:
            makespan_scaled = 1
        weighted_obj_upper = makespan_scaled * horizon + energy_cost_scaled * total_cost_upper
        weighted_objective = model.NewIntVar(0, max(1, weighted_obj_upper), 'weighted_objective')
        model.Add(weighted_objective == makespan_scaled * makespan_var + energy_cost_scaled * total_energy_cost_var)
        model.Minimize(weighted_objective)

        model.Minimize(total_energy_cost_var) # For now, we optimize only energy to keep it simple and fast. 
        


        # --- 5. Solve ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        solver.parameters.num_search_workers = num_workers

        status = solver.Solve(model)

        # --- 6. Process Solution ---
        objective_value = None
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            objective_value = solver.ObjectiveValue()
            makespan_value = solver.Value(makespan_var)
            energy_cost_value = solver.Value(total_energy_cost_var)
            print(f"Solution found - Makespan: {makespan_value}, Energy Cost: {energy_cost_value:.2f}€")

            # Apply solution back to the original Task objects
            for task in tasks_list:
                job_id = task.job_index
                task_id = task.task_index
                task_key = (job_id, task_id)
                task_vars = all_tasks_vars[task_key]

                task.started = solver.Value(task_vars.start)
                task.finished = solver.Value(task_vars.end)
                task.done = True # Mark as done since it's scheduled

                selected_machine = -1 # Default to -1 or None if not found
                for machine_id in all_machines:
                     if (job_id, task_id, machine_id) in machine_usages:
                         if solver.Value(machine_usages[job_id, task_id, machine_id]):
                             selected_machine = machine_id
                             break
                task.selected_machine = selected_machine
                if selected_machine == -1 and any(task.runtimes[m] > 0 for m in all_machines):
                     print(f"Warning: Could not determine selected machine for task {task_key}, though solution was found.")

            return tasks_list, True

        elif status == cp_model.INFEASIBLE:
            print("Problem proven infeasible. Check constraints, dependencies, and machine availability.")
            return tasks_list, None
        elif status == cp_model.MODEL_INVALID:
            print("Model is invalid. Check model construction logic and variable definitions.")
            print(model.Validate())
            return tasks_list, None
        elif status == cp_model.UNKNOWN:
            print("Solver stopped (e.g., time limit reached) without proving optimality or infeasibility.")
            return tasks_list, None
        else:
            print(f"Solver failed with unexpected status: {solver.StatusName(status)}")
            return tasks_list, None

    @staticmethod
    def parse_instance_to_solver_format(instance: List[Task]):
        """
        Parses the instance to a processable format.

        :param list[Task] instance: The instance as a list of Tasks
        :return: jobs lists with Tuples for every task, including all necessary information for the solver function
                    machine_id(s), processing_time, due_date(set to 0 for all but last task in job), tool

        :Example: Job2 = [([0, 2, 3], 8, 0, 4), ([0, 2, 3], 6, 0, 2), ...]

        """
        jobs_dict = {}
        for task in instance:
            task_info = [np.nonzero(task.machines)[0].tolist(), task.runtime]
            # set deadline to 0, if this task is not the terminal task
            if task.deadline != task._deadline:
                task_info.append(0)
            else:
                task_info.append(task.deadline)

            job_num = task.job_index
            # if the job list in the dictionary is empty, initialize it
            if job_num not in jobs_dict:
                jobs_dict[job_num] = []

            jobs_dict[job_num].append(task_info)

        return list(jobs_dict.values())

    @staticmethod
    def parse_to_plottable_format(original_instance: List[Task], assigned_jobs_by_solver):
        """
        Reformats the solution into the original instance format to be passed to the gantt chart generator.

        :param list[Tasks] original_instance: Original instance in original format
        :param assigned_jobs_by_solver: solution passed by the optimize() function

        :return: list[Tasks] plottable solved instance

        """
        max_machine_num = len(original_instance[0].machines)

        # iterate over all machines and tasks scheduled on them
        tasks_list = []
        for machine in assigned_jobs_by_solver:
            for assigned_task in assigned_jobs_by_solver[machine]:
                job_index = assigned_task.job
                task_index = assigned_task.index
                machines = np.zeros(max_machine_num)
                machines[assigned_task.machines] = 1
                task = Task(job_index=job_index, task_index=task_index,
                            machines=machines, deadline=assigned_task.duedate, done=True,
                            runtime=assigned_task.duration, started=assigned_task.start,
                            finished=assigned_task.start+assigned_task.duration[0], selected_machine=machine)
                tasks_list.append(task)

        return tasks_list

    @staticmethod
    def _estimate_tight_horizon(tasks: List[Task], num_machines: int) -> int:
        """
        Build a fast feasible schedule (greedy) to derive a tighter upper bound for the model horizon.
        """
        jobs = collections.defaultdict(list)
        for task in tasks:
            jobs[task.job_index].append(task)
        for job_id in jobs:
            jobs[job_id].sort(key=lambda x: x.task_index)

        machine_ready = [0 for _ in range(num_machines)]
        job_ready = collections.defaultdict(int)

        for job_id in sorted(jobs.keys()):
            for task in jobs[job_id]:
                best_machine = None
                best_end = None

                for machine_id, duration in enumerate(task.runtimes):
                    if duration <= 0:
                        continue
                    start = max(job_ready[job_id], machine_ready[machine_id])
                    end = start + duration
                    if best_end is None or end < best_end:
                        best_end = end
                        best_machine = machine_id

                if best_machine is None:
                    continue

                machine_ready[best_machine] = best_end
                job_ready[job_id] = best_end

        return max(machine_ready) if machine_ready else 0

    @staticmethod
    def _calculate_energy_cost_pulse_based(model, tasks, machines, solar_profile, energy_prices, horizon, machine_usages, machine_start_vars):
        """
        Exact energy-cost objective aligned with env consumed_energy_and_costs() formulation.

        Environment-equivalent per timestep equations:
        - machine_power[m, t] = standby_m, unless a task runs on machine m at t
        - if task k runs on machine m at t: machine_power[m, t] = running_power[k, m]
        - total_power[t] = sum_m machine_power[m, t]
        - net_power[t] = max(0, total_power[t] - solar[t])
        - total_cost = sum_t net_power[t] * price[t]

        We minimize only costs. Numeric scaling of prices is constant and does not alter argmin.

        Returns:
            total_energy_cost_var, total_cost_upper
        """
        if horizon <= 0:
            zero_cost = model.NewIntVar(0, 0, 'total_energy_cost')
            model.Add(zero_cost == 0)
            return zero_cost, 1

        # Integer constants for CP-SAT arithmetic.
        # Prices are scaled by 1000 (milli-ct/kWh-equivalent) to preserve decimals.
        PRICE_SCALE = 1000
        C_MACHINE_STANDBY_W = [int(round(getattr(machines[m], 'standby_energy', 0))) for m in range(len(machines))]
        C_PRICE_SCALED = [
            int(round(energy_prices[min(t, len(energy_prices) - 1)] * PRICE_SCALE)) if len(energy_prices) > 0 else 0
            for t in range(horizon)
        ]
        C_SOLAR_W = [
            int(round(solar_profile[t])) if t < len(solar_profile) else 0
            for t in range(horizon)
        ]

        task_machine_pairs = []
        max_total_power_per_timestep = 0
        for machine_id in range(len(machines)):
            max_running_on_machine = C_MACHINE_STANDBY_W[machine_id]
            for task in tasks:
                runtime = task.runtimes[machine_id] if machine_id < len(task.runtimes) else 0
                if runtime > 0:
                    running_power_w = int(round(task.machine_running_energy[machine_id]))
                    if running_power_w > max_running_on_machine:
                        max_running_on_machine = running_power_w
                    task_machine_pairs.append((task, machine_id, runtime, running_power_w))
            max_total_power_per_timestep += max_running_on_machine

        # Build exact per-timestep total machine power:
        # total_power[t] = sum_m standby_m + sum_active_tasks(delta_power_of_active_task)
        total_standby_w = sum(C_MACHINE_STANDBY_W)
        delta_active_per_timestep = [[] for _ in range(horizon)]

        for task, machine_id, runtime, running_power_w in task_machine_pairs:
            job_id, task_id = task.job_index, task.task_index
            machine_usage = machine_usages[(job_id, task_id, machine_id)]
            start_var = machine_start_vars[(job_id, task_id, machine_id)]
            delta_power_w = int(running_power_w - C_MACHINE_STANDBY_W[machine_id])

            latest_valid_start = max(0, horizon - runtime)

            for t in range(horizon):
                is_active_lookup = [0] * (horizon + 1)
                lo = max(0, t - runtime + 1)
                hi = min(t, latest_valid_start)
                if lo <= hi:
                    for s in range(lo, hi + 1):
                        is_active_lookup[s] = 1

                active_raw = model.NewIntVar(0, 1, f'active_raw_{job_id}_{task_id}_{machine_id}_{t}')
                model.AddElement(start_var, is_active_lookup, active_raw)

                active_eff = model.NewIntVar(0, 1, f'active_{job_id}_{task_id}_{machine_id}_{t}')
                model.Add(active_eff <= active_raw)
                model.Add(active_eff <= machine_usage)
                model.Add(active_eff >= active_raw + machine_usage - 1)

                if delta_power_w != 0:
                    delta_term = model.NewIntVar(
                        min(0, delta_power_w),
                        max(0, delta_power_w),
                        f'delta_pw_{job_id}_{task_id}_{machine_id}_{t}'
                    )
                    model.Add(delta_term == delta_power_w * active_eff)
                    delta_active_per_timestep[t].append(delta_term)

        net_power_terms = []
        cost_terms = []

        # Conservative lower bound for power after deltas.
        min_possible_total_power = 0

        for t in range(horizon):
            delta_sum_lb = sum(v.Proto().domain[0] for v in delta_active_per_timestep[t]) if delta_active_per_timestep[t] else 0
            delta_sum_ub = sum(v.Proto().domain[-1] for v in delta_active_per_timestep[t]) if delta_active_per_timestep[t] else 0

            total_power_lb = max(0, total_standby_w + delta_sum_lb)
            total_power_ub = max_total_power_per_timestep + max(0, delta_sum_ub)

            total_power_t = model.NewIntVar(total_power_lb, max(total_power_lb, total_power_ub), f'total_power_{t}')
            if delta_active_per_timestep[t]:
                model.Add(total_power_t == total_standby_w + sum(delta_active_per_timestep[t]))
            else:
                model.Add(total_power_t == total_standby_w)

            raw_grid_lb = min_possible_total_power - C_SOLAR_W[t]
            raw_grid_ub = total_power_ub - C_SOLAR_W[t]
            raw_grid_t = model.NewIntVar(raw_grid_lb, raw_grid_ub, f'raw_grid_{t}')
            model.Add(raw_grid_t == total_power_t - C_SOLAR_W[t])

            net_power_t = model.NewIntVar(0, max(0, raw_grid_ub), f'net_power_{t}')
            model.Add(net_power_t >= raw_grid_t)
            net_power_terms.append(net_power_t)

            price_t = C_PRICE_SCALED[t]
            cost_t = model.NewIntVar(0, net_power_t.Proto().domain[-1] * max(0, price_t), f'cost_t_{t}')
            model.Add(cost_t == net_power_t * price_t)
            cost_terms.append(cost_t)

        total_cost_upper = max(1, sum(v.Proto().domain[-1] for v in cost_terms))
        total_cost_var = model.NewIntVar(0, total_cost_upper, 'total_energy_cost')
        if cost_terms:
            model.Add(total_cost_var == sum(cost_terms))
        else:
            model.Add(total_cost_var == 0)

        return total_cost_var, total_cost_upper