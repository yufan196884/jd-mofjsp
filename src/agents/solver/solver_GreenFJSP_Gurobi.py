"""
Green FJSP Solver using Gurobi MIP Optimizer

This solver handles:
- Flexible Job Shop Scheduling Problems (FJSP)
- Energy-cost optimization with solar integration
- Multi-objective optimization (makespan + energy cost)

Advantages over CP-SAT for energy problems:
- Native support for continuous variables (power, energy)
- Better handling of time-dependent costs (electricity prices)
- Faster for large horizons with complex cost functions
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import collections
from typing import List, Tuple, Optional

from src.data_generator.task import Task


class GurobiSolver_GreenFJSP:
    """
    Gurobi-based solver for Green Flexible Job Shop Scheduling Problems.
    
    Optimizes scheduling decisions to minimize energy costs while respecting:
    - Job precedence constraints
    - Machine capacity constraints
    - Flexible machine assignments
    """

    @classmethod
    def optimize(cls, 
                 instance: dict, 
                 objective_weights: List[float] = [0.0, 1.0],  # [makespan, energy_cost]
                 time_limit_seconds: float = 10.0,
                 mip_gap: float = 0.01,  # 1% optimality gap
                 verbose: bool = True) -> Tuple[List[Task], bool]:
        """
        Optimize the scheduling instance using Gurobi MIP solver.
        
        Args:
            instance: Dict with keys 'tasks', 'machines', 'solar_profile', 'energy_prices'
            objective_weights: [makespan_weight, energy_cost_weight]
            time_limit_seconds: Maximum solve time
            mip_gap: MIP optimality gap (0.01 = 1%)
            verbose: Print solver progress
            
        Returns:
            (tasks_list, success): Modified task list and success flag
        """
        tasks_list = instance["tasks"]
        machines = instance["machines"]
        solar_profile = instance["solar_profile"]
        energy_prices = instance["energy_prices"]
        
        if not tasks_list:
            print("Error: Instance is empty.")
            return tasks_list, False

        # Normalize weights
        makespan_weight = objective_weights[0]
        energy_cost_weight = objective_weights[1]

        # Extract problem dimensions
        num_machines = len(machines)
        num_tasks = len(tasks_list)
        
        # Group tasks by job
        jobs = collections.defaultdict(list)
        for task in tasks_list:
            jobs[task.job_index].append(task)
        for job_id in jobs:
            jobs[job_id].sort(key=lambda x: x.task_index)
        
        # Calculate horizon (greedy upper bound)
        horizon = cls._estimate_horizon(tasks_list, num_machines)
        if horizon <= 0:
            horizon = sum(max(t.runtimes) for t in tasks_list)
        
        # Extend energy profiles to horizon
        if len(energy_prices) < horizon:
            last_price = energy_prices[-1] if len(energy_prices) > 0 else 0.1
            energy_prices = list(energy_prices) + [last_price] * (horizon - len(energy_prices))
        if len(solar_profile) < horizon:
            last_solar = solar_profile[-1] if len(solar_profile) > 0 else 0
            solar_profile = list(solar_profile) + [last_solar] * (horizon - len(solar_profile))
        
        # Extract machine standby powers
        machine_standby = [float(getattr(m, 'standby_energy', 0)) for m in machines]
        
        if verbose:
            print(f"Building Gurobi model: {len(tasks_list)} tasks, {num_machines} machines, horizon={horizon}")
        
        # Create Gurobi model
        model = gp.Model("GreenFJSP")
        model.Params.TimeLimit = time_limit_seconds
        model.Params.MIPGap = mip_gap
        model.Params.OutputFlag = 1 if verbose else 0
        
        # ===== Decision Variables =====
        
        # start[j,t]: start time of task t in job j
        start = {}
        for task in tasks_list:
            j, t = task.job_index, task.task_index
            start[j, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=horizon, name=f"start_{j}_{t}")
        
        # machine[j,t,m]: binary, 1 if task (j,t) runs on machine m
        machine = {}
        for task in tasks_list:
            j, t = task.job_index, task.task_index
            for m in range(num_machines):
                if task.runtimes[m] > 0:
                    machine[j, t, m] = model.addVar(vtype=GRB.BINARY, name=f"machine_{j}_{t}_{m}")
        
        # makespan variable
        makespan = model.addVar(vtype=GRB.INTEGER, lb=0, ub=horizon, name="makespan")
        
        # Power and energy variables (per timestep)
        machine_power = {}  # machine_power[m, tau]: power of machine m at time tau
        total_power = {}    # total_power[tau]: total power at time tau
        net_power = {}      # net_power[tau]: grid power after solar offset
        
        for tau in range(horizon):
            for m in range(num_machines):
                machine_power[m, tau] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"mach_pow_{m}_{tau}")
            total_power[tau] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"total_pow_{tau}")
            net_power[tau] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"net_pow_{tau}")
        
        model.update()
        
        # ===== Constraints =====
        
        # 1. Each task must run on exactly one machine
        for task in tasks_list:
            j, t = task.job_index, task.task_index
            valid_machines = [m for m in range(num_machines) if task.runtimes[m] > 0]
            if valid_machines:
                model.addConstr(gp.quicksum(machine[j, t, m] for m in valid_machines) == 1, 
                               name=f"one_machine_{j}_{t}")
        
        # 2. Job precedence constraints
        for job_id, tasks_in_job in jobs.items():
            for i in range(len(tasks_in_job) - 1):
                curr_task = tasks_in_job[i]
                next_task = tasks_in_job[i + 1]
                j_curr, t_curr = curr_task.job_index, curr_task.task_index
                j_next, t_next = next_task.job_index, next_task.task_index
                
                # end(curr) <= start(next)
                # end(curr) = start(curr) + sum(duration[m] * machine[j,t,m])
                end_curr = start[j_curr, t_curr] + gp.quicksum(
                    curr_task.runtimes[m] * machine[j_curr, t_curr, m]
                    for m in range(num_machines) if curr_task.runtimes[m] > 0
                )
                model.addConstr(end_curr <= start[j_next, t_next], 
                               name=f"precedence_{j_curr}_{t_curr}_{t_next}")
        
        # 3. Machine non-overlap constraints (no two tasks on same machine overlap in time)
        # For each machine, for each pair of tasks that could run on it
        task_machine_pairs = []
        for task in tasks_list:
            j, t = task.job_index, task.task_index
            for m in range(num_machines):
                if task.runtimes[m] > 0:
                    task_machine_pairs.append((j, t, m, task.runtimes[m]))
        
        for m in range(num_machines):
            tasks_on_m = [(j, t, dur) for j, t, mm, dur in task_machine_pairs if mm == m]
            for idx1 in range(len(tasks_on_m)):
                for idx2 in range(idx1 + 1, len(tasks_on_m)):
                    j1, t1, dur1 = tasks_on_m[idx1]
                    j2, t2, dur2 = tasks_on_m[idx2]
                    
                    # Either task1 ends before task2 starts, or task2 ends before task1 starts
                    # Use big-M formulation with binary indicator
                    M = horizon * 2
                    y = model.addVar(vtype=GRB.BINARY, name=f"order_{j1}_{t1}_{j2}_{t2}_{m}")
                    
                    # If y=1: task1 before task2
                    # If machine[j1,t1,m]=1 and machine[j2,t2,m]=1 and y=1, then start[j1,t1]+dur1 <= start[j2,t2]
                    model.addConstr(
                        start[j1, t1] + dur1 <= start[j2, t2] + M * (1 - y) + M * (2 - machine[j1, t1, m] - machine[j2, t2, m]),
                        name=f"no_overlap_1_{j1}_{t1}_{j2}_{t2}_{m}"
                    )
                    
                    # If y=0: task2 before task1
                    model.addConstr(
                        start[j2, t2] + dur2 <= start[j1, t1] + M * y + M * (2 - machine[j1, t1, m] - machine[j2, t2, m]),
                        name=f"no_overlap_2_{j1}_{t1}_{j2}_{t2}_{m}"
                    )
        
        # 4. Makespan definition
        for job_id, tasks_in_job in jobs.items():
            if tasks_in_job:
                last_task = tasks_in_job[-1]
                j, t = last_task.job_index, last_task.task_index
                end_last = start[j, t] + gp.quicksum(
                    last_task.runtimes[m] * machine[j, t, m]
                    for m in range(num_machines) if last_task.runtimes[m] > 0
                )
                model.addConstr(makespan >= end_last, name=f"makespan_job_{job_id}")
        
        # 5. Energy constraints (exact env formulation)
        # For each timestep tau:
        #   - Each machine power = standby OR running_power (if task active)
        #   - total_power = sum of machine powers
        #   - net_power = max(0, total_power - solar)
        
        # Create binary indicators for task activity at each timestep
        task_active = {}  # task_active[j,t,m,tau]: 1 if task (j,t) runs on machine m at time tau
        
        for j, t, m, dur in task_machine_pairs:
            for tau in range(horizon):
                task_active[j, t, m, tau] = model.addVar(vtype=GRB.BINARY, 
                                                         name=f"active_{j}_{t}_{m}_{tau}")
                
                # Link to start time and machine selection
                # active if: machine selected AND tau in [start, start+dur)
                # active = 1 <=> machine[j,t,m]=1 AND start[j,t] <= tau < start[j,t]+dur
                
                # Linearize: active <= machine[j,t,m]
                model.addConstr(task_active[j, t, m, tau] <= machine[j, t, m])
                
                # Linearize: active = 1 => start <= tau and tau < start+dur
                # tau >= start - M*(1-active)
                # tau < start+dur + M*(1-active)
                M = horizon
                model.addConstr(tau >= start[j, t] - M * (1 - task_active[j, t, m, tau]))
                model.addConstr(tau <= start[j, t] + dur - 1 + M * (1 - task_active[j, t, m, tau]))
                
                # If machine selected and tau in range, then active must be 1
                # This is enforced by: active >= machine + (indicator for time window) - 1
                # Use auxiliary binary for time window
                in_window = model.addVar(vtype=GRB.BINARY, name=f"window_{j}_{t}_{m}_{tau}")
                model.addConstr(tau >= start[j, t] - M * (1 - in_window))
                model.addConstr(tau <= start[j, t] + dur - 1 + M * in_window)
                model.addConstr(task_active[j, t, m, tau] >= machine[j, t, m] + in_window - 1)
        
        # Machine power at each timestep
        for tau in range(horizon):
            for m in range(num_machines):
                # machine_power[m,tau] = standby + sum over tasks of (running-standby)*active
                running_contributions = []
                for task in tasks_list:
                    j, t = task.job_index, task.task_index
                    if task.runtimes[m] > 0:
                        running_pow = float(task.machine_running_energy[m])
                        delta_pow = running_pow - machine_standby[m]
                        running_contributions.append(delta_pow * task_active[j, t, m, tau])
                
                if running_contributions:
                    model.addConstr(
                        machine_power[m, tau] == machine_standby[m] + gp.quicksum(running_contributions),
                        name=f"machine_power_{m}_{tau}"
                    )
                else:
                    model.addConstr(machine_power[m, tau] == machine_standby[m])
        
        # Total power per timestep
        for tau in range(horizon):
            model.addConstr(
                total_power[tau] == gp.quicksum(machine_power[m, tau] for m in range(num_machines)),
                name=f"total_power_{tau}"
            )
        
        # Net power (grid power after solar offset)
        for tau in range(horizon):
            solar_tau = float(solar_profile[tau])
            # net_power[tau] = max(0, total_power[tau] - solar_tau)
            model.addConstr(net_power[tau] >= total_power[tau] - solar_tau, name=f"net_power_{tau}")
            model.addConstr(net_power[tau] >= 0)
        
        # ===== Objective Function =====
        
        # Energy cost (convert to €): sum_tau net_power[tau] * price[tau] / 100 (convert W*min + ct/kWh to €)
        # Formula from env: energy_per_timestep_kwh = net_energy / 100.0
        #                   cost = energy_kwh * price (price in ct/kWh), then / 100 to convert to €
        energy_cost_terms = []
        for tau in range(horizon):
            price_tau = float(energy_prices[tau])  # ct/kWh
            # W * ct/kWh need conversion: W-minutes to kWh = W / 60 / 1000, then * price, then /100 for ct to €
            # Final: W * price / 60000 (€ per minute per W)
            # But price is in ct/kWh, so: W * (ct/kWh) / (60 min/h * 1000 W/kW) * (1€ / 100ct)
            # = W * price / 60 / 1000 / 100 = W * price / 6,000,000
            # Actually from env code: energy_kwh = W / 100.0, then cost = energy_kwh * price, then / 100
            # So: (W / 100) * price / 100 = W * price / 10000
            # Wait, let me check env code again...
            # env: energy_per_timestep_kwh = net_energy_per_timestep / 100.0
            #      costs_per_timestep = energy_per_timestep_kwh * energy_prices  # energy_prices in ct/kWh
            #      cost_per_timestep_euro = costs_per_timestep / 100.0
            # So: cost_euro = (W / 100) * (ct/kWh) / 100 = W * ct/kWh / 10000
            cost_contribution = net_power[tau] * price_tau / 10000.0
            energy_cost_terms.append(cost_contribution)
        
        total_energy_cost = gp.quicksum(energy_cost_terms)
        
        # Multi-objective
        if makespan_weight > 0 and energy_cost_weight > 0:
            # Normalize objectives to similar scales
            makespan_scaled = makespan / horizon  # 0-1 range
            energy_scaled = total_energy_cost / (horizon * sum(machine_standby) * max(energy_prices) / 10000.0)
            objective = makespan_weight * makespan_scaled + energy_cost_weight * energy_scaled
        elif energy_cost_weight > 0:
            objective = total_energy_cost
        else:
            objective = makespan
        
        model.setObjective(objective, GRB.MINIMIZE)
        
        # ===== Solve =====
        if verbose:
            print("Solving model...")
        
        model.optimize()
        
        # ===== Extract Solution =====
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT and model.SolCount > 0:
            if verbose:
                makespan_val = makespan.X
                energy_val = total_energy_cost.getValue()
                print(f"Solution found - Makespan: {makespan_val:.0f}, Energy Cost: {energy_val:.4f}€")
                if model.status == GRB.TIME_LIMIT:
                    print(f"Time limit reached. MIP gap: {model.MIPGap * 100:.2f}%")
            
            # Apply solution to tasks
            for task in tasks_list:
                j, t = task.job_index, task.task_index
                task.started = int(round(start[j, t].X))
                
                # Find selected machine
                selected_machine = -1
                for m in range(num_machines):
                    if task.runtimes[m] > 0 and (j, t, m) in machine:
                        if machine[j, t, m].X > 0.5:
                            selected_machine = m
                            task.finished = task.started + task.runtimes[m]
                            break
                
                task.selected_machine = selected_machine
                task.done = True
            
            return tasks_list, True
        
        else:
            if verbose:
                print(f"No solution found. Status: {model.status}")
            return tasks_list, False

    @staticmethod
    def _estimate_horizon(tasks: List[Task], num_machines: int) -> int:
        """Greedy upper bound on makespan"""
        jobs = collections.defaultdict(list)
        for task in tasks:
            jobs[task.job_index].append(task)
        for job_id in jobs:
            jobs[job_id].sort(key=lambda x: x.task_index)
        
        machine_ready = [0] * num_machines
        job_ready = collections.defaultdict(int)
        
        for job_id in sorted(jobs.keys()):
            for task in jobs[job_id]:
                best_end = None
                for m, dur in enumerate(task.runtimes):
                    if dur > 0:
                        start = max(job_ready[job_id], machine_ready[m])
                        end = start + dur
                        if best_end is None or end < best_end:
                            best_end = end
                            best_machine = m
                
                if best_end is not None:
                    machine_ready[best_machine] = best_end
                    job_ready[job_id] = best_end
        
        return max(machine_ready) if machine_ready else 0
