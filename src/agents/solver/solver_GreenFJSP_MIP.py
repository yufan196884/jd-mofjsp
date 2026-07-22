"""
SCIP-based MIP Solver for Green Flexible Job Shop Scheduling Problem (GreenFJSP)

Uses PySCIPOpt - an open-source MIP solver (no license required)
Optimizes: Makespan and/or Energy Cost with solar profile

Installation:
    pip install pyscipopt
"""

from pyscipopt import Model, quicksum
from typing import List, Dict, Tuple
import collections
import numpy as np

from src.data_generator.task import Task


class SCIPSolver_GreenFJSP:
    """
    SCIP-based MIP solver for GreenFJSP with exact energy cost formulation.
    
    Features:
    - Multi-objective optimization (makespan + energy cost)
    - Solar profile integration
    - Time-varying energy prices
    - No commercial license required (fully open-source)
    """

    @classmethod
    def optimize(cls, 
                 instance: Dict,
                 objective_weights: List[float] = [0.5, 0.5],
                 time_limit_seconds: float = 10.0,
                 mip_gap: float = 0.01,
                 num_threads: int = 0) -> Tuple[List[Task], bool]:
        """
        Optimize GreenFJSP instance using SCIP MIP solver.
        
        Args:
            instance: Dict with keys:
                - 'tasks': List[Task]
                - 'machines': List of machine objects with standby_energy
                - 'solar_profile': List[float] - solar power at each timestep (Watts)
                - 'energy_prices': List[float] - €/kWh at each timestep
            objective_weights: [makespan_weight, energy_cost_weight]
            time_limit_seconds: Maximum solve time
            mip_gap: Relative optimality gap (0.01 = 1%)
            num_threads: Number of parallel threads (0 = auto)
            
        Returns:
            (tasks_list, success): Modified tasks with solution, and success flag
        """
        
        tasks_list = instance["tasks"]
        machines = instance["machines"]
        solar_profile = instance["solar_profile"]
        energy_prices = instance["energy_prices"]
        
        if not tasks_list:
            print("Error: Instance is empty.")
            return tasks_list, False
        
        # Extract weights
        makespan_weight = objective_weights[0]
        energy_cost_weight = objective_weights[1]
        
        # Group tasks by job
        jobs = collections.defaultdict(list)
        for task in tasks_list:
            jobs[task.job_index].append(task)
        for job_id in jobs:
            jobs[job_id].sort(key=lambda x: x.task_index)
        
        num_machines = len(machines)
        
        # Estimate tight horizon using greedy heuristic
        horizon = cls._estimate_tight_horizon(tasks_list, num_machines)
        if horizon <= 0:
            horizon = sum(max(t.runtimes) for t in tasks_list)
        
        # Extend energy prices if needed
        if len(energy_prices) < horizon:
            last_price = energy_prices[-1] if energy_prices else 0.1
            energy_prices = list(energy_prices) + [last_price] * (horizon - len(energy_prices))
        
        # Extend solar profile if needed
        if len(solar_profile) < horizon:
            solar_profile = list(solar_profile) + [0] * (horizon - len(solar_profile))
        
        print(f"Starting SCIP optimization - Horizon: {horizon}, Tasks: {len(tasks_list)}, Machines: {num_machines}")
        
        # Create SCIP model
        model = Model("GreenFJSP")
        model.hideOutput()  # Suppress verbose output
        
        # Set parameters
        model.setRealParam('limits/time', time_limit_seconds)
        model.setRealParam('limits/gap', mip_gap)
        if num_threads > 0:
            model.setIntParam('parallel/maxnthreads', num_threads)
        
        # ===== DECISION VARIABLES =====
        
        # Start time for each task
        start = {}
        for task in tasks_list:
            job_id, task_id = task.job_index, task.task_index
            start[job_id, task_id] = model.addVar(
                vtype="INTEGER",
                lb=0,
                ub=horizon,
                name=f"start_{job_id}_{task_id}"
            )
        
        # Machine assignment: x[job, task, machine] = 1 if task assigned to machine
        x = {}
        for task in tasks_list:
            job_id, task_id = task.job_index, task.task_index
            for m in range(num_machines):
                if task.runtimes[m] > 0:  # Only create variables for valid machines
                    x[job_id, task_id, m] = model.addVar(
                        vtype="BINARY",
                        name=f"x_{job_id}_{task_id}_{m}"
                    )

        # Start-time binaries: z[job, task, machine, s] = 1 iff task starts at time s on machine
        z = {}
        for task in tasks_list:
            job_id, task_id = task.job_index, task.task_index
            for m in range(num_machines):
                runtime = int(task.runtimes[m])
                if runtime <= 0:
                    continue
                latest_start = max(0, horizon - runtime)
                for s in range(latest_start + 1):
                    z[job_id, task_id, m, s] = model.addVar(
                        vtype="BINARY",
                        name=f"z_{job_id}_{task_id}_{m}_{s}"
                    )
        
        # Pairwise order variables for machine disjunction constraints
        # order[(j1,t1,j2,t2,m)] = 1 => (j1,t1) before (j2,t2) on machine m
        order = {}
        for m in range(num_machines):
            compatible_tasks = [
                (task.job_index, task.task_index, int(task.runtimes[m]))
                for task in tasks_list if task.runtimes[m] > 0
            ]
            for idx1 in range(len(compatible_tasks)):
                for idx2 in range(idx1 + 1, len(compatible_tasks)):
                    j1, t1, _ = compatible_tasks[idx1]
                    j2, t2, _ = compatible_tasks[idx2]
                    order[j1, t1, j2, t2, m] = model.addVar(
                        vtype="BINARY",
                        name=f"ord_{j1}_{t1}_{j2}_{t2}_{m}"
                    )
        
        # Makespan variable
        makespan = model.addVar(vtype="INTEGER", lb=0, ub=horizon, name="makespan")

        # Energy variables (exact env formulation)
        machine_power = {}
        for m in range(num_machines):
            standby_w = float(getattr(machines[m], 'standby_energy', 0))
            max_running_w = standby_w
            for task in tasks_list:
                if task.runtimes[m] > 0:
                    max_running_w = max(max_running_w, float(task.machine_running_energy[m]))
            for t in range(horizon):
                machine_power[m, t] = model.addVar(
                    vtype="CONTINUOUS",
                    lb=standby_w,
                    ub=max_running_w,
                    name=f"machine_power_{m}_{t}"
                )

        total_power = {}
        grid_power = {}
        max_total_power = 0.0
        for m in range(num_machines):
            standby_w = float(getattr(machines[m], 'standby_energy', 0))
            max_running_w = standby_w
            for task in tasks_list:
                if task.runtimes[m] > 0:
                    max_running_w = max(max_running_w, float(task.machine_running_energy[m]))
            max_total_power += max_running_w

        for t in range(horizon):
            total_power[t] = model.addVar(
                vtype="CONTINUOUS",
                lb=0.0,
                ub=max_total_power,
                name=f"total_power_{t}"
            )
            solar_t = float(solar_profile[t])
            grid_power[t] = model.addVar(
                vtype="CONTINUOUS",
                lb=0.0,
                ub=max(0.0, max_total_power - solar_t),
                name=f"grid_power_{t}"
            )

        total_energy_cost = model.addVar(vtype="CONTINUOUS", lb=0.0, name="total_energy_cost")
        
        # ===== CONSTRAINTS =====
        
        # 1. Each task assigned to exactly one machine
        for task in tasks_list:
            job_id, task_id = task.job_index, task.task_index
            valid_machines = [m for m in range(num_machines) if task.runtimes[m] > 0]
            if valid_machines:
                model.addCons(
                    quicksum(x[job_id, task_id, m] for m in valid_machines) == 1,
                    name=f"assign_{job_id}_{task_id}"
                )

                # Link start-time binaries to machine assignment
                for m in valid_machines:
                    runtime = int(task.runtimes[m])
                    latest_start = max(0, horizon - runtime)
                    model.addCons(
                        quicksum(z[job_id, task_id, m, s] for s in range(latest_start + 1)) == x[job_id, task_id, m],
                        name=f"z_link_{job_id}_{task_id}_{m}"
                    )

                # Link integer start variable to chosen start-time binary
                model.addCons(
                    start[job_id, task_id] == quicksum(
                        s * z[job_id, task_id, m, s]
                        for m in valid_machines
                        for s in range(max(0, horizon - int(task.runtimes[m])) + 1)
                    ),
                    name=f"start_link_{job_id}_{task_id}"
                )
        
        # 2. Job precedence constraints
        for job_id, tasks_in_job in jobs.items():
            for i in range(len(tasks_in_job) - 1):
                pred = tasks_in_job[i]
                succ = tasks_in_job[i + 1]
                
                # Successor starts after predecessor ends (considering all possible machines)
                for m_pred in range(num_machines):
                    if pred.runtimes[m_pred] > 0:
                        duration_pred = int(pred.runtimes[m_pred])
                        model.addCons(
                            start[succ.job_index, succ.task_index] >= 
                            start[pred.job_index, pred.task_index] + duration_pred - 
                            horizon * (1 - x[pred.job_index, pred.task_index, m_pred]),
                            name=f"prec_{job_id}_{i}_{m_pred}"
                        )
        
        # 3. Machine non-overlap (pairwise disjunctive constraints)
        M = horizon * 2
        for m in range(num_machines):
            compatible_tasks = [
                (task.job_index, task.task_index, int(task.runtimes[m]))
                for task in tasks_list if task.runtimes[m] > 0
            ]
            for idx1 in range(len(compatible_tasks)):
                for idx2 in range(idx1 + 1, len(compatible_tasks)):
                    j1, t1, d1 = compatible_tasks[idx1]
                    j2, t2, d2 = compatible_tasks[idx2]
                    ord_var = order[j1, t1, j2, t2, m]

                    # If both tasks are assigned to m and ord_var=1, task1 precedes task2
                    model.addCons(
                        start[j1, t1] + d1 <= start[j2, t2] + M * (1 - ord_var) + M * (2 - x[j1, t1, m] - x[j2, t2, m]),
                        name=f"no_overlap_1_{j1}_{t1}_{j2}_{t2}_{m}"
                    )

                    # If both tasks are assigned to m and ord_var=0, task2 precedes task1
                    model.addCons(
                        start[j2, t2] + d2 <= start[j1, t1] + M * ord_var + M * (2 - x[j1, t1, m] - x[j2, t2, m]),
                        name=f"no_overlap_2_{j1}_{t1}_{j2}_{t2}_{m}"
                    )
        
        # 4. Makespan constraints
        for job_id, tasks_in_job in jobs.items():
            if tasks_in_job:
                last_task = tasks_in_job[-1]
                for m in range(num_machines):
                    if last_task.runtimes[m] > 0:
                        duration = int(last_task.runtimes[m])
                        model.addCons(
                            makespan >= start[last_task.job_index, last_task.task_index] + 
                            duration * x[last_task.job_index, last_task.task_index, m],
                            name=f"makespan_{job_id}_{m}"
                        )

        # 5. Exact per-timestep machine power and grid power (aligned with env)
        # machine_power[m,t] = standby_m + sum_active_tasks (running_m - standby_m)
        for m in range(num_machines):
            standby_w = float(getattr(machines[m], 'standby_energy', 0))
            for t in range(horizon):
                delta_terms = []
                for task in tasks_list:
                    job_id, task_id = task.job_index, task.task_index
                    runtime = int(task.runtimes[m]) if m < len(task.runtimes) else 0
                    if runtime <= 0:
                        continue
                    running_w = float(task.machine_running_energy[m])
                    delta_w = running_w - standby_w
                    latest_start = max(0, horizon - runtime)
                    active_starts = [s for s in range(latest_start + 1) if s <= t < s + runtime]
                    if active_starts:
                        delta_terms.append(delta_w * quicksum(z[job_id, task_id, m, s] for s in active_starts))

                if delta_terms:
                    model.addCons(
                        machine_power[m, t] == standby_w + quicksum(delta_terms),
                        name=f"machine_power_def_{m}_{t}"
                    )
                else:
                    model.addCons(
                        machine_power[m, t] == standby_w,
                        name=f"machine_power_def_{m}_{t}"
                    )

        # total_power[t] and grid_power[t] = max(0, total_power[t] - solar[t]) via >= constraints
        for t in range(horizon):
            model.addCons(
                total_power[t] == quicksum(machine_power[m, t] for m in range(num_machines)),
                name=f"total_power_def_{t}"
            )
            solar_t = float(solar_profile[t])
            model.addCons(
                grid_power[t] >= total_power[t] - solar_t,
                name=f"grid_relu_lb_{t}"
            )
        
        # 6. Total energy cost (env-equivalent units)
        # env: energy_kwh = net_energy_per_timestep / 100.0
        #      costs_ct = energy_kwh * energy_prices(ct/kWh)
        #      costs_eur = costs_ct / 100.0
        # => costs_eur = net_energy * price / 10000.0
        model.addCons(
            total_energy_cost == quicksum(grid_power[t] * float(energy_prices[t]) / 10000.0 for t in range(horizon)),
            name="total_energy_cost_def"
        )

        # ===== OBJECTIVE =====
        # Weighted multi-objective
        objective = makespan_weight * makespan + energy_cost_weight * total_energy_cost
        model.setObjective(objective, "minimize")
        
        # ===== SOLVE =====
        
        print("Starting SCIP solver...")
        model.optimize()
        
        status = model.getStatus()
        print(f"SCIP solver finished with status: {status}")
        
        # ===== EXTRACT SOLUTION =====
        
        if status == "optimal" or status == "bestsollimit" or status == "timelimit":
            if model.getNSols() > 0:
                sol = model.getBestSol()
                
                makespan_value = model.getSolVal(sol, makespan)
                energy_cost_value = model.getSolVal(sol, total_energy_cost)
                
                print(f"Solution found - Makespan: {makespan_value:.0f}, Energy Cost: {energy_cost_value:.4f}€")
                
                # Apply solution to tasks
                for task in tasks_list:
                    job_id, task_id = task.job_index, task.task_index
                    
                    task.started = int(round(model.getSolVal(sol, start[job_id, task_id])))
                    task.done = True
                    
                    # Find selected machine
                    selected_machine = -1
                    for m in range(num_machines):
                        if task.runtimes[m] > 0:
                            if model.getSolVal(sol, x[job_id, task_id, m]) > 0.5:
                                selected_machine = m
                                task.finished = task.started + int(task.runtimes[m])
                                break
                    
                    task.selected_machine = selected_machine
                
                return tasks_list, True
            else:
                print("No solution found within time limit.")
                return tasks_list, False
        else:
            print(f"Solver failed with status: {status}")
            return tasks_list, False
    
    @staticmethod
    def _estimate_tight_horizon(tasks: List[Task], num_machines: int) -> int:
        """Greedy heuristic to estimate tight horizon."""
        jobs = collections.defaultdict(list)
        for task in tasks:
            jobs[task.job_index].append(task)
        for job_id in jobs:
            jobs[job_id].sort(key=lambda x: x.task_index)
        
        machine_ready = [0 for _ in range(num_machines)]
        job_ready = collections.defaultdict(int)
        
        for job_id in sorted(jobs.keys()):
            for task in jobs[job_id]:
                best_end = None
                best_machine = None
                
                for machine_id, duration in enumerate(task.runtimes):
                    if duration <= 0:
                        continue
                    start = max(job_ready[job_id], machine_ready[machine_id])
                    end = start + duration
                    if best_end is None or end < best_end:
                        best_end = end
                        best_machine = machine_id
                
                if best_machine is not None:
                    machine_ready[best_machine] = best_end
                    job_ready[job_id] = best_end
        
        return max(machine_ready) if machine_ready else 0