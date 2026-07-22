"""
This file provides the test_model function to evaluate an agent or a heuristic on a set of instances.
Furthermore, test_model_and_heuristics can be used to evaluate an agent and all heuristics specified in the
TEST_HEURISTICS constant on the same set of instances.
Using this file requires a testing config. For example, it is necessary to specify the name of the model
you want to test.
Running this file will automatically call test_model_and_heuristics.
You can adapt the heuristics used for testing in the TEST_HEURISTICS constant. An empty list is admissible.
When running the file from a console you can use --plot-ganttchart to show the generated gantt_chart figures.

# >>> FAIRNESS (overview of changes)
#   The stochastic baselines (NSGA-II and the sampling agent) are now run with K independent
#   seeds per instance, while the deterministic methods (greedy agent, dispatching heuristics)
#   stay at a single run. K is read from config['NSGA2']['n_runs'].
#   The raw-solution contract is unified: raw_solutions[algo][instance] is ALWAYS a list over
#   runs of solution-point lists. Deterministic methods store a single-element list.
#   For the normalized metrics, every run of a stochastic method is passed to evaluate_fronts as
#   its own key, so the combined reference front pools all runs of all methods and every run is
#   scored against that shared front. The per-run scores are then averaged back per method. This
#   avoids the best-of-K bias that scoring the union of the runs would introduce.
#   NSGA-II's budget (population_size, generations) is read from config['NSGA2'] and is scaled per
#   problem size in the config, not here.
"""
import argparse
from pathlib import Path
import itertools
from typing import Tuple, List, Dict, Union
import numpy as np
import torch
from tqdm import tqdm
import time
import copy
from sklearn.model_selection import train_test_split
import os
from src.environments.environment_loader import EnvironmentLoader
from src.agents.heuristic.heuristic_agent import HeuristicSelectionAgent
from src.utils.evaluations import EvaluationHandler
from src.utils.logger import Logger
from src.utils.file_handler.data_handler import DataHandler
from src.utils.file_handler.model_handler import ModelHandler
from src.data_generator.task import Task
from src.agents.train_test_utility_functions import get_agent_class_from_config, load_config, load_data
from src.agents.solver import get_solver
from src.agents.metaheuristic.nsga2 import NSGA2Scheduler, Individual, select_best_weighted_solution
from src.agents.Run_Episode import run_episode, run_episode_rl
from src.visuals_generator.GreenFJSP_Functions import get_pareto_front
from src.utils.normalized_hv import evaluate_fronts  # per-instance norm-HV, spacing, IGD
# constants
TEST_HEURISTICS: List[str] = ['rand', 'LPT', 'MTR', 'LTR', 'EDT', 'MWKR', 'energy', 'MEM', 'SW']


class Individual:
    def __init__(
        self,
        objectives: List[float]
    ):
        self.objectives: List[float] = objectives


def uniform_grid_pairs(M, H):
    """
    Generate uniform lambda vectors on the M-dimensional simplex.
    For M=2, this returns exactly H equally spaced vectors.
    """
    if H <= 0:
        return []
    # avoid division by zero; one neutral weight vector
    if H == 1:
        return [tuple([1.0 / M] * M)]

    def compositions(n, k):
        """All k-tuples of nonnegative integers summing to n."""
        if k == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for tail in compositions(n - i, k - 1):
                yield (i,) + tail
    grid_n = H - 1
    return [tuple(h_i / grid_n for h_i in h) for h in compositions(grid_n, M)]


def get_action(env, model, heuristic_id: str, heuristic_agent: Union[HeuristicSelectionAgent, None], deterministic: bool = False) -> Tuple[int, str]:
    """
    This function determines the next action according to the input model or heuristic
    :param env: Environment object
    :param model: Model object. E.g. PPO object
    :param heuristic_id: Heuristic identifier. Can be None
    :param heuristic_agent: HeuristicSelectionAgent object. Can be None
    :return: ID of the selected action
    """
    assert bool(heuristic_id) != bool(model), \
        "You have to pass an agent model XOR a heuristic id to solve the scheduling problem"
    if heuristic_id:
        action_mode = 'heuristic'
        tasks = env.tasks
        mask = env.get_action_mask_heuristic()
        task_mask = mask
        if heuristic_id != 'LRT':
            selected_action = heuristic_agent(tasks, task_mask, heuristic_id)
        else:
            selected_action = heuristic_agent(tasks, task_mask, heuristic_id, env.previous_task_machine, env.Task_setup_times)
    else:
        action_mode = 'agent'
        obs = env._state_obs
        mask = env.get_action_mask()
        selected_action, _ = model.predict(observation=obs, action_mask=mask, deterministic=deterministic)
    return selected_action, action_mode


def select_best_weighted_solution(pareto_front: list, weights: list = [0.5, 0.5]):
    """Selects the best solution from the Pareto front via min-max normalization."""
    if not pareto_front:
        return None
    if len(pareto_front) == 1:
        return pareto_front[0]
    makespans = np.array([ind.objectives[0] for ind in pareto_front])
    energies = np.array([ind.objectives[1] for ind in pareto_front])

    min_ms, max_ms = makespans.min(), makespans.max()
    min_ec, max_ec = energies.min(), energies.max()
    range_ms = max(max_ms - min_ms, 1e-9)
    range_ec = max(max_ec - min_ec, 1e-9)

    best_ind, best_score = None, float('inf')
    for ind in pareto_front:
        ms, ec = ind.objectives[0], ind.objectives[1]
        score = weights[0] * ((ms - min_ms) / range_ms) + weights[1] * ((ec - min_ec) / range_ec)
        if score < best_score:
            best_score = score
            best_ind = ind

    return best_ind


def log_results_MO(plot_logger: Logger, heuristic: str, hypervolume: float,
                   time: float, best_makespan: float, best_cost: float, spacing: float = 0.0, Pareto_front_image=None
                   ):

    metrics = [hypervolume, spacing, time, best_makespan, best_cost]

    plot_logger.add_row_to_wandb_table_MO(heuristic, metrics, Pareto_front_image)


def log_results_SO(plot_logger: Logger, heuristic: str, weights: List,
                env, rewards, greedy: bool = False,
                plot_gantchart: bool = False) -> None:
    makespan = env.get_makespan()
    energy_consumed, total_costs = env.consumed_energy_and_costs()
    measures = []
    measures.append(rewards)
    measures.append(makespan)
    measures.append(energy_consumed.item())
    measures.append(total_costs.item())
    measures.append(str(weights))
    # Get visualizations if needed
    Charts = [None] * 7
    if plot_gantchart:
        # Get image visualizations for table
        Charts[0] = env.render(mode="image")
        Charts[1] = env.get_solar_plot(show_plot=False)
        Charts[2] = env.get_energy_price_plot(show_plot=False)
        Charts[3] = env.get_energy_consumption_plot(show_plot=False, makespan=makespan)
        Charts[4] = env.get_cost_per_timestep(show_plot=False, makespan=makespan)
        Charts[5] = env.get_cumulative_energy_plot(show_plot=False, makespan=makespan)
        Charts[6] = env.get_cumulative_cost_plot(show_plot=False, makespan=makespan)

    # Generate a unique name for this run
    agent_name = heuristic if heuristic else "RL-Agent"
    if not heuristic and greedy:
        agent_name += " Greedy"

    plot_logger.add_row_to_wandb_table_SO(agent_name, measures, Charts)


# >>> FAIRNESS helpers ------------------------------------------------------
def _build_run_aware_inputs(raw_solutions, raw_times, compare, n_instances):
    """
    Expand the unified raw-solution contract into per-run keys for evaluate_fronts.

    raw_solutions[algo][i] is a list over runs of solution-point lists. Deterministic
    methods have one run, stochastic methods have K runs. Each run becomes its own key
    so that (a) the combined reference front pools all runs of all methods and (b) each
    run is scored separately against that shared front.

    Returns (solutions, times, key_to_algo) where solutions/times are keyed by run-key
    and key_to_algo maps each run-key back to its base method.
    """
    solutions, times, key_to_algo = {}, {}, {}
    for algo in compare:
        per_inst = raw_solutions.get(algo)
        if per_inst is None or len(per_inst) != n_instances:
            have = 0 if per_inst is None else len(per_inst)
            print(f"[metrics] uebersprungen: {algo} hat {have}/{n_instances} Instanzen")
            continue
        runs0 = per_inst.get(0, [])
        n_runs_algo = len(runs0) if isinstance(runs0, list) else 0
        if n_runs_algo == 0:
            print(f"[metrics] uebersprungen: {algo} hat keine Laeufe")
            continue
        t_algo = raw_times.get(algo, {})
        for r in range(n_runs_algo):
            key = algo if n_runs_algo == 1 else f"{algo}@run{r}"
            key_to_algo[key] = algo
            solutions[key] = [per_inst[i][r] for i in range(n_instances)]
            times[key] = {i: t_algo.get(i, 0.0) for i in range(n_instances)}
    return solutions, times, key_to_algo


def _reaggregate_runs(metrics, key_to_algo, compare):
    """
    Collapse the per-run results from evaluate_fronts back into one entry per method.

    For every instance the per-run metric values of a method are averaged. The aggregate
    is then recomputed from the averaged per-instance values, so mean and std over the
    fifty instances describe the method, not the individual runs.
    """
    METRIC_KEYS = ("norm_hv", "igd", "spacing", "time")
    per_instance_raw = metrics.get("per_instance", {})
    per_instance = {}
    for inst_key, algo_map in per_instance_raw.items():
        grouped, out = {}, {}
        for k, vals in algo_map.items():
            if not isinstance(vals, dict):
                out[k] = vals  # e.g. reference_front_size
                continue
            base = key_to_algo.get(k, k)
            grouped.setdefault(base, []).append(vals)
        for base, lst in grouped.items():
            out[base] = {mk: float(np.mean([v[mk] for v in lst if mk in v]))
                         for mk in METRIC_KEYS if any(mk in v for v in lst)}
        per_instance[inst_key] = out

    aggregate = {}
    for base in compare:
        rows = [pi[base] for pi in per_instance.values() if base in pi and isinstance(pi[base], dict)]
        if not rows:
            continue
        entry = {"n_instances": len(rows)}
        for mk in METRIC_KEYS:
            arr = [r[mk] for r in rows if mk in r]
            if arr:
                entry[f"{mk}_mean"] = float(np.mean(arr))
                entry[f"{mk}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        aggregate[base] = entry
    return per_instance, aggregate
# <<< FAIRNESS helpers ------------------------------------------------------


def test_joined_heuristics(config: Dict, data: List[List[Task]], logger: Logger, plot: bool = None, log_episode: bool = None,
               model=None, heuristics: List[str] = None, intermediate_test_idx=None, greedy: bool=False, num_episodes: int = 1, nadir_refs=None,
               raw_solutions=None, algo_key: str = "Joined Heuristics", raw_times=None) -> dict:
    """
    This function tests all heuristics specified in the TEST_HEURISTICS constant on the provided test data

    :param config: Environment configuration
    :param data: Data containing problem instances used for testing
    :param logger: Logger object
    :param plot: Plot a gantt chart of all tests
    :param heuristics: List of heuristic identifiers to test
    :param log_episode: If true, logs episode results as table
    :return: evaluation metrics for all heuristics
    """
    # Create evaluation handler
    eval_handler = EvaluationHandler()
    config = copy.deepcopy(config)
    max_steps = config.get('num_steps_max', 1000)
    for test_i in range(len(data)):
        objectives = []
        start_time = time.perf_counter()

        for heuristic_id in heuristics:
            # Create environment
            environment, _ = EnvironmentLoader.load(
                config,
                data=[data[test_i]],
                agent_name=heuristic_id,
                check_env_agent_compatibility=False
            )
            environment.runs = test_i
            environment, mean_reward = run_episode(environment,
                                               model,
                                               heuristic_id,
                                               eval_handler,
                                               episode_iter=num_episodes,
                                               greedy=greedy,
                                               max_steps=max_steps,
                                               data_index = test_i)
            for env, reward in environment:
                _, total_costs = env.consumed_energy_and_costs()
                makespan = env.get_makespan()
                objectives.append((makespan.item(), total_costs.item()))
        objectives_ind = [Individual(list(obj)) for obj in objectives]
        # >>> FAIRNESS: unified contract, deterministic method -> single run
        if raw_solutions is not None:
            raw_solutions.setdefault(algo_key, {})[test_i] = [[tuple(obj) for obj in objectives]]
        # <<< FAIRNESS
        # Stop the solve timer here so subsequent metric computations are
        # NOT included in the reported time
        elapsed_time = time.perf_counter() - start_time
        # Time per instance for the per-instance evaluation
        if raw_times is not None:
            raw_times.setdefault(algo_key, {})[test_i] = elapsed_time
        if nadir_refs is not None and test_i in nadir_refs:
            nadir_ref = nadir_refs[test_i]
        else:
            nsga_tmp = NSGA2Scheduler(data[test_i])
            nadir_ref = nsga_tmp.compute_nadir_reference(buffer=1.5)
        hypervolume = NSGA2Scheduler._hypervolume_2d(objectives_ind, ref=nadir_ref)
        spacing_val = NSGA2Scheduler.spacing(objectives_ind)
        if plot:
            pareto_front = get_pareto_front(pareto=objectives_ind)
        else:
            pareto_front = None
        best_makespan = min([obj[0] for obj in objectives])
        best_cost = min([obj[1] for obj in objectives])
        eval_handler.hypervolumes.append(hypervolume)
        eval_handler.spacings.append(spacing_val)
        eval_handler.time_taken.append(elapsed_time)
        eval_handler.makespan.append(best_makespan)
        eval_handler.total_costs.append(best_cost)
        log_results_MO(logger, "Joined Heuristics", hypervolume, elapsed_time, best_makespan, best_cost, spacing = spacing_val, Pareto_front_image=pareto_front)
    return eval_handler.evaluate_test()


def test_heuristics(config: Dict, data: List[List[Task]], logger: Logger, plot: bool = None, log_episode: bool = None,
               model=None, heuristic_id: str = None, num_episodes: int = 1, nadir_refs=None,
               raw_solutions=None, algo_key: str = None, raw_times=None) -> dict:
    """
    This function tests a model in the passed environment for all problem instances passed as data_test and returns an
    evaluation summary
    """
    # Create evaluation handler
    evaluation_handler = EvaluationHandler()

    Number_weights = config.get('Objective', {}).get('Number_Weights', 10)
    Number_objectives = config.get('Objective', {}).get('Num_objectives', 2)
    lambda_weights = uniform_grid_pairs(Number_objectives, Number_weights)
    # Disable visualization for greedy mode with constraint violations
    plot_ganttchart = plot and config.get('Ganttcharts', False)
    Success_Ratio = 0
    instance_wall_time = []
    max_steps = config.get('num_steps_max', 1000)
    for test_i in range(len(data)):
        # track time
        start_time = time.perf_counter()
        # Create environment
        environment, _ = EnvironmentLoader.load(
            config,
            data=[data[test_i]],
            agent_name=heuristic_id,
            check_env_agent_compatibility=False
        )
        environment.runs = test_i
        environments, mean_reward = run_episode(environment,
                                           model,
                                           heuristic_id,
                                           evaluation_handler,
                                           episode_iter=num_episodes,
                                           max_steps=max_steps,
                                           data_index = test_i)
        objectives = []
        for env, reward in environments:
            _, total_costs = env.consumed_energy_and_costs()
            makespan = env.get_makespan()
            objectives.append(Individual((makespan.item(), total_costs.item())))
        # >>> FAIRNESS: unified contract, deterministic method -> single run
        if raw_solutions is not None:
            key = algo_key if algo_key is not None else heuristic_id
            raw_solutions.setdefault(key, {})[test_i] = [[
                (ind.objectives[0], ind.objectives[1]) for ind in objectives
            ]]
        # <<< FAIRNESS
        # Stop the solve timer here so subsequent metric computations are
        # NOT included in the reported time
        elapsed_time = time.perf_counter() - start_time
        # collect metrics
        instance_wall_time.append(elapsed_time)
        # Time per instance for the per-instance evaluation
        if raw_times is not None:
            key = algo_key if algo_key is not None else heuristic_id
            raw_times.setdefault(key, {})[test_i] = elapsed_time
        if nadir_refs is not None and test_i in nadir_refs:
            nadir_ref = nadir_refs[test_i]
        else:
            nsga_tmp = NSGA2Scheduler(data[test_i])
            nadir_ref = nsga_tmp.compute_nadir_reference(buffer=1.5)
        hypervolume = NSGA2Scheduler._hypervolume_2d(objectives, ref=nadir_ref)
        spacing_val = NSGA2Scheduler.spacing(objectives)
        evaluation_handler.hypervolumes.append(hypervolume)
        evaluation_handler.spacings.append(spacing_val)
        evaluation_handler.time_taken.append(elapsed_time)
        evaluation_handler.makespan.append(makespan.item())
        evaluation_handler.total_costs.append(total_costs.item())
        # Mode-dependent weight for logging
        mode = config.get('Objective', {}).get('mode', 'single_makespan')
        if mode == 'single_makespan':
            weights = [1.0, 0.0]
        elif mode == 'single_energy':
            weights = [0.0, 1.0]
        else:
            weights = None
        # Log results only if environment is valid (no constraint violations in greedy mode)
        if log_episode:
            # For greedy mode with constraint breaks, disable plotting regardless of config
            should_plot = plot_ganttchart
            environment = environments[0][0]  # Take the first environment for logging and visualization
            log_results_SO(
                logger,
                heuristic_id,
                weights,
                environment,
                0,
                greedy=True,
                plot_gantchart=should_plot
            )
            if should_plot:
                pareto_front = get_pareto_front(pareto=[Individual(objectives)])
            else:
                pareto_front = None
            log_results_MO(logger, heuristic_id, hypervolume, elapsed_time, makespan.item(), total_costs.item(), Pareto_front_image=pareto_front)
    return evaluation_handler.evaluate_test()


def test_model(config: Dict, data: List[List[Task]], logger: Logger, plot: bool = None, log_episode: bool = None,
               model=None, nadir_refs=None, num_episodes=1, greedy=True, name=None, raw_solutions=None, raw_times=None,
               n_runs: int = 1) -> dict:
    """
    Tests an RL model on the provided test data.

    # >>> FAIRNESS
    #   n_runs is the number of independent outer evaluation repeats with distinct seeds.
    #   It is applied ONLY when greedy is False, because the sampling agent is stochastic.
    #   The greedy agent is deterministic and always uses a single run. num_episodes is the
    #   unchanged inner non-greedy episode count per weight vector.
    """
    # Create evaluation handler
    eval_handler = EvaluationHandler()
    config = copy.deepcopy(config)
    objective_config = config.get('Objective', {})
    mode = objective_config.get('mode', 'single_makespan')
    if mode == 'single_makespan':
        lambda_weights = [(1.0, 0.0)] # Default to makespan only
    elif mode == 'single_energy':
        lambda_weights = [(0.0, 1.0)] # energy only
    else:  # multi
        N = objective_config.get('Number_Weights', 10)
        lambda_weights = uniform_grid_pairs(2, N)
    objective_weights_eval = objective_config.get('Eval_Weights', [[0.5, 0.5]])
    #get closest weights from lambda_weights to objective_weights_eval for logging
    closest_weights = [min(lambda_weights, key=lambda x: np.linalg.norm(np.array(x) - np.array(eval_weight))) for eval_weight in objective_weights_eval]
    max_steps = config.get('num_steps_max', 1000)
    device = torch.device("cpu")

    # >>> FAIRNESS: deterministic greedy -> 1 run, stochastic sampling -> K runs
    n_eval_runs = n_runs if not greedy else 1
    base_seed = config.get('seed', 42)
    # <<< FAIRNESS

    for test_i, instance in tqdm(enumerate(data), desc='Computing Model'):
        env, _ = EnvironmentLoader.load(config, data=[instance], agent_name='solver', check_env_agent_compatibility=False)

        # >>> FAIRNESS: collect one solution cloud per run
        run_clouds = []   # list over runs of [(makespan, cost), ...]
        run_hvs = []
        run_times = []

        for run_idx in range(n_eval_runs):
            if not greedy:
                torch.manual_seed(base_seed + run_idx)
                np.random.seed(base_seed + run_idx)

            time_taken = 0
            Makespans = []
            costs = []

            for objective_weights in lambda_weights:
                # Run episode with RL model
                time_start = time.time()
                envs, mean_reward = run_episode_rl(
                    env,
                    model,
                    objective_weights,
                    eval_handler,
                    greedy=greedy,
                    episode_iter=num_episodes,
                    max_steps=max_steps,
                    data_index = test_i,
                    device=device
                )
                time_end = time.time()
                time_taken += (time_end - time_start)

                max_reward = -np.inf
                env_logging = None
                for env_r, mr in envs:
                    makespan = env_r.get_makespan().item()
                    _, total_costs = env_r.consumed_energy_and_costs()
                    Makespans.append(makespan)
                    costs.append(total_costs.item())
                    if mr > max_reward:
                        max_reward = mr
                        env_logging = env_r

                # log and bookkeep only on the first run to keep table cardinality stable
                if run_idx == 0:
                    eval_handler.rewards.append(mean_reward)
                    if log_episode and objective_weights in closest_weights and env_logging is not None:
                        log_results_SO(
                            logger,
                            name,
                            str(objective_weights),
                            env_logging,
                            None,
                            greedy=False,
                            plot_gantchart=plot
                        )

            cloud = [(Makespans[i], costs[i]) for i in range(len(Makespans))]
            run_clouds.append(cloud)
            run_times.append(time_taken)

            obj_hv_run = [Individual([m, c]) for (m, c) in cloud]
            if obj_hv_run:
                if nadir_refs is not None and test_i in nadir_refs:
                    nadir_ref = nadir_refs[test_i]
                else:
                    nsga_tmp = NSGA2Scheduler(instance)
                    nadir_ref = nsga_tmp.compute_nadir_reference(buffer=1.5)
                run_hvs.append(NSGA2Scheduler._hypervolume_2d(obj_hv_run, ref=nadir_ref))
            else:
                run_hvs.append(0.0)
        # <<< FAIRNESS

        # representative run (the first) for best-solution selection and plotting
        cloud0 = run_clouds[0] if run_clouds else []
        obj_hv = [Individual([m, c]) for (m, c) in cloud0]

        spacing_val = 0.0  # guard for empty fronts
        if obj_hv:
            best_ind = select_best_weighted_solution(obj_hv, weights=objective_weights_eval[0])
            best_makespan = best_ind.objectives[0]
            best_cost = best_ind.objectives[1]
            spacing_val = NSGA2Scheduler.spacing(obj_hv)
        else:
            best_makespan = None
            best_cost = None

        mean_hv = float(np.mean(run_hvs)) if run_hvs else 0.0
        mean_time = float(np.mean(run_times)) if run_times else 0.0

        # >>> FAIRNESS: per-run clouds for the reference front, mean run time for reporting
        if raw_solutions is not None and name is not None:
            raw_solutions.setdefault(name, {})[test_i] = run_clouds
        if raw_times is not None and name is not None:
            raw_times.setdefault(name, {})[test_i] = mean_time
        # <<< FAIRNESS

        eval_handler.makespan.append(best_makespan)
        eval_handler.total_costs.append(best_cost)
        eval_handler.hypervolumes.append(mean_hv)
        eval_handler.spacings.append(spacing_val)
        eval_handler.time_taken.append(mean_time)

        if plot:
            pareto_front = get_pareto_front(pareto=obj_hv)
        else:
            pareto_front = None

        if log_episode:
            log_results_MO(logger, "RL Agent", mean_hv, mean_time, best_makespan, best_cost, spacing=spacing_val, Pareto_front_image=pareto_front)
    return eval_handler.evaluate_test()


def test_solver(config: Dict, data_test: List[List[Task]], logger: Logger, plot: bool = None, log_episode: bool = None) -> dict:
    """"
    This function tests a model in the passed environment for all problem instances passed as data_test and returns an
    evaluation summary"
    """
    # create evaluation handler
    eval_handler = EvaluationHandler()
    config = copy.deepcopy(config)
    solver_time = config.get('time_limit', 3000.0)
    objective_config = config.get('Objective', {})
    Multi_Objective = objective_config.get('MOFJSP', False)
    if not Multi_Objective:
        objective_weights = objective_config.get('Weights', [1.0, 0.0]) # Default to makespan only
    else:
        objective_weights = objective_config.get('Weights', [0.5, 0.5])
    objective_weights_eval = objective_config.get('Eval_Weights', [[0.5, 0.5]])
    solver = get_solver(config)
    Number_weights = config.get('Objective', {}).get('Number_Weights', 10)
    Number_objectives = config.get('Objective', {}).get('Num_objectives', 2)
    lambda_weights = uniform_grid_pairs(Number_objectives, Number_weights)
    #get closest weights from lambda_weights to objective_weights_eval for logging
    closest_weights = [min(lambda_weights, key=lambda x: np.linalg.norm(np.array(x) - np.array(eval_weight))) for eval_weight in objective_weights_eval]
    # for each test instance
    for test_i,instance in tqdm(enumerate(data_test), desc='Computing solver solution if necessary'):
        env, _ = EnvironmentLoader.load(config, data=[instance], agent_name='solver', check_env_agent_compatibility=False)
        env.reset()
        env.tasks_original = copy.deepcopy(env.tasks)

        time_taken = 0
        Makespans = []
        costs = []
        for objective_weights in lambda_weights:
            time_start = time.time()
            solved_instance, Success = solver.optimize(instance, objective_weights=objective_weights, time_limit_seconds=solver_time)
            time_end = time.time()
            time_taken += (time_end - time_start)
            if Success:
                env.tasks = copy.deepcopy(solved_instance)
                solved_env = eval_handler.update_episode_with_solver(env)
                makespan = solved_env.get_makespan().item()
                _, total_costs = solved_env.consumed_energy_and_costs()
                print(total_costs)
                Makespans.append(makespan)
                costs.append(total_costs.item())

                if log_episode:
                    if objective_weights in closest_weights:
                        log_results_SO(
                            logger,
                            "Solver",
                            str(objective_weights),
                            env,
                            None,
                            greedy=False,
                            plot_gantchart=plot
                        )

        best_makespan = min(Makespans) if Makespans else None
        best_cost = min(costs) if costs else None
        nsga_tmp = NSGA2Scheduler(
            instance,
        )
        obj_hv = [Individual([Makespans[i], costs[i]]) for i in range(len(Makespans))]
        nadir_ref = nsga_tmp.compute_nadir_reference(buffer=1.5)
        hypervolume = NSGA2Scheduler._hypervolume_2d(obj_hv, ref=nadir_ref)
        eval_handler.makespan.append(best_makespan)
        eval_handler.total_costs.append(best_cost)
        eval_handler.hypervolumes.append(hypervolume)
        eval_handler.time_taken.append(time_taken)
        if plot:
            pareto_front = get_pareto_front(pareto=obj_hv)
        else:
            pareto_front = None
        log_results_MO(logger, "Solver", hypervolume, time_taken, best_makespan, best_cost, Pareto_front_image=pareto_front)
    return eval_handler.evaluate_test()


def test_nsga2(config: Dict, data_test: List[List[Task]], logger: Logger, plot: bool = None, log_episode: bool = None, nadir_refs=None,
               raw_solutions=None, algo_key: str = "NSGA2", raw_times=None) -> dict:
    """
    This function tests the NSGA-II solver for all problem instances passed as data_test.

    # >>> FAIRNESS
    #   population_size and generations are read from config['NSGA2'] and scaled per problem
    #   size in the config. NSGA-II is stochastic, so it is run for n_runs independent seeds
    #   per instance. The per-instance scalars reported to WandB are the mean over the runs,
    #   and each run's Pareto front is stored separately for the shared reference front.
    """
    eval_handler = EvaluationHandler()
    config = copy.deepcopy(config)

    plot_ganttchart = config.get('Ganttcharts', False)

    # Get multi-objective weights from the config
    objective_config = config.get('Objective', {})

    # Mode-dependent weights for SO logging
    mode = objective_config.get('mode', 'single_makespan')
    if mode == 'single_makespan':
        objective_weights = [[1.0, 0.0]]
    elif mode == 'single_energy':
        objective_weights = [[0.0, 1.0]]
    else:
        objective_weights = objective_config.get('Eval_Weights', [[0.5, 0.5]])

    nsga2_config = config.get('NSGA2', {})
    pop_size = nsga2_config.get('population_size', 100)
    generations = nsga2_config.get('generations', 100)
    # >>> FAIRNESS: K independent runs with distinct seeds
    n_runs = nsga2_config.get('n_runs', 1)
    base_seed = nsga2_config.get('seed', config.get('seed', 42))
    # <<< FAIRNESS

    for test_i, instance in tqdm(enumerate(data_test), desc='Computing NSGA-II solutions'):

        # Load the env to access machines, solar, and prices
        env, _ = EnvironmentLoader.load(config, data=[instance], agent_name='NSGA2', check_env_agent_compatibility=False)
        env.reset()
        env.tasks_original = copy.deepcopy(env.tasks)
        # Build the input dictionary for NSGA-II from the environment
        instance_data = {
            "tasks": env.tasks,
            "machines": env.machine_info,
            "solar_profile": env.solar_info,
            "energy_prices": env.energy_prices
        }

        # fixed nadir reference for all runs of this instance, if precomputed
        if nadir_refs is not None and test_i in nadir_refs:
            nadir_ref = nadir_refs[test_i]
        else:
            nadir_ref = None  # fall back to each run's internal nadir

        # >>> FAIRNESS: run K seeds, collect per-run fronts and scalars
        run_paretos = []
        run_hvs, run_times, run_makespans, run_costs = [], [], [], []
        for run_idx in range(n_runs):
            time_start = time.time()
            scheduler = NSGA2Scheduler(
                instance_data=instance_data,
                population_size=pop_size,
                generations=generations,
                seed=base_seed + run_idx
            )
            pareto, hv_prog, nadir_ref_internal, spacing_val = scheduler.run()
            run_time = time.time() - time_start

            ref = nadir_ref if nadir_ref is not None else nadir_ref_internal
            run_hvs.append(NSGA2Scheduler._hypervolume_2d(pareto, ref=ref))
            run_times.append(run_time)
            run_paretos.append(pareto)
            if pareto:
                run_makespans.append(min(ind.objectives[0] for ind in pareto))
                run_costs.append(min(ind.objectives[1] for ind in pareto))

        # per-instance scalars are the mean over the runs
        mean_hv = float(np.mean(run_hvs)) if run_hvs else 0.0
        mean_time = float(np.mean(run_times)) if run_times else 0.0
        best_makespan = float(np.mean(run_makespans)) if run_makespans else None
        best_cost = float(np.mean(run_costs)) if run_costs else None
        spacing_val = NSGA2Scheduler.spacing(run_paretos[0]) if run_paretos and run_paretos[0] else 0.0

        # per-run fronts for the shared reference front, mean time for reporting
        if raw_solutions is not None:
            raw_solutions.setdefault(algo_key, {})[test_i] = [
                [(ind.objectives[0], ind.objectives[1]) for ind in p] for p in run_paretos
            ]
        if raw_times is not None:
            raw_times.setdefault(algo_key, {})[test_i] = mean_time
        # <<< FAIRNESS

        eval_handler.makespan.append(best_makespan)
        eval_handler.total_costs.append(best_cost)
        eval_handler.hypervolumes.append(mean_hv)
        eval_handler.spacings.append(spacing_val)
        eval_handler.time_taken.append(mean_time)

        # log once per instance using the run closest to the mean HV as representative
        if log_episode and run_hvs:
            rep_idx = int(np.argmin([abs(h - mean_hv) for h in run_hvs]))
            rep_pareto = run_paretos[rep_idx]
            should_plot = plot_ganttchart

            for eval_weights in objective_weights:
                best_ind = select_best_weighted_solution(rep_pareto, weights=eval_weights)
                if best_ind is None:
                    continue
                for task in env.tasks:
                    schedule_info = best_ind.schedule_details.get((task.job_index, task.task_index))
                    if schedule_info:
                        task.started = int(schedule_info["start"])
                        task.finished = int(schedule_info["end"])
                        task.selected_machine = int(schedule_info["machine"])
                    task.done = True
                # Update makespan in the env (needed for the plots and the EvalHandler)
                env.makespan = max([t.finished for t in env.tasks if t.done])

                log_results_SO(
                    logger,
                    "NSGA2",
                    eval_weights,
                    env,
                    None,
                    greedy=False,
                    plot_gantchart=should_plot
                )

            if should_plot:
                pareto_front = get_pareto_front(pareto=rep_pareto)
            else:
                pareto_front = None
            log_results_MO(logger, "NSGA2", mean_hv, mean_time, best_makespan, best_cost, spacing=spacing_val, Pareto_front_image=pareto_front)
    return eval_handler.evaluate_test()


def get_perser_args():
    # Arguments for function
    parser = argparse.ArgumentParser(description='Test Agent in Production Scheduling Environment')
    parser.add_argument('-fp', '--config_file_path', type=str, required=True,
                        help='Path to config file you want to use for training')
    parser.add_argument('-plot', '--plot-ganttchart', dest="plot_ganttchart", action="store_true",
                        help='Enable or disable model result plot.')
    parser.add_argument('-nge', '--non_greedy_epsiodes', type=int, default=100,
                        help='Number of episodes to run for non greedy agent')
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


def final_evaluation(config: dict, test_data: List[List[Task]], logger: Logger) -> None:
    results = {}
    # Collect raw solutions per algorithm and instance, so that a shared
    # reference front can be built at the end
    # >>> FAIRNESS: raw_solutions[algo][i] is now a LIST OVER RUNS of point lists
    raw_solutions: Dict[str, Dict[int, list]] = {}
    # Collect wall-clock time per algorithm and instance (mean over runs for stochastic)
    raw_times: Dict[str, Dict[int, float]] = {}

    # >>> FAIRNESS: single source for the number of independent runs of the stochastic methods
    k_runs = config.get('NSGA2', {}).get('n_runs', 1)
    # <<< FAIRNESS

    # Precompute the nadir reference once for all test instances
    print("Precomputing nadir reference points for final evaluation")
    nadir_refs = {}
    for i, instance in enumerate(test_data):
        nsga_tmp = NSGA2Scheduler(instance)
        nadir_refs[i] = nsga_tmp.compute_nadir_reference(buffer=1.5)

    for heuristic in config.get("test_heuristics", ["rand"]):
        res = test_heuristics(config=config,
                              data=copy.deepcopy(test_data),
                              num_episodes= 1,
                              logger=logger,
                              heuristic_id=heuristic,
                              log_episode=True,
                              plot = config.get('Ganttcharts'),
                              nadir_refs=nadir_refs,
                              raw_solutions=raw_solutions,
                              raw_times=raw_times)
        results.update({heuristic:res})

    heuristic_list = config.get("test_heuristics", ["rand"])
    if len(heuristic_list) > 1:
        res = test_joined_heuristics(config=config,
                                     data=copy.deepcopy(test_data),
                                     logger=logger,
                                     heuristics=heuristic_list,
                                     log_episode=True,
                                     num_episodes=1,
                                     plot = config.get('Ganttcharts'),
                                     nadir_refs=nadir_refs,
                                     raw_solutions=raw_solutions,
                                     raw_times=raw_times)
        results.update({"Joined Heuristics": res})

    if config.get('test_nsga2', False):
        # >>> FAIRNESS: NSGA-II is stochastic, runs k_runs seeds (read inside via config['NSGA2'])
        res = test_nsga2(config=config,
                         data_test=copy.deepcopy(test_data),
                         logger=logger,
                         log_episode=True,
                         plot=config.get('Ganttcharts'),
                         nadir_refs=nadir_refs,
                         raw_solutions=raw_solutions,
                         raw_times=raw_times
                         )
        results.update({"NSGA2": res})

    # # get model path from config
    best_model_path = ModelHandler.get_best_model_path(config)
    model = get_agent_class_from_config(config=config).load(file=best_model_path, config=config, logger=logger)

    # RL-Agent, greedy (deterministisch, eine Episode, ein Lauf)
    res = test_model(
         config=config,
         data=copy.deepcopy(test_data),
         greedy=True,
         num_episodes=1,
         logger=logger,
         model=model,
         log_episode=True,
         name="RL-Agent-Greedy",
         plot=config.get('Ganttcharts'),
         nadir_refs=nadir_refs,
         raw_solutions=raw_solutions,
         raw_times=raw_times,
         n_runs=1,
         )
    results.update({"RL-Agent-Greedy": res})

    # RL agent, sampled (stochastic, n=3 episodes per weight vector, k_runs seeds)
    res = test_model(
         config=config,
         data=copy.deepcopy(test_data),
         greedy=False,
         num_episodes=config.get('non_greedy_episodes', 3),
         logger=logger,
         model=model,
         log_episode=True,
         name="RL-Agent-Sampled",
         plot=config.get('Ganttcharts'),
         nadir_refs=nadir_refs,
         raw_solutions=raw_solutions,
         raw_times=raw_times,
         n_runs=k_runs,                       # >>> FAIRNESS
         )
    results.update({"RL-Agent-Sampled": res})

    if config.get('test_solver', False):
        # Test with solver
        res = test_solver(config=config,
                          data_test=copy.deepcopy(test_data),
                          logger=logger,
                          log_episode=True,
                          plot = config.get('Ganttcharts'))
        results.update({"solver":res})
        results = EvaluationHandler.add_solver_gap_to_results(results)

    # ------------------------------------------------------------------
    # Per-instance metrics against a shared reference front.
    # >>> FAIRNESS
    #   Each run of each stochastic method is entered as its own key in
    #   evaluate_fronts. This way the reference front pools all runs of all methods,
    #   and each run is scored against the same front. The per-run values are then
    #   averaged back per method. The four target algorithms are compared.
    # ------------------------------------------------------------------
    COMPARE = ["NSGA2", "Joined Heuristics", "RL-Agent-Greedy", "RL-Agent-Sampled"]
    n_instances = len(test_data)

    solutions, times, key_to_algo = _build_run_aware_inputs(raw_solutions, raw_times, COMPARE, n_instances)

    if solutions:
        metrics = evaluate_fronts(solutions, times=times)            # ref (1.1, 1.1)
        per_instance, aggregate = _reaggregate_runs(metrics, key_to_algo, COMPARE)
        # <<< FAIRNESS

        # Console table of the aggregates
        print("\n" + "=" * 84)
        print(f"Metrics over {n_instances} test instances (normalized space, IGD against combined front)")
        print(f"{'Algorithm':22s} {'normHV':>9s} {'+/-':>7s} {'Spacing':>9s} "
              f"{'IGD':>9s} {'Time[s]':>9s}")
        print("-" * 84)
        for algo in COMPARE:
            if algo not in aggregate:
                continue
            m = aggregate[algo]
            print(f"{algo:22s} {m.get('norm_hv_mean', 0):9.4f} {m.get('norm_hv_std', 0):7.4f} "
                  f"{m.get('spacing_mean', 0):9.4f} {m.get('igd_mean', 0):9.4f} {m.get('time_mean', 0):9.3f}")
        print("=" * 84)

        # Per-instance JSON in detailed_results/ with a problem-size-specific name
        try:
            import json, re
            label_source = config.get('saved_model_name', '') or config.get('instances_file', '')
            m = re.search(r'(\d+x\d+(?:x\d+)?)', label_source)
            size_tag = m.group(1) if m else 'unknown'
            out_dir = 'detailed_results'
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'metrics_per_instance_{size_tag}.json')
            with open(out_path, 'w') as f:
                json.dump({"per_instance": per_instance, "aggregate": aggregate}, f, indent=2)
            print(f"[metrics] written: {out_path}")
        except Exception as e:
            print(f"[metrics] JSON export failed: {e}")

        # WandB summary with the aggregates
        try:
            import wandb
            if wandb.run is not None:
                flat = {}
                for algo, m in aggregate.items():
                    safe = algo.replace(' ', '_')
                    for k, v in m.items():
                        flat[f"metrics/{safe}/{k}"] = v
                wandb.summary.update(flat)
        except Exception as e:
            print(f"[metrics] WandB logging failed: {e}")

    logger.dump()
    # log evaluation to wandb
    logger.write_to_wandb_summary(results)


def compare_pareto_fronts(
    config:        dict,
    model,
    instance:      dict,
    H:             int  = 25,       # number of lambda weight vectors for the RL agent
    nsga_pop:      int  = 100,      # NSGA-II population
    nsga_gen:      int  = 100,      # NSGA-II generations
    greedy:        bool = True,
    out_path:      str  = "figures/pareto_comparison.pdf",
    seed:          int  = 42,
) -> None:
    """
    Compares the Pareto front of the trained RL agent with NSGA-II
    on the same instance.
    RL front   : H evenly spaced weight vectors lambda in the 2-simplex
                 (uniform_grid_pairs), one episode per lambda, Pareto-filtered.
    NSGA-II    : NSGA2Scheduler.run() on the same instance.
    Objectives : Makespan [IM] vs. energy cost [euro].
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from src.environments.environment_loader import EnvironmentLoader
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "axes.linewidth": 0.8, "pdf.fonttype": 42,
    })
    # ── 1. NSGA-II ────────────────────────────────────────────────────────────
    print(f"[1/2] Starting NSGA-II  (pop={nsga_pop}, gen={nsga_gen}) ...")
    scheduler = NSGA2Scheduler(
        instance, population_size=nsga_pop, generations=nsga_gen, seed=seed
    )
    nsga_pareto, _, nadir_ref, _ = scheduler.run()
    nsga_pts = [(ind.objectives[0], ind.objectives[1]) for ind in nsga_pareto]
    nsga_pts.sort(key=lambda p: p[0])
    # ── 2. RL agent  ─────────────────────────────────────────────────────────
    print(f"[2/2] Sweeping RL agent over {H} weight vectors ...")
    weight_vectors = uniform_grid_pairs(M=2, H=H)
    rl_all = []
    for lam in weight_vectors:
        env, _ = EnvironmentLoader.load(config, data=[instance])
        envs, _ = run_episode_rl(
            env, model, lambda_weights=lam,
            handler=None, greedy=greedy, episode_iter=1,
        )
        e = envs[0][0]
        makespan          = float(e.get_makespan())
        _, energy_cost    = e.consumed_energy_and_costs()
        rl_all.append((makespan, float(energy_cost)))
    # Pareto filter
    def _pareto_filter(pts):
        front = []
        for p in pts:
            if not any(o[0] <= p[0] and o[1] <= p[1] and o != p for o in pts):
                front.append(p)
        return sorted(front, key=lambda p: p[0])
    rl_pareto = _pareto_filter(rl_all)
    rl_all_ms,    rl_all_ec    = zip(*rl_all)
    rl_par_ms,    rl_par_ec    = zip(*rl_pareto)
    nsga_par_ms,  nsga_par_ec  = zip(*nsga_pts)
    # ── 3. Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(rl_par_ms, rl_par_ec,
            color="#1565C0", linewidth=1.3, marker="o", markersize=5,
            zorder=4, label="RL Pareto front")
    ax.plot(nsga_par_ms, nsga_par_ec,
            color="#E07B00", linewidth=1.3, marker="s", markersize=5,
            linestyle="--", zorder=3, label="NSGA-II Pareto front")
    ax.set_xlabel("Makespan (IM)")
    ax.set_ylabel("Energy Cost (€)")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.55)
    ax.legend(framealpha=0.9, edgecolor="#AAAAAA", loc="upper right")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(pad=0.6)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUT] {out_path}")
    print(f"      RL:     {len(rl_pareto):>3} Pareto points out of {H} weight vectors")
    print(f"      NSGA-II:{len(nsga_pts):>3} Pareto points")


def main(config_file_name: dict = None, external_config: dict = None) -> None:
    # get config and data
    config = load_config(config_file_name, None)
    # Update config with external_config values if keys match
    if external_config:
        for key, value in external_config.items():
            if value is not None and key in config:
                config[key] = value
    data = load_data(config)
    # create logger and update config
    logger = Logger(config=config)
    config = logger.config
    # Random seed for numpy as given by config
    np.random.seed(config['seed'])
    # train/test/validation data split
    if not config.get('Benchmark_Instance', False):
        split_random_seed = config['seed'] if not config.get('overwrite_split_seed', False) else 1111
        _, test_data = train_test_split(
            data, train_size=config.get('train_test_split'), random_state=split_random_seed)
        test_data, _ = train_test_split(
            test_data, train_size=config.get('test_validation_split'), random_state=split_random_seed)
    else:
        test_data = data
    final_evaluation(config=config, test_data=test_data, logger=logger)


if __name__ == '__main__':
    # get config_file from terminal input
    parse_args = get_perser_args()
    config_file_path = parse_args.config_file_path
    external_config = vars(parse_args)
    main(config_file_name=config_file_path, external_config=external_config)