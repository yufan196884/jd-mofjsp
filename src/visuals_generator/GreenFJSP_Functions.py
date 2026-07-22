"""
This file provides functions to visualize the current states of a scheduling problems as gantt charts.
Moreover, the generated gantt chart figures can be saved (e.g. as gif).
"""
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.colors as mcolors
from matplotlib import pyplot
import numpy as np
import PIL.Image

from src.data_generator.sp_factory import Task
from src.utils.ui_tools.progressbar import progressbar

# Constants
VISUALS_DIRECTORY: Path = Path(__file__).parent.parent / 'visuals'

class Individual:

    def __init__(
        self,
        objectives: List[float]
    ):
        self.objectives: List[float] = objectives

def create_colormap():
    colors = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
        '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
        '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
        '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#5254a3',
        '#6b4c9a', '#8ca252', '#bd9e39', '#ad494a', '#636363',
        '#8c6d8c', '#9c9ede', '#cedb9c', '#e7ba52', '#e7cb94',
        '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173',
        '#a55194', '#ce6dbd', '#de9ed6', '#f1b6da', '#fde0ef',
        '#636363', '#969696', '#bdbdbd', '#d9d9d9', '#f0f0f0',
        '#3182bd', '#6baed6', '#9ecae1', '#e6550d',
        '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476',
        '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc',
        '#dadaeb', '#636363', '#969696', '#bdbdbd', '#d9d9d9',
        '#f0f0f0', '#a63603', '#e6550d', '#fd8d3c', '#fdae6b',
        '#fdd0a2', '#31a354', '#74c476', '#a1d99b', '#c7e9c0',
        '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb', '#636363',
        '#969696', '#bdbdbd', '#d9d9d9', '#f0f0f0', '#6a3d9a',
        '#8e7cc3', '#b5a0d8', '#ce6dbd', '#de9ed6', '#f1b6da',
        '#fde0ef', '#3182bd', '#6baed6', '#9ecae1'
    ]
    return mcolors.ListedColormap(colors)


def plot_solar_info(solar_profile: List[float], show_plot: bool = False, 
                   quality_dpi: int = 100) -> PIL.Image:
    """
    Plot solar generation profile over time.
    
    :param solar_profile: List of solar generation values at each time step (in kW or normalized units)
    :param show_plot: Whether to display the plot
    :param quality_dpi: DPI quality of the saved figure
    :return: PIL.Image of the plot
    """
    matplotlib.use('agg')
    
    # Create figure
    fig, ax = pyplot.subplots(figsize=(12, 6), dpi=quality_dpi)
    
    # Time steps
    time_steps = list(range(len(solar_profile)))
    
    # Plot solar profile
    ax.plot(time_steps, solar_profile, color='#FDB813', linewidth=2, marker='o', markersize=4)
    ax.fill_between(time_steps, solar_profile, alpha=0.3, color='#FDB813')
    
    # Labels and title
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Solar Generation (kW)', fontsize=12)
    ax.set_title('Solar Generation Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    fig.tight_layout()
    
    # Convert to PIL Image
    fig.canvas.draw()
    image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),
                                fig.canvas.tostring_rgb())
    
    # Show if requested
    if show_plot:
        pyplot.show()
    
    # Cleanup
    pyplot.close(fig)
    
    return image


def plot_energy_prices(energy_prices: List[float], show_plot: bool = False,
                       quality_dpi: int = 100) -> PIL.Image:
    """
    Plot energy prices over time.
    
    :param energy_prices: List of energy price values at each time step (in ct/kWh or similar units)
    :param show_plot: Whether to display the plot
    :param quality_dpi: DPI quality of the saved figure
    :return: PIL.Image of the plot
    """
    matplotlib.use('agg')
    
    # Create figure
    fig, ax = pyplot.subplots(figsize=(12, 6), dpi=quality_dpi)
    
    # Time steps
    time_steps = list(range(len(energy_prices)))
    
    # Plot energy prices
    ax.plot(time_steps, energy_prices, color='#d62728', linewidth=2, marker='s', markersize=4)
    ax.fill_between(time_steps, energy_prices, alpha=0.3, color='#d62728')
    
    # Labels and title
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Energy Price (ct/kWh)', fontsize=12)
    ax.set_title('Energy Price Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    fig.tight_layout()
    
    # Convert to PIL Image
    fig.canvas.draw()
    image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),
                                fig.canvas.tostring_rgb())
    
    # Show if requested
    if show_plot:
        pyplot.show()
    
    # Cleanup
    pyplot.close(fig)
    
    return image


def plot_energy_per_timestep(tasks: List[Task], machine_info: List, energy_prices: List[float],
                             makespan: int = None, show_plot: bool = False,
                             quality_dpi: int = 100) -> PIL.Image:
    """
    Plot energy consumption per time step as a stacked bar chart, showing contribution of each machine.
    
    :param tasks: List of scheduled tasks
    :param machine_info: List of machine objects with energy consumption info
    :param energy_prices: List of energy prices at each time step
    :param makespan: Total makespan (if None, will be calculated from tasks)
    :param show_plot: Whether to display the plot
    :param quality_dpi: DPI quality of the saved figure
    :return: PIL.Image of the plot
    """
    matplotlib.use('agg')
    
    if makespan is None:
        makespan = max([task.finished for task in tasks if task.done])

    num_machines = len(machine_info)

    # Energy matrix; initially idle consumption everywhere
    energy_matrix = np.array([
        [machine_info[m].standby_energy * 0.01] * makespan # convert kWIM to kWh
        for m in range(num_machines)
    ], dtype=np.float64)  # shape: (M, T)

    for task in tasks:
        if not task.done:
            continue
        m = task.selected_machine
        t_start, t_end = int(task.started), int(task.finished)
        energy_matrix[m, t_start:t_end] = task.machine_running_energy[m] * 0.01 # convert kWIM to kWh

    # Plot
    fig, ax = pyplot.subplots(figsize=(14, 7), dpi=quality_dpi)
    colormap = create_colormap()

    bottom = np.zeros(makespan)
    for m in range(num_machines):
        ax.bar(np.arange(makespan), energy_matrix[m], bottom=bottom,
               label=f'Machine {m}',
               color=colormap(m % colormap.N),
               edgecolor=None, linewidth=0)
        bottom += energy_matrix[m]  # 

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
    ax.set_title('Energy Consumption per Time Step by Machine', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    fig.tight_layout()

    fig.canvas.draw()
    image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),
                                fig.canvas.tostring_rgb())
    if show_plot:
        pyplot.show()
    pyplot.close(fig)
    return image


def plot_cost_per_timestep(tasks: List[Task], machine_info: List, energy_prices: List[float],
                           solar_profile: List[float] = None, makespan: int = None, show_plot: bool = False,
                           quality_dpi: int = 100) -> PIL.Image:
    """
    Plot energy cost per time step as a stacked bar chart, showing contribution of each machine.
    Accounts for solar production - only grid energy is charged.
    
    :param tasks: List of scheduled tasks
    :param machine_info: List of machine objects with energy consumption info
    :param energy_prices: List of energy prices at each time step
    :param solar_profile: List of solar generation values at each time step (in kW)
    :param makespan: Total makespan (if None, will be calculated from tasks)
    :param show_plot: Whether to display the plot
    :param quality_dpi: DPI quality of the saved figure
    :return: PIL.Image of the plot
    """
    matplotlib.use('agg')
    
    if makespan is None:
        makespan = max([task.finished for task in tasks if task.done])

    num_machines = len(machine_info)

    # Energy matrix
    # Shape: (num_machines, makespan); initially idle consumption everywhere
    energy_matrix = np.array([
        [machine_info[m].standby_energy] * makespan
        for m in range(num_machines)
    ], dtype=np.float64)

    for task in tasks:
        if not task.done:
            continue
        m = task.selected_machine
        t_start, t_end = int(task.started), int(task.finished)
        energy_matrix[m, t_start:t_end] = task.machine_running_energy[m]

    # Cost matrix
    solar  = np.asarray(solar_profile, dtype=np.float64)[:makespan]
    prices = np.asarray(energy_prices, dtype=np.float64)[:makespan]


    total_energy = energy_matrix.sum(axis=0)                # shape: (T,)

    # Fraction of grid energy per timestep (0 if no consumption)
    net_grid = np.maximum(0.0, total_energy - solar)        # shape: (T,)
    grid_fraction = np.where(
        total_energy > 0,
        net_grid / total_energy,
        0.0
    )                                                        # shape: (T,)

    # Cost per machine and timestep
    cost_matrix = (
        energy_matrix              # (M, T)
        * grid_fraction[np.newaxis, :]   
        * prices[np.newaxis, :]         
        / 10_000
    )                                                       

    # Plot
    fig, ax = pyplot.subplots(figsize=(14, 7), dpi=quality_dpi)
    colormap  = create_colormap()
    time_steps = np.arange(makespan)

    bottom = np.zeros(makespan)
    for m in range(num_machines):
        ax.bar(time_steps, cost_matrix[m], bottom=bottom,
               label=f'Machine {m}',
               color=colormap(m % colormap.N),
               edgecolor=None, linewidth=0)
        bottom += cost_matrix[m]          

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Energy Cost (€)', fontsize=12)
    ax.set_title('Energy Cost per Time Step by Machine', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    fig.tight_layout()

    fig.canvas.draw()
    image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),
                                fig.canvas.tostring_rgb())
    if show_plot:
        pyplot.show()
    pyplot.close(fig)
    return image


def plot_cumulative_energy(tasks: List[Task], machine_info: List, energy_prices: List[float],
                           makespan: int = None, show_plot: bool = False,
                           quality_dpi: int = 100) -> PIL.Image:
    """
    Plot cumulative energy consumption over time as a stacked bar chart, showing contribution of each machine.
    
    :param tasks: List of scheduled tasks
    :param machine_info: List of machine objects with energy consumption info
    :param energy_prices: List of energy prices at each time step
    :param makespan: Total makespan (if None, will be calculated from tasks)
    :param show_plot: Whether to display the plot
    :param quality_dpi: DPI quality of the saved figure
    :return: PIL.Image of the plot
    """
    matplotlib.use('agg')

    if makespan is None:
        makespan = max([task.finished for task in tasks if task.done])

    num_machines = len(machine_info)

    # Energiematrix
    energy_matrix = np.array([
        [machine_info[m].standby_energy] * makespan
        for m in range(num_machines)
    ], dtype=np.float64)  # shape: (M, T)

    for task in tasks:
        if not task.done:
            continue
        m = task.selected_machine
        t_start, t_end = int(task.started), int(task.finished)
        energy_matrix[m, t_start:t_end] = task.machine_running_energy[m]

    # Kumulative Energie 
    cumulative_energy_matrix = np.cumsum(energy_matrix * 0.01, axis=1)  # shape: (M, T)

    # Plot
    fig, ax = pyplot.subplots(figsize=(14, 7), dpi=quality_dpi)
    colormap = create_colormap()

    bottom = np.zeros(makespan)
    for m in range(num_machines):
        ax.bar(np.arange(makespan), cumulative_energy_matrix[m], bottom=bottom,
               label=f'Machine {m}',
               color=colormap(m % colormap.N),
               edgecolor=None, linewidth=0)
        bottom += cumulative_energy_matrix[m]

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Cumulative Energy (kWh)', fontsize=12)
    ax.set_title('Cumulative Energy Consumption by Machine', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    fig.tight_layout()

    fig.canvas.draw()
    image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),
                                fig.canvas.tostring_rgb())
    if show_plot:
        pyplot.show()
    pyplot.close(fig)
    return image


def plot_cumulative_cost(tasks: List[Task], machine_info: List, energy_prices: List[float],
                        solar_profile: List[float] = None, makespan: int = None, show_plot: bool = False,
                        quality_dpi: int = 100) -> PIL.Image:
    """
    Plot cumulative energy cost over time as a stacked bar chart, showing contribution of each machine.
    Accounts for solar production - only grid energy is charged.
    
    :param tasks: List of scheduled tasks
    :param machine_info: List of machine objects with energy consumption info
    :param energy_prices: List of energy prices at each time step
    :param solar_profile: List of solar generation values at each time step (in kW)
    :param makespan: Total makespan (if None, will be calculated from tasks)
    :param show_plot: Whether to display the plot
    :param quality_dpi: DPI quality of the saved figure
    :return: PIL.Image of the plot
    """

    matplotlib.use('agg')

    if makespan is None:
        makespan = max([task.finished for task in tasks if task.done])

    num_machines = len(machine_info)

    # Energiematrix
    energy_matrix = np.array([
        [machine_info[m].standby_energy] * makespan
        for m in range(num_machines)
    ], dtype=np.float64)  # shape: (M, T)

    for task in tasks:
        if not task.done:
            continue
        m = task.selected_machine
        t_start, t_end = int(task.started), int(task.finished)
        energy_matrix[m, t_start:t_end] = task.machine_running_energy[m]

    # Kostenmatrix
    solar  = np.asarray(solar_profile, dtype=np.float64)[:makespan]  
    prices = np.asarray(energy_prices, dtype=np.float64)[:makespan]

    total_energy  = energy_matrix.sum(axis=0)
    net_grid      = np.maximum(0.0, total_energy - solar)
    grid_fraction = np.where(total_energy > 0, net_grid / total_energy, 0.0)

    cost_matrix = (
        energy_matrix
        * grid_fraction[np.newaxis, :]
        * prices[np.newaxis, :]
        / 10_000
    )  # shape: (M, T)

    # Kumulative Kosten 
    cumulative_cost_matrix = np.cumsum(cost_matrix, axis=1)  # shape: (M, T)

    # Plot
    fig, ax = pyplot.subplots(figsize=(14, 7), dpi=quality_dpi)
    colormap = create_colormap()

    bottom = np.zeros(makespan)
    for m in range(num_machines):
        ax.bar(np.arange(makespan), cumulative_cost_matrix[m], bottom=bottom,
               label=f'Machine {m}',
               color=colormap(m % colormap.N),
               edgecolor=None, linewidth=0)
        bottom += cumulative_cost_matrix[m]

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Cumulative Cost (€)', fontsize=12)
    ax.set_title('Cumulative Energy Cost by Machine', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    fig.tight_layout()

    fig.canvas.draw()
    image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),
                                fig.canvas.tostring_rgb())
    if show_plot:
        pyplot.show()
    pyplot.close(fig)
    return image

def get_pareto_front(
    pareto: List[Individual],
    weighted_best: Optional[Individual] = None,
    quality_dpi: int = 100,
    ) -> "PIL.Image.Image":
    
    fig, ax = pyplot.subplots(figsize=(8, 5), dpi=quality_dpi)

    makespans = [ind.objectives[0] for ind in pareto]
    energies = [ind.objectives[1] for ind in pareto]

    ax.scatter(makespans, energies, c="steelblue", s=60, alpha=0.8, zorder=2,
        label=f"Pareto-Front ({len(pareto)} solutions)")

    sorted_pairs = sorted(zip(makespans, energies), key=lambda x: x[0])
    ax.step([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
        where="post", color="steelblue", linewidth=1.0, alpha=0.4, zorder=1)

    if weighted_best is not None:
        ax.scatter(weighted_best.objectives[0], weighted_best.objectives[1],
            c="crimson", s=120, zorder=3, marker="*",
            label=f"Best (MS={weighted_best.objectives[0]:.1f})")

    ax.set_xlabel("Makespan", fontsize=12)
    ax.set_ylabel("Energy Cost [€]", fontsize=12)
    ax.set_title(f"Pareto-Front", fontsize=14, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3, axis='both')
    fig.tight_layout()

    fig.canvas.draw()
    image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),
                    fig.canvas.tostring_rgb())
    pyplot.close(fig)
    return image