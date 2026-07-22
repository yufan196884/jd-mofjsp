"""
This file provides functions to visualize the current states of a scheduling problems as gantt charts.
Moreover, the generated gantt chart figures can be saved (e.g. as gif).
"""
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.colors as mcolors
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

from src.data_generator.sp_factory import Task
from src.utils.ui_tools.progressbar import progressbar
from matplotlib.patches import Patch

# Constants
VISUALS_DIRECTORY: Path = Path(__file__).parent.parent / 'visuals'

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

class GanttChartPlotter:
    """
    This class provides functions to visualize the current states of a scheduling problems as gantt charts and
    save them as image or gif.
    """
    @classmethod
    def get_gantt_chart_image(cls, tasks: List[Task], show_image: bool = False, return_image: bool = True,
                              quality_dpi: int = 100, overall_makespan: int = None, overall_num_machines: int = None,
                              overall_task_position_list: List[int] = None) -> PIL.Image:
        """
        Can be used to visualize the current state of a scheduling problem as a gantt chart. Note that the figure
        becomes too large with large processing times and numbers of tasks.

        :param tasks: List of tasks (instance) to be visualized
        :param show_image: True, if the generated image is to be visualized
        :param return_image: True if the generated image is to be returned
        :param quality_dpi: dpi of the generated image
        :param overall_makespan: Makespan of the scheduling problem. Can be None
        :param overall_num_machines: Number of machines available in the scheduling problem. Can be None
        :param overall_task_position_list: Task position in original list. Can be None

        :return: Gantt chart image

        """
        pyplot.rcParams['savefig.dpi']=300
        if show_image:
            matplotlib.use('TkAgg')
            _, _ = cls._make_gantt_chart_image(tasks, quality_dpi, overall_makespan, overall_num_machines,
                                               overall_task_position_list)
            pyplot.show()
            pyplot.figure().clear()
            pyplot.clf()
            pyplot.close()
        if return_image:
            matplotlib.use('agg')
            fig, _ = cls._make_gantt_chart_image(tasks, quality_dpi, overall_makespan, overall_num_machines,
                                                 overall_task_position_list)
            fig.canvas.draw()
            image_gantt_chart: PIL.Image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),
                                                               fig.canvas.tostring_rgb())
            # Cleanup
            pyplot.figure().clear()
            pyplot.clf()
            pyplot.close(fig)
            pyplot.close()

            return image_gantt_chart

    @classmethod
    def _make_gantt_chart_image(cls, tasks: List[Task], quality_dpi: int = 100, overall_makespan: int = None,
                                overall_num_machines: int = None,
                                overall_task_position_list: List[int] = None) -> PIL.Image:
        """
        Can be used to visualize the current state of a scheduling problem as a gantt chart

        :param tasks: List of tasks (instance) to be visualized
        :param quality_dpi: dpi of the generated image
        :param overall_makespan: Makespan of the scheduling problem. Can be None
        :param overall_num_machines: Number of machines available in the scheduling problem. Can be None
        :param overall_task_position_list: Task position in original list. Can be None

        :return: Gantt chart image

        """
        # Set important variables
        # Overall makespan of the schedule (+ 1 so that the Gantt Chart x-axis is complete)
        makespan: int = max([task.finished for task in tasks]) + 1 \
            if overall_makespan is None else overall_makespan
        
        # Number of machines available in the schedule
        num_machines: int = max([task.selected_machine for task in tasks if hasattr(task, 'selected_machine')]) + 1 \
            if overall_num_machines is None else overall_num_machines
        
        # Create a figure and axis with a reasonable size
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.set_dpi(quality_dpi)
        
        # Get number of jobs for color mapping
        n_jobs = max([task.job_index for task in tasks]) + 1
        
        # Create a colormap with distinct colors
        colors = plt.cm.get_cmap('tab20', n_jobs)
        
        # Set up the axis
        ax.set_title('Gantt Chart', fontsize=16)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Machine', fontsize=12)
        
        # Set y-axis ticks to machine names
        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
        
        # Set x-axis limits with a bit of padding
        ax.set_xlim([-0.5, makespan + 0.5])
        ax.set_ylim([-0.5, num_machines - 0.5])
        
        # Add grid lines for better readability
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Create a legend for jobs
        legend_handles = []
        
        # Plot tasks as horizontal bars
        for task in tasks:
            # Skip tasks that are not done
            if not task.done:
                continue
            
            # Get task attributes
            job_id = task.job_index
            task_id = task.task_index
            machine = task.selected_machine
            start_time = task.started
            duration = task.finished - task.started
            
            # Get color based on job ID
            color = colors(job_id % n_jobs)
            
            # Plot the task as a horizontal bar
            bar = ax.barh(
                y=machine,
                width=duration,
                left=start_time,
                height=0.8,
                color=color,
                edgecolor='black',
                linewidth=1,
                alpha=0.8
            )
            
            # Add a label inside the bar
            label_text = f'J{job_id}\nT{task_id}'
            ax.text(
                x=start_time + duration/2,
                y=machine,
                s=label_text,
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
            
            # Add this job to the legend (only once per job)
            if job_id not in [h.get_label() for h in legend_handles]:
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f'Job {job_id}'))
        
        # Add the legend
        # Place the legend outside the plot on the right
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
        
        # Add time markers at regular intervals
        time_step = max(1, int(makespan) // 20)  # Adjust for readability
        ax.set_xticks(range(0, int(makespan) + 1, time_step))
        
        # Make layout tight
        fig.tight_layout()
        
        return fig, ax

    @staticmethod
    def save_gantt_chart_image(gantt_chart: PIL.Image.Image, save_path_dir: Path = Path(VISUALS_DIRECTORY),
                               filename: str = "gantt_chart", file_type: str = "png") -> Path:
        """
        Saves the input image

        :param gantt_chart: Gantt chart image
        :param save_path_dir: Relative path where the image is to be saved
        :param filename: Name under the image is to be saved
        :param file_type: Suffix with the image is to be saved

        :return: Path of the saved image

        """
        # Check path existence - create if not exist
        if not save_path_dir.exists():
            save_path_dir.mkdir(parents=True, exist_ok=True)

        full_file_path: Path = save_path_dir / f'{filename}.{file_type}'
        gantt_chart.save(full_file_path)

        return full_file_path

    @classmethod
    def get_gantt_chart_image_and_save(cls, tasks: List[Task], show_image: bool = False, quality_dpi: int = 100,
                                       overall_makespan: int = None, overall_num_machines: int = None,
                                       overall_task_position_list: List[int] = None,
                                       save_path_dir: Path = Path(VISUALS_DIRECTORY),
                                       filename: str = "gantt_chart", file_type: str = "png") -> Path:
        """
        Initializes the creation and saving of a gantt chart image

        :param tasks: List of tasks (instance) to be visualized
        :param show_image: True, if the generated image is to be visualized
        :param quality_dpi: dpi of the generated image
        :param overall_makespan: Makespan of the scheduling problem. Can be None
        :param overall_num_machines: Number of machines available in the scheduling problem. Can be None
        :param overall_task_position_list: Task position in original list. Can be None
        :param save_path_dir: Relative path where the image is to be saved
        :param filename: Name under the image is to be saved
        :param file_type: Suffix with the image is to be saved

        :return: Path of the saved image

        """
        image: PIL.Image.Image = cls.get_gantt_chart_image(tasks=tasks, show_image=show_image, quality_dpi=quality_dpi,
                                                           overall_makespan=overall_makespan,
                                                           overall_num_machines=overall_num_machines,
                                                           overall_task_position_list=overall_task_position_list)

        image_path: Path = cls.save_gantt_chart_image(gantt_chart=image, save_path_dir=save_path_dir, filename=filename,
                                                      file_type=file_type)
        return image_path
