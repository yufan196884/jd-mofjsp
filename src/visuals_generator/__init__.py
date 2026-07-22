from src.visuals_generator.gantt_chart import GanttChartPlotter

def choose_gantt_chart_plotter(config):
    """
    Returns the GanttChartPlotter class based on the config
    """
    problem_instance = config["sp_type"]

    if problem_instance.lower() in ["greenfjsp"]:
        return GanttChartPlotter