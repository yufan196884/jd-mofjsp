from src.agents.solver.solver_GreenFJSP import OrToolSolver_GreenFJSP

def get_solver(config):

    problem_instance = config["sp_type"]
    if problem_instance.lower() in ['green_fjsp', 'green_fsp', 'greenfjsp', 'greenfjssp']:
        return OrToolSolver_GreenFJSP
    else:   
        raise NotImplementedError("No solver implemented for this problem instance.")
