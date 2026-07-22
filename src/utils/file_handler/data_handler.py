"""
This file provides functions to handle the import and export of datasets.

Constants:
DATA_DIRECTORY: Path: Parent directory for the data import
FILE_PATH_VARIABLES: list[str]: Necessary variables to load data by config
"""
# OS imports
import json
from pathlib import Path
import os
import re
# Functional imports
import pickle
import numpy as np
from typing import List, Dict
from src.data_generator.task import Task

# Constants
DATA_DIRECTORY: Path = Path(__file__).parent.parent.parent.parent / 'data' / 'instances'
DATA_BENCHMARK_DIRECTORY: Path = Path(__file__).parent.parent.parent.parent / 'data' / 'benchmark_instances'
SOLVER_DATA_DIRECTORY: Path = Path(__file__).parent.parent.parent.parent / 'data' / 'models' / 'solver_solution'
GA_DIRECTORY: Path = Path(__file__).parent.parent.parent.parent / 'data' / 'models' / 'GA_solution'
FILE_PATH_VARIABLES: List[str] = ['sp_type', 'num_jobs', 'num_tasks', 'num_tools']
INSTANCES_FILE_CONFIG_KEY: str = 'instances_file'


class DataHandler:
    """
    Handles the import and export of datasets.
    """
    @classmethod
    def _data_file_path_from_variables(cls, sp_type: str, num_jobs: int, num_tasks: int, num_tools: int,
                                       **_kwargs) -> Path:
        """
        Creates a name fpr the datafile from config variables

        :param sp_type: Scheduling problem type (e.g. "jssp")
        :param num_jobs: number of jobs generated in an instance
        :param num_tasks: number of tasks per job generated in an instance
        :param num_tools: number of tools available
        :param kwargs: Additional kwargs

        :return: Path to datafile

        """
        return DATA_DIRECTORY / sp_type / f"config_job{num_jobs}_task{num_tasks}_tools{num_tools}.pkl"

    @classmethod
    def reset_instances_data_file(cls, config: dict) -> None:
        """
        Deletes the instances data file according to the config

        :param config: Config with at least FILE_PATH_VARIABLES

        :return: None

        """
        # get path
        if INSTANCES_FILE_CONFIG_KEY in config:
            data_file_path = DATA_DIRECTORY / config.get('sp_type') / config.get(INSTANCES_FILE_CONFIG_KEY)
        elif all(keyword in config for keyword in FILE_PATH_VARIABLES):
            data_file_path = cls._data_file_path_from_variables(**config)
        else:
            assert False, f"You either have to specify the {INSTANCES_FILE_CONFIG_KEY} or all following " \
                          f"parameters {FILE_PATH_VARIABLES} in your config, to write your generated data to a file"
        # delete file if exists
        if data_file_path.exists():
            os.remove(data_file_path)

    @classmethod
    def save_instances_data_file(cls, config: dict, data: List[List[Task]]) -> None:
        """
        Saves instances as file

        :param config: Config with at least FILE_PATH_VARIABLES
        :param data: List of instances

        :return: None

        """
        # get path
        if INSTANCES_FILE_CONFIG_KEY in config:
            data_file_path = DATA_DIRECTORY / config.get('sp_type') / config.get(INSTANCES_FILE_CONFIG_KEY)
        elif all(keyword in config for keyword in FILE_PATH_VARIABLES):
            data_file_path = cls._data_file_path_from_variables(**config)
        else:
            assert False, f"You either have to specify the {INSTANCES_FILE_CONFIG_KEY} or all following " \
                          f"parameters {FILE_PATH_VARIABLES} in your config, to write your generated data to a file"
        # create path and save
        if not data_file_path.exists():
            data_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Load existing data if file exists
        existing_data = []
        if data_file_path.exists():
            with open(data_file_path, 'rb') as handle:
                existing_data = pickle.load(handle)
        
        # Combine existing data with new data
        combined_data = existing_data + data
        
        with open(data_file_path, 'wb') as handle:
            pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load_instances_data_file(cls, config: dict = None,
                                 instances_data_file_path: Path = None) -> List[List[Task]]:
        """
        Load instances by path, config or filedialog

        :param config: Config with at least TASK_FILE_CONFIG_VARIABLE or FILE_PATH_VARIABLES parameters
        :param instances_data_file_path: Relative path to the instance_data_file you want to load
        (e.g. jssp/config_job3_task4_tools0.pkl)

        :return: List of instances

        """

        assert instances_data_file_path or config, 'You either have to specify a path or a config' \
                                                   'to load an instances data file'

        data: List[List[Task]]

        if instances_data_file_path:
            data_file_path = DATA_DIRECTORY / instances_data_file_path
        elif INSTANCES_FILE_CONFIG_KEY in config:
            data_file_path = DATA_DIRECTORY / config[INSTANCES_FILE_CONFIG_KEY]
        else:
            assert False, f"Missing {INSTANCES_FILE_CONFIG_KEY} variable"
        if data_file_path.exists():
            with open(data_file_path, 'rb') as handle:
                data = pickle.load(handle)
        else:
            assert False, f"Missing file at path: {data_file_path}"

        return data

    @classmethod
    def write_solved_data_to_file(cls, instances_file_path: dict, data: List[List[Task]], success: bool) -> None:
        """
        Saves the solutions computed by the solver

        :param instances_file_path: Path to store the solution file
        :param data: List of solved instances to be saved
        :param success: Boolean indicating if solver was successful

        :return: None

        """
        # complete path
        solved_data_path = SOLVER_DATA_DIRECTORY / f"{instances_file_path}"
        if not solved_data_path.exists():
            solved_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a dictionary containing both data and success flag
        save_data = {
            'data': data,
            'success': success
        }
        
        with open(solved_data_path, 'wb') as handle:
            pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_solved_data(cls, load_config):
        """
        Loads the solutions computed by the solver

        :param load_config: Config with at least sp_type, num_jobs, num_tasks, num_tools parameters

        :return: Tuple containing (list of solved instances, success boolean) 
                 if file specified by config parameters exists. Else (None, False)

        """
        solved_data_file = SOLVER_DATA_DIRECTORY / f"{load_config[INSTANCES_FILE_CONFIG_KEY]}"
        data = None
        success = False
        
        if solved_data_file.exists():
            with open(solved_data_file, 'rb') as handle:
                loaded_data = pickle.load(handle)
                if isinstance(loaded_data, dict) and 'data' in loaded_data and 'success' in loaded_data:
                    data = loaded_data['data']
                    success = loaded_data['success']
                else:
                    # For backward compatibility with older files
                    data = loaded_data
                    success = True  # Assume success for old files
                    
        return data, success

    @classmethod
    def load_solved_instance_by_hash(cls, config: dict, instance_hash: int):
        """
        Searches and loads the solved instance according to the hash if exists

        :param config: Config with instance file information
        :param instance_hash: Individual hash of the instance to be loaded

        :return: Tuple of (Instance from the solver, time taken, success boolean) if exists in file, else (None, None, False)
        """
        solved_instance = None
        time_taken = None
        success = False

        if not config.get("Benchmark_Instance", False):
            solver_data_file = SOLVER_DATA_DIRECTORY / f"{config[INSTANCES_FILE_CONFIG_KEY]}"
        else:
            solver_data_file = SOLVER_DATA_DIRECTORY / f"{config[INSTANCES_FILE_CONFIG_KEY]}" / Path("solved_instances.pkl")

        if solver_data_file.exists():
            with open(solver_data_file, 'rb') as handle:
                data_dict = pickle.load(handle)
                
            if instance_hash in data_dict:
                instance_data = data_dict[instance_hash]
                solved_instance = instance_data.get('data')
                time_taken = instance_data.get('time')
                success = instance_data.get('success', False)

        return solved_instance, time_taken, success

    @classmethod
    def write_solved_instance_by_hash(cls, config: dict, solved_instance: List[Task], instance_hash: int, time_taken: float, success: bool = True, **kwargs) -> None:
        """
        Writes the solved instance to file using the hash as key

        :param config: Config with instance file information
        :param solved_instance: Instance solution to be saved
        :param instance_hash: Individual hash of the instance to be saved
        :param time_taken: Time taken to solve the instance
        :param success: Boolean indicating if solver was successful
        :param kwargs: Additional key, value pairs to be stored with the instance (optional)

        :return: None
        """
        if not config.get("Benchmark_Instance", False):
            solver_data_file = SOLVER_DATA_DIRECTORY / f"{config[INSTANCES_FILE_CONFIG_KEY]}"
        else:
            solver_data_file = SOLVER_DATA_DIRECTORY / f"{config[INSTANCES_FILE_CONFIG_KEY]}" / Path("solved_instances.pkl")
        
        # Create directory if it doesn't exist
        if not solver_data_file.parent.exists():
            solver_data_file.parent.mkdir(parents=True, exist_ok=True)
        
        data_dict = {}
        if solver_data_file.exists():
            with open(solver_data_file, 'rb') as handle:
                try:
                    data_dict = pickle.load(handle)
                except:
                    data_dict = {}
        
        # Store instance data, time and success in a single dictionary
        data_dict[instance_hash] = {
            'data': solved_instance,
            'time': time_taken,
            'success': success,
            **kwargs  # Include any additional key-value pairs
        }
        
        # Save the updated dictionary
        with open(solver_data_file, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_GA_instance_by_hash(cls, config: dict, instance_hash: int) -> List[Task]:
        """
        Searches and loads the GA instance according to the hash if exists

        :param instance_hash: Individual hash of the instance to be loaded

        :return: Instance from the solver parse_to_plottable_format function if exists in file, else None

        """
        solved_instance = None
        time_taken = None

        solver_data_dict = config['instances_file'].split('/')[0]+'/'+config['instances_file'].split('/')[1].split('.')[0]+'.pkl'
        solver_data_dict_time = config['instances_file'].split('/')[0]+'/'+config['instances_file'].split('/')[1].split('.')[0]+'_time.pkl'

        # check if ile exist
        if (GA_DIRECTORY / solver_data_dict).is_file():
            with open(file=GA_DIRECTORY / solver_data_dict, mode='rb') as handle:
                data = pickle.load(handle)
                # check if hash exists in dict
            if instance_hash in data.keys():
                # assign solved_instance from dict to output variable
                solved_instance = data[instance_hash]

            with open(file=GA_DIRECTORY / solver_data_dict_time, mode='rb') as handle:
                time = pickle.load(handle)
                # check if hash exists in dict
            if instance_hash in time.keys():
                # assign solved_instance from dict to output variable
                time_taken = time[instance_hash]

        # return instance value of directory according to the hash key
        return solved_instance, time_taken

    @classmethod
    def write_GA_instance_by_hash(cls, config: dict, solved_instance: List[Task], instance_hash: int, time_taken: int) -> None:
        """
        Writes the solved_instance as value to the SOLVER_DATA_DICT file. Instance_hash is used as key

        :param solved_instance: Instance from the solver parse_to_plottable_format function to be saved
        :param instance_hash: Individual hash of the instance to be saved

        :return: None

        """
        # create item with has as key and solution as value
        new_instance = {instance_hash: solved_instance}
        time_instance = {instance_hash: time_taken}

        solver_data_dict = config['instances_file'].split('/')[0]+'/'+config['instances_file'].split('/')[1].split('.')[0]+'.pkl'
        solver_data_dict_time = config['instances_file'].split('/')[0]+'/'+config['instances_file'].split('/')[1].split('.')[0]+'_time.pkl'

        # check if file exist
        if (GA_DIRECTORY / solver_data_dict).is_file():
            with open(file=GA_DIRECTORY / solver_data_dict, mode='rb') as handle:
                data: Dict = pickle.load(handle)
            data.update(new_instance)
            new_data = data
            with open(file=GA_DIRECTORY / solver_data_dict_time, mode='rb') as handle:
                time: Dict = pickle.load(handle)
            time.update(time_instance)
            new_time = time
        else:
            new_data = new_instance
            new_time = time_instance
            # create path if it doesn't exist
            if not (GA_DIRECTORY / config['instances_file'].split('/')[0]).exists():
                (GA_DIRECTORY / config['instances_file'].split('/')[0]).mkdir(parents=True, exist_ok=True)

        # dump new data
        with open(file=GA_DIRECTORY / solver_data_dict, mode='wb') as handle:
            pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(file=GA_DIRECTORY / solver_data_dict_time, mode='wb') as handle:
            pickle.dump(new_time, handle, protocol=pickle.HIGHEST_PROTOCOL)


def SD_to_schlably(dataset_job_length, dataset_op_pt) -> List[List[Task]]:
    """
    Converts the SD dataset to the format used in the Schlably repository

    :param data: SD dataset

    :return: Schlably dataset

    """

    instances = []

    for job_length, op_pt in zip(dataset_job_length, dataset_op_pt):
        tasks = []
        for job_id, length in enumerate(job_length):
            for task_id in range(length):
                machines = np.argwhere(op_pt[task_id] > 0).flatten().tolist()
                task = Task(job_index=job_id, task_index=task_id, machines=machines,
                                  done=False, _n_machines=len(op_pt[task_id]), _n_tools=0,
                                  tools=[], runtime=0, instance_hash=0, deadline=0)
                task.runtimes = op_pt[task_id]
            
                tasks.append(task)
        instances.append(tasks)

    for instance in instances:
        instance_hash = hash(tuple(instance))
        # set hash attributes of each task
        for task in instance:
            task.instance_hash = instance_hash

    return instances


def load_data_from_files(directory):
    """
        load all files within the specified directory
    :param directory: the directory of files
    :return: a list of data (matrix form) in the directory
    """
    if not os.path.exists(directory):
        return [], []

    dataset_job_length = []
    dataset_op_pt = []
    for root, dirs, files in os.walk(directory):
        # sort files by index
        files.sort(key=lambda s: int(re.findall("\d+", s)[0]))
        files.sort(key=lambda s: int(re.findall("\d+", s)[-1]))
        for f in files:
            g = open(os.path.join(root, f), 'r').readlines()
            job_length, op_pt = text_to_matrix(g)
            dataset_job_length.append(job_length)
            dataset_op_pt.append(op_pt)
    return dataset_job_length, dataset_op_pt

def text_to_matrix(text):
    """
            Convert text form of the data into matrix form
    :param text: the standard text form of the instance
    :return:  the matrix form of the instance
            job_length: the number of operations in each job (shape [J])
            op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$
    """
    n_j = int(re.findall(r'\d+\.?\d*', text[0])[0])
    n_m = int(re.findall(r'\d+\.?\d*', text[0])[1])

    job_length = np.zeros(n_j, dtype='int32')
    op_pt = []

    for i in range(n_j):
        content = np.array([int(s) for s in re.findall(r'\d+\.?\d*', text[i + 1])])
        job_length[i] = content[0]

        idx = 1
        for j in range(content[0]):
            op_pt_row = np.zeros(n_m, dtype='int32')
            mch_num = content[idx]
            next_idx = idx + 2 * mch_num + 1
            for k in range(mch_num):
                mch_idx = content[idx + 2 * k + 1]
                pt = content[idx + 2 * k + 2]
                op_pt_row[mch_idx - 1] = pt

            idx = next_idx
            op_pt.append(op_pt_row)

    op_pt = np.array(op_pt)

    return job_length, op_pt

def load_behnke_data_from_files(dataset_dir: str) -> List[List[Task]]:
    """
    Loads Instances from behnke Dataset.

    """
    Instances = []
    for fname in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, fname)
        if not os.path.isfile(path):
            continue
        with open(path, 'r') as f:
            # First line: jobs and machines count
            first = f.readline().strip().split()
            num_jobs, num_machines = map(int, first[:2])

            for job_idx in range(num_jobs):
                parts = list(map(int, f.readline().strip().split()))
                idx = 0
                num_ops = parts[idx]
                idx += 1
                tasks = []
                for op_idx in range(num_ops):
                    k_machines = parts[idx]
                    idx += 1
                    machines = []
                    runtimes = [0] * num_machines
                    for _m in range(k_machines):
                        m_idx = parts[idx]
                        ptime = parts[idx + 1]
                        machines.append(m_idx)
                        runtimes[m_idx] = ptime
                        idx += 2
                    task = Task(job_index=job_idx, task_index=op_idx, machines=machines,runtime=0,_n_machines = num_machines)
                    task.runtimes = runtimes
                    tasks.append(task)
            Instances.append(tasks)

    return Instances
