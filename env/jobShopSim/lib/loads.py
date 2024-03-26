from __future__ import annotations
import numpy as np
import numpy.typing as npt
from numpy.random._generator import Generator as NPRandomGenerator
import random
from .utils import DTManager

class RandomJobGenerator:
    
    def __init__(
        self,
        seed: int = 42,
    ) -> None:
        """
        seed: seed value for random number generator
        """
        self._np_rnd_gen: NPRandomGenerator = np.random.default_rng(seed=seed)
        self._dt_mgr: DTManager = DTManager()
        random.seed(seed)
        
    def gen_rnd_JSSP_inst(
        self,
        n_jobs: int,
        n_machines: int,
    ) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:
        """
        Generates random job shop instance with given number of and machines
        - each job on all machines
        - max processing time = 9
        
        Output:
        n_jobs: number of jobs
        n_machines: number of machines
        n_tasks: number of tasks
        mat_ProcTimes: matrix of processing times | shape=(n_jobs,n_machines)
        mat_JobMachID: matrix of machine IDs per job starting by index 1 | shape=(n_jobs,n_machines)
        mat_OpID: matrix of operation IDs starting by index 1 | shape=(n_jobs,n_machines)
        """
        # generate random process time matrix shape=(n_jobs, n_machines)
        mat_ProcTimes = self._np_rnd_gen.integers(1, 10, size=(n_jobs,n_machines), dtype=np.uint16)
        
        # generate randomly shuffled job machine combinations
        # machine IDs from 1 to n_machines
        temp = np.arange(0, n_machines, step=1, dtype=np.uint16)
        temp = np.expand_dims(temp, axis=0)
        # repeat dummy line until number n_jobs is reached
        temp = np.repeat(temp, n_jobs, axis=0)
        # randomly permute the machine indices job-wise
        mat_JobMachID = self._np_rnd_gen.permuted(temp, axis=1)
        
        # generate operation ID matrix
        # not mandatory because operations are registered in the environment's dispatcher
        n_ops = n_jobs * n_machines
        temp2 = np.arange(0, (n_ops), step=1, dtype=np.uint16)
        mat_OpID = temp2.reshape(n_jobs, -1)
        
        return mat_ProcTimes, mat_JobMachID
    
    def gen_rnd_job(
        self,
        n_machines: int,
    ) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:
        """generates random job with machine IDs
        [OUTDATED] Should be replaced by the more generic 'gen_rnd_job_by_ids' method
        which uses any IDs provided as NumPy array

        Parameters
        ----------
        n_machines : int
            _description_

        Returns
        -------
        tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16]]
            _description_
        """
        
        # generate random process time matrix shape=(n_machines)
        mat_ProcTimes = self._np_rnd_gen.integers(1, 10, size=n_machines, dtype=np.uint16)
        
        # generate randomly shuffled job machine combinations
        # machine IDs from 1 to n_machines
        temp = np.arange(0, n_machines, step=1, dtype=np.uint16)
        # randomly permute the machine indices job-wise
        mat_JobMachID = self._np_rnd_gen.permuted(temp)
        
        return mat_ProcTimes, mat_JobMachID
    
    def gen_rnd_job_by_ids(
        self,
        exec_system_ids: Sequence[CustomID],
        target_station_group_ids: dict[CustomID, Sequence[CustomID]] | None = None,
        min_proc_time: int = 1,
        max_proc_time: int = 10,
        gen_setup_times: bool = False,
        min_setup_time: int = 1,
        max_setup_time: int = 10,
        time_unit: str = 'hours',
    ) -> tuple[list[CustomID], list[CustomID] | None, list[Timedelta], list[Timedelta] | None]:
        """Generic function to generate processing times and execution flow of a job object
        """
        n_objects = len(exec_system_ids)
        
        # processing times
        proc_times: list[Timedelta] = list()
        proc_times_time_unit: list[int] = self._np_rnd_gen.integers(
                                            min_proc_time, 
                                            max_proc_time, 
                                            size=n_objects, 
                                            dtype=np.uint16).tolist()
        for time in proc_times_time_unit:
            # build timedelta object
            td = self._dt_mgr.timedelta_from_val(val=time,
                                                    time_unit=time_unit)
            proc_times.append(td)
        
        
        # setup times
        setup_times: list[Timedelta] = list()
        setup_times_time_unit: list[int] | None = None
        if gen_setup_times:
            setup_times_time_unit = self._np_rnd_gen.integers(
                                                min_setup_time, 
                                                max_setup_time, 
                                                size=n_objects, 
                                                dtype=np.uint16).tolist()
            for time in setup_times_time_unit:
                # build timedelta object
                td = self._dt_mgr.timedelta_from_val(val=time,
                                                        time_unit=time_unit)
                # append object
                setup_times.append(td)
        
        # randomly permute the execution systems indices
        job_ex_order = self._np_rnd_gen.permuted(exec_system_ids).tolist()
        
        job_target_station_groups: list[CustomID] | None = None
        if target_station_group_ids is not None:
            job_target_station_groups = list()
            
            for exec_system_id in job_ex_order:
                # multiple candidates: random choice
                candidates = target_station_group_ids[exec_system_id]
                
                if len(candidates) > 1:
                    #candidate = self._np_rnd_gen.choice(candidates)
                    candidate = random.choice(candidates)
                # only one entry
                else:
                    candidate = candidates[0]
                
                job_target_station_groups.append(candidate)
        
        return job_ex_order, job_target_station_groups, proc_times, setup_times
    
    def gen_prio(
        self,
        lowest: int = 1,
        highest: int = 9,
    ) -> int:
        """Generates a single priority score

        Parameters
        ----------
        lowest_prio : int
            lowest available priority
        highest_prio : int
            highest available priority

        Returns
        -------
        int
            randomly chosen priority between lowest and highest value
        """
        return int(self._np_rnd_gen.integers(low=lowest, high=highest))