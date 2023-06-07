import numpy as np
import numpy.typing as npt
import simpy
from typing import TypeAlias
from collections import OrderedDict

SimPyEnv: TypeAlias = simpy.core.Environment


def gen_rnd_JSSP_inst(
    n_jobs: int,
    n_machines: int,
    seed: int = 42,
) -> tuple[int, int, int, npt.NDArray[np.uint16], npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:
    """
    Generates random job shop instance with given number of jobs and machines'
    - each job on all machines
    - max processing time = 9
    
    Output:
        - n_jobs: number of jobs
        - n_machines: number of machines
        - n_tasks: number of tasks
        - mat_ProcTimes: matrix of processing times | shape=(n_jobs,n_machines)
        - mat_JobMachID: matrix of machine IDs per job starting by index 1 | shape=(n_jobs,n_machines)
        - mat_OpID: matrix of operation IDs starting by index 1 | shape=(n_jobs,n_machines)
    """
    # generate random process time matrix shape=(n_jobs, n_machines)
    np_rnd_gen = np.random.default_rng(seed=seed)
    mat_ProcTimes = np_rnd_gen.integers(1, 10, size=(n_jobs,n_machines), dtype=np.uint16)
    
    # generate randomly shuffled job machine combinations
    # machine IDs from 1 to n_machines
    temp = np.arange(0, (n_machines), step=1, dtype=np.uint16)
    temp = np.expand_dims(temp, axis=0)
    # repeat dummy line until number n_jobs is reached
    temp = np.repeat(temp, n_jobs, axis=0)
    # randomly permute the machine indices job-wise
    mat_JobMachID = np_rnd_gen.permuted(temp, axis=1)
    
    # generate operation ID matrix
    n_ops = n_jobs * n_machines
    temp2 = np.arange(0, (n_ops), step=1, dtype=np.uint16)
    mat_OpID = temp2.reshape(n_jobs, -1)
    
    return n_jobs, n_machines, n_ops, mat_ProcTimes, mat_JobMachID, mat_OpID


class Machine(simpy.resources.resource.Resource):
    
    def __init__(
        self,
        env: SimPyEnv,
        identifier: int,
        name: str | None = None,
        num_slots: int = 1,
    ) -> None:
        """
        env:        SimPy Environment in which machine is embedded
        num_slots:  capacity of the machine, if multiple processing 
                    slots available at the same time > 1
        """
        # intialize base class
        super().__init__(env=env, capacity=num_slots)
        
        # assert machine information
        self.identifier = identifier
        # custom name
        if name is not None:
            self.name = name
        else:
            self.name = f'M{identifier}'
        
        # currently processed job
        self.current_job_ID: int | None = None
        self.current_job: Job | None = None
        
        # machine state parameters
        self.is_occupied: bool = False
        self.is_waiting: bool = False
        self.is_blocked: bool = False
        self.is_failed: bool = False
        # maybe for future, curently no working time calendars planned
        self.is_paused: bool = False
        
        # time in state parameters
        self.time_occupied: float = 0.
        self.time_waiting: float = 0.
        self.time_blocked: float = 0.
        self.time_failed: float = 0.
        
        # number of inputs/outputs
        self.num_jobs_input: int = 0
        self.num_jobs_output: int = 0
        

class Operation(object):
    
    def __init__(
        self,
        identifier: int,
        proc_times: float,
        machine_identifier: int,
        name: str | None = None,
    ) -> None:
        """
        identifier:         operation's ID
        proc_times:         operation's processing times
        machine_identifier: ID of machine on which operation is processed
        """
        # !!!!!!!!! perhaps processing times in future multiple entries depending on associated machine
        # change of input format necessary, currently only one machine for each operation
        # no groups, no differing processing times for different machines 

        # assert operation information
        self.identifier = identifier
        # custom name
        if name is not None:
            self.name = name
        else:
            self.name = f'O{identifier}'
            
        # process information
        self.proc_time = proc_times
        self.target_machine = machine_identifier
        
        
class Job(object):
    
    def __init__(
        self,
        identifier: int,
        proc_times: list[float],
        machine_order: list[int],
        operation_identifiers: list[int],
        name: str | None = None,
    ) -> None:
        """
        identifier:             job's ID
        proc_times:             list of processing times for each operation
        machine_order:          list of machine IDs
        operation_identifiers:  list of operation IDs
        """
        # intialize base class
        super().__init__()
        
        ### BASIC INFORMATION ###
        # assert job information
        self.identifier = identifier
        # custom name
        if name is not None:
            self.name = name
        else:
            self.name = f'J{identifier}'
        
        ### OPERATIONS ##
        self.operations = list()
        
        for idx, op_ID in enumerate(operation_identifiers):
            op = Operation(
                identifier=op_ID,
                proc_times=proc_times[idx],
                machine_identifier=machine_order[idx],
            )
            self.operations.append(op)
            
        self.total_num_ops: int = len(self.operations)
        self.num_finished_ops: int = 0
        self.next_op: Operation = self.operations[0]
        
        ### STATE ###
        # intra-process job state parameters
        # job is being processed, maybe better naming in future
        self.is_occupied: bool = False
        # waiting state only when released
        self.is_waiting: bool = False
        # if lying on failed machine
        self.is_failed: bool = False
        
        # intra-process time characteristics
        self.time_occupied: float = 0.
        self.time_waiting: float = 0.
        self.time_failed: float = 0.
        
        # inter-process job state parameters
        # first operation scheduled --> released job
        self.is_released: bool = False
        # last operation ended --> finished job
        self.is_finished: bool = False
        
        # inter-process time characteristics
        # time of first operation starting point
        self.time_entry: float = 0.
        # time of last operation ending point
        self.time_exit: float = 0.
        
        # current resource location
        self.current_resource: object | None = None # specify type if class definition finished
        
        
    def obtain_next_op(self) -> Operation:
        return self.next_op