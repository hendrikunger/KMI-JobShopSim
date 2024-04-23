import numpy as np
import numpy.typing as npt
import simpy
from typing import TypeAlias
from collections import OrderedDict
import logging

# type aliases
SimPyEnv: TypeAlias = simpy.core.Environment
EnvID: TypeAlias = int
JobID: TypeAlias = int
CustomID: TypeAlias = str | int

# forward reference, referenced before assignment
#Job: TypeAlias = 'Job'
#Dispatcher: TypeAlias = 'Dispatcher'

# logging
LOGGING_LEVEL = 'DEBUG'
Logger = logging.getLogger('base')
Logger.setLevel(LOGGING_LEVEL)


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


class SimulationEnvironment(simpy.core.Environment):
    
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        """
        
        """
        super().__init__(*args, **kwargs)
        
        self.id_counter: EnvID = 0
        self.resources: dict[EnvID, object] = dict()
        self._custom_identifiers: set[CustomID] = set()
        self._custom_from_env_ids: dict[EnvID, CustomID] = dict()
        self._custom_to_env_ids: dict[CustomID, EnvID] = dict()
    
    def register_object(
        self,
        object
    ) -> EnvID:
        """
        registers an object in the environment by assigning an unique id and 
        adding the object to the associated resources
        
        object:     env resource
        returns:
            env_id: assigned env ID
        """
        env_id = self.id_counter
        self.resources[env_id] = object
        self.id_counter += 1
        
        return env_id
    
    def register_custom_identifier(
        self,
        env_ID: EnvID,
        custom_identifier: CustomID,
    ) -> bool:
        """
        maps custom identifiers of resources to env_id and vice versa
        """
        self._custom_from_env_ids[env_ID] = custom_identifier
        # check if custom identifier is a duplicate
        if custom_identifier not in self._custom_identifiers:
            self._custom_identifiers.add(custom_identifier)
            self._custom_to_env_ids[custom_identifier] = env_ID
        elif self._custom_to_env_ids[custom_identifier] == env_ID:
            Logger.info("Custom identifier already associated with given environment ID.")
        # ambigious, custom_identifier is a duplicate
        else:
            Logger.warning("The custom identifier is ambigous and was already used before for another environment ID. \
                This object can only be uniquely identified by its environment ID, not by the custom identifier provided.")
        
        return True
    
    def get_custom_from_env_id(
        self,
        env_ID: EnvID,
    ) -> CustomID:
        """
        get custom ID by an object's environment id
        """
        try:
            return self._custom_from_env_ids[env_ID]
        except KeyError as error:
            Logger.error(f"The provided key {error} does not exist!")
            raise
    
    def get_custom_to_env_id(
        self,
        custom_identifier: CustomID,
    ) -> EnvID:
        """
        get environment ID by an object's custom identifier
        """
        try:
            return self._custom_to_env_ids[custom_identifier]
        except KeyError as error:
            Logger.error(f"The provided key {error} does not exist!")
            raise


class Machine(simpy.resources.resource.Resource):
    
    def __init__(
        self,
        env: SimulationEnvironment,
        custom_identifier: CustomID | None = None,
        name: str | None = None,
        num_slots: int = 1,
    ) -> None:
        """
        env:        SimPy Environment in which machine is embedded
        num_slots:  capacity of the machine, if multiple processing 
                    slots available at the same time > 1, default=1
        """
        # intialize base class
        super().__init__(env=env, capacity=num_slots)
        
        ############# custom identifiers only over env_id
        ### associate env_id with custom_id in env
        ### lookup env_id of object in environment and obtain custom_id
        # assert machine information
        self.env = env
        self.env_id = env.register_object(self)
        if custom_identifier is not None:
            ret = env.register_custom_identifier(
                    env_ID=self.env_id, custom_identifier=custom_identifier)
        
        # custom name
        if name is not None:
            self.name = name
        else:
            self.name = f'M_env_{self.env_id}'
        
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


class Dispatcher(object):
    Job: TypeAlias = 'Job'
    Dispatcher: TypeAlias = 'Dispatcher'
    
    def __init__(
        self,
    ) -> None:
        """
        JobWatcher class for given environment (only one watcher for each environment)
        - different functions to monitor all jobs in the environment
        - jobs report back their states to the JobWatcher
        """
        self.disposable_jobs: dict[JobID, Job] = dict()
        self.job_pool: OrderedDict[JobID, Job] = OrderedDict()
    
    def gen_machine_pool_generic(
        self,
        env: SimulationEnvironment,
        mat_JobMachID: npt.NDArray[np.uint16],
    ):
        """
        
        """
        ######### infrastrucutre in dispatcher?
        ### other class better, division of resources and jobs
        ### initialize self data structures in init function
        #self.machine_infrastructure = OrderedDict()
        
        
    
    def gen_job_pool_generic(
        self,
        mat_ProcTimes: npt.NDArray[np.uint16],
        mat_JobMachID: npt.NDArray[np.uint16],
        mat_OpID: npt.NDArray[np.uint16],
    ) -> OrderedDict[JobID, Job]:
        """
        function to build a integrated job pool if generic JxM JSSP instances are used
        mat_ProcTimes: matrix of processing times | shape=(n_jobs,n_machines)
        mat_JobMachID: matrix of machine IDs per job starting by index 1 | shape=(n_jobs,n_machines)
        mat_OpID: matrix of operation IDs starting by index 1 | shape=(n_jobs,n_machines)
        """
            
        for job_id in range(len(mat_ProcTimes)):
            temp1 = mat_ProcTimes[job_id].tolist()
            temp2 = mat_JobMachID[job_id].tolist()
            temp3 = mat_OpID[job_id].tolist()
            JobInst = Job(
                identifier=job_id,
                proc_times=temp1,
                machine_order=temp2,
                operation_identifiers=temp3,
                dispatcher=self,
            )
            ######### ADD TO JOB (bottom-up approach)
            ### jobs add themselves to job pool
            self.job_pool[job_id] = JobInst
            
        return self.job_pool
     
    def get_disposable_jobs(
        self,
        job_set: OrderedDict,
    ) -> tuple[list[JobID], list[Job]]:
        """
        function needs to be reworked, jobs should report back information to a dispatcher instance
        (bottom-up instead of top-down)
        """
        self._disposable_jobs_ID: list[int] = list()
        self._disposable_jobs: list[Job] = list()
        
        for job_id, job in job_set.items():
            if job.is_disposable:
                self._disposable_jobs_ID.append(job_id)
                self._disposable_jobs.append(job)
                
        return self._disposable_jobs_ID, self._disposable_jobs


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
        ########### adding machine instances
        ### perhaps adding machine sets if multiple machines possible (machine groups)
        self.target_machine = machine_identifier

 
class Job(object):
    Job: TypeAlias = 'Job'
    Dispatcher: TypeAlias = 'Dispatcher'
    
    def __init__(
        self,
        identifier: JobID,
        proc_times: list[float],
        machine_order: list[int],
        operation_identifiers: list[int],
        dispatcher: Dispatcher,
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
        self.Dispatcher = dispatcher
        
        
        ### OPERATIONS ##
        self.operations = list()
        
        for idx, op_ID in enumerate(operation_identifiers):
            Op = Operation(
                identifier=op_ID,
                proc_times=proc_times[idx],
                machine_identifier=machine_order[idx],
            )
            self.operations.append(Op)
            
        self.total_num_ops: int = len(self.operations)
        self.num_finished_ops: int = 0
        self.current_op: Operation = self.operations[0]
        
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
        # job's next operation is disposable
        # true for each new job, maybe reworked in future for jobs with
        # a start date later than creation date
        self.is_disposable: bool = True
        self.Dispatcher.disposable_jobs[self.identifier] = self
        # last operation ended --> finished job
        self.is_finished: bool = False
        
        # inter-process time characteristics
        # time of first operation starting point
        self.time_entry: float = 0.
        # time of last operation ending point
        self.time_exit: float = 0.
        
        # current resource location
        self.current_resource: object | None = None # specify type if class definition finished
        
        
    def obtain_current_op(self) -> Operation:
        """
        returns the current operation of the job
        If a job is currently being processed its current operation is 
        not changed until this operation is finished.
        """
        return self.current_op