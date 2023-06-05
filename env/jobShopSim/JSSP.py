import numpy as np
import numpy.typing as npt


def gen_rnd_JSSP_inst(
    n_jobs: int,
    n_machines: int,
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
    np_rnd_gen = np.random.default_rng(seed=42)
    mat_ProcTimes = np_rnd_gen.integers(1, 10, size=(n_jobs,n_machines), dtype=np.uint16)
    
    # generate randomly shuffled job machine combinations
    # machine IDs from 1 to n_machines
    temp = np.arange(0, (n_machines+0), step=1, dtype=np.uint16)
    temp = np.expand_dims(temp, axis=0)
    # repeat dummy line until number n_jobs is reached
    temp = np.repeat(temp, n_jobs, axis=0)
    # randomly permute the machine indices job-wise
    mat_JobMachID = np_rnd_gen.permuted(temp, axis=1)
    
    # generate operation ID matrix
    n_ops = n_jobs * n_machines
    temp2 = np.arange(1, (n_ops+1), step=1, dtype=np.uint16)
    mat_OpID = temp2.reshape(n_jobs, -1)
    
    return n_jobs, n_machines, n_ops, mat_ProcTimes, mat_JobMachID, mat_OpID