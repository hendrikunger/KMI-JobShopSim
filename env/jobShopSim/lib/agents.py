from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING
from datetime import timedelta as Timedelta
import logging
# TODO: remove later if not needed
import random
import sys

if TYPE_CHECKING:
    from .sim_env import SimulationEnvironment, System, Job

logging.basicConfig(stream=sys.stdout)
LOGGING_LEVEL_AGENTS = 'DEBUG'
logger_agents = logging.getLogger('agents.agents')
logger_agents.setLevel(LOGGING_LEVEL_AGENTS)

class Agent:
    
    def __init__(
        self,
        assoc_system: 'System',
        agent_type: str,
    ) -> None:
        # basic information
        self._agent_type = agent_type.upper()
        
        # associated system
        self._assoc_system, self._env = assoc_system.register_agent(
                                    agent=self,
                                    agent_type=self._agent_type)
        
        # dispatching signal: no matter if allocation or sequencing
        self._dispatching_signal: bool = False
    
    def __str__(self) -> str:
        return f"Agent(type={self._agent_type}, Assoc_Syst_ID={self._assoc_system.system_id})"
    
    @property
    def assoc_system(self) -> 'System':
        return self._assoc_system
    
    @property
    def agent_type(self) -> str:
        return self._agent_type
    
    @property
    def env(self) -> 'SimulationEnvironment':
        return self._env
    
    @property
    def dispatching_signal(self) -> bool:
        return self._dispatching_signal
    
    def set_dispatching_signal(
        self,
        reset: bool = False,
    ) -> None:
        
        # check flag and determine value
        if not reset:
            # check if already set
            if not self._dispatching_signal:
                self._dispatching_signal = True
            else:
                raise RuntimeError(f"Dispatching signal for >>{self}<< was already set.")
        # reset
        else:
            # check if already not set
            if self._dispatching_signal:
                self._dispatching_signal = False
            else:
                raise RuntimeError(f"Dispatching signal for >>{self}<< was already reset.")
        
        logger_agents.debug(f"Dispatching signal for >>{self}<< was set to >>{self._dispatching_signal}<<.")
    
    def build_feat_vec(self):
        """
        building feature vector for prediction
        has to be implemented in child classes
        """
        raise NotImplementedError(f"No feature vector building method for {self} of type {self.__class__.__name__} defined.")
    
    def reward(self) -> float:
        raise NotImplementedError(f"No reward calculation method for {self} of type {self.__class__.__name__} defined.")

class AllocationAgent(Agent):
    
    def __init__(
        self,
        **kwargs,
    ) -> None:
        # init base class
        super().__init__(agent_type='ALLOC', **kwargs)
        
        # get associated systems
        self._assoc_infstrct_objs = \
            self._assoc_system.lowest_level_subsystems(only_processing_stations=True)
            
        # RL related properties
        self.feat_vec: npt.NDArray[np.float32] | None = None
        
        # execution control
        # flag to indicate that action was obtained from RL backend
        # indicator for internal loop
        #self._RL_decision_done: bool = False
        # indicator for external loop in Gym Env
        #self._RL_decision_request: bool = False
        # [ACTIONS]
        # action chosen by RL agent
        self._action: 'Action' | None = None
        self.action_feasible: bool = False
        """
        # sync state
        self._RL_decision_done_state: State = sim.State(f'sync_{self.name()}_done')
        self._RL_decision_request_state: State = sim.State(f'sync_{self.name()}_request')
        """
        
    @property
    def RL_decision_done(self) -> bool:
        return self._RL_decision_done
    
    @property
    def RL_decision_request(self) -> bool:
        return self._RL_decision_request
    
    @property
    def action(self) -> 'Action':
        return self._action
    
    @property
    def assoc_infstrct_objs(self) -> tuple:
        return self._assoc_infstrct_objs
    
    def update_assoc_infstrct_objs(self) -> None:
        # get associated systems
        self._assoc_infstrct_objs = \
            self._assoc_system.lowest_level_subsystems(only_processing_stations=True)
            
    def request_decision(
        self,
        disposable_job: 'Job',
    ) -> None:
        # for each request, decision not done yet
        # indicator for internal loop
        #self._RL_decision_done = False
        # set flag indicating an request was made
        # indicator for external loop in Gym Env
        #self._RL_decision_request = True
        
        # indicator that request is being made
        self.set_dispatching_signal(reset=False)
        
        # build feature vector
        self.feat_vec = self._build_feat_vec(disposable_job=disposable_job)
        
        logger_agents.debug(f"[REQUEST Agent {self}]: built FeatVec.")
    
    def set_decision(
        self,
        action: 'Action',
    ) -> None:
        # get action from RL agent
        self._action = action
        # decision done
        # indicator for internal loop
        #self._RL_decision_done = True
        # reset request indicator
        # indicator for external loop in Gym Env
        #self._RL_decision_request = False
        
        # indicator that request was processed
        # reset dispatching signal
        self.set_dispatching_signal(reset=True)
        
        logger_agents.debug(f"[DECISION SET Agent {self}]: Set {self._action=}")
    
    # REWORK
    def _build_feat_vec(
        self,
        disposable_job: 'Job',
    ) -> 'FeatureVector':
        
        # resources
        # needed properties
        # station group, availability, WIP_time
        for i, res in enumerate(self._assoc_infstrct_objs):
            # T1 build feature vector for one machine
            monitor = res.stat_monitor
            # station group identifier should be the system's one 
            # because custom IDs can be non-numeric which is bad for an agent
            # use only first identifier although multiple values are possible
            res_SGI: ObjectID = list(res.supersystems_ids)[0]
            # availability: boolean to integer
            avail = int(monitor.is_available)
            # WIP_time in hours
            WIP_time: float = monitor.WIP_load_time / Timedelta(hours=1)
            
            temp1: tuple[ObjectID, int, float] = (res_SGI, avail, WIP_time)
            temp2 = np.array(temp1)
            
            if i == 0:
                arr = temp2
            else:
                arr = np.concatenate((arr, temp2))
        
        # job
        # needed properties
        # order time, target station group ID
        order_time: float = disposable_job.current_order_time / Timedelta(hours=1)
        job_SGI = disposable_job.current_op.target_station_group_identifier
        # SGI is type CustomID, but system ID (ObjectID) is needed
        # lookup system ID by custom ID in Infrastructure Manager
        infstruct_mgr = self.env.infstruct_mgr
        system_id = infstruct_mgr.lookup_system_ID(
            subsystem_type='StationGroup',
            custom_ID=job_SGI,
        )
        temp1: tuple[float, ObjectID] = (order_time, system_id)
        temp2 = np.array(temp1)
        
        # concat job information
        arr = np.concatenate((arr, temp2))
        
        
        #self.feat_vec = arr
        
        # REWORK: simulate agent decision making in Gym Env
        #time.sleep(7)
        #self._RL_decision_done = True
        
        return arr
    
    def random_action(self) -> int:
        """
        Generate random action based on associated objects
        """
        return random.randint(0, len(self._assoc_infstrct_objs)-1)

class SequencingAgent(Agent):
    
    def __init__(
        self,
        **kwargs,
    ) -> None:
        
        super().__init__(agent_type='SEQ', **kwargs)