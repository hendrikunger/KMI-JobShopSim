from __future__ import annotations
import salabim as sim
from salabim import Queue, State
from typing import TypeAlias, Self, Any
from collections import OrderedDict, deque
from collections.abc import Iterable, Sequence, Generator
from operator import attrgetter
from functools import lru_cache
from pprint import pprint
import logging
import sys
import random
import datetime
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame, Series
import plotly.express as px
from plotly.graph_objs._figure import Figure
from .utils import flatten, DTParser
from .agents import AllocationAgent, SequencingAgent

# set Salabim to yield mode (using yield is mandatory)
sim.yieldless(False)

# type aliases
SalabimEnv: TypeAlias = sim.Environment
ObjectID: TypeAlias = int
CustomID: TypeAlias = int | str
PlotlyFigure: TypeAlias = Figure

# constants
# infinity
INF: float = float('inf')
# definition of routing system level
EXEC_SYSTEM_TYPE: str = 'ProductionArea'
# time after a store request is failed
FAIL_DELAY: float = 20.

# logging
# IPython compatibility
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
LOGGING_LEVEL = 'ERROR'
LOGGING_LEVEL_ENV = 'INFO'
LOGGING_LEVEL_DISPATCHER = 'DEBUG'
LOGGING_LEVEL_INFSTRCT = 'INFO'
LOGGING_LEVEL_SOURCES = 'ERROR'
LOGGING_LEVEL_SINKS = 'ERROR'
LOGGING_LEVEL_PRODSTATIONS = 'ERROR'
LOGGING_LEVEL_JOBS = 'ERROR'
LOGGING_LEVEL_OPERATIONS = 'ERROR'
LOGGING_LEVEL_BUFFERS = 'ERROR'
LOGGING_LEVEL_MONITORS = 'ERROR'
LOGGING_LEVEL_AGENTS = 'DEBUG'


logger = logging.getLogger('base')
logger.setLevel(LOGGING_LEVEL)
logger_env = logging.getLogger('env')
logger_env.setLevel(LOGGING_LEVEL_ENV)
logger_dispatcher = logging.getLogger('dispatcher')
logger_dispatcher.setLevel(LOGGING_LEVEL_DISPATCHER)
logger_infstrct = logging.getLogger('infstrct')
logger_infstrct.setLevel(LOGGING_LEVEL_INFSTRCT)
logger_sources = logging.getLogger('sources')
logger_sources.setLevel(LOGGING_LEVEL_SOURCES)
logger_sinks = logging.getLogger('sinks')
logger_sinks.setLevel(LOGGING_LEVEL_SINKS)
logger_prodStations = logging.getLogger('prodStations')
logger_prodStations.setLevel(LOGGING_LEVEL_PRODSTATIONS)
logger_buffers = logging.getLogger('buffers')
logger_buffers.setLevel(LOGGING_LEVEL_BUFFERS)
logger_monitors = logging.getLogger('monitors')
logger_monitors.setLevel(LOGGING_LEVEL_MONITORS)
logger_agents = logging.getLogger('agents')
logger_agents.setLevel(LOGGING_LEVEL_AGENTS)

logger_jobs = logging.getLogger('jobs')
logger_jobs.setLevel(LOGGING_LEVEL_JOBS)
logger_operations = logging.getLogger('operations')
logger_operations.setLevel(LOGGING_LEVEL_OPERATIONS)


# utility
def filter_processing_stations(
    infstruct_obj_collection: Iterable[InfrastructureObject]
) -> list[ProcessingStation]:
    """Filters an iterable with InfrastructureObjects for ProcessingStations

    Parameters
    ----------
    infstruct_obj_collection : Iterable[InfrastructureObject]
        collection of InfrastrcutureObjects

    Returns
    -------
    list[ProcessingStation]
        list of ProcessingStations for the given collection
    """
    
    return [x for x in infstruct_obj_collection if isinstance(x, ProcessingStation)]


# environment

class SimulationEnvironment(sim.Environment):
    
    def __init__(
        self,
        time_unit: str = 'seconds',
        starting_datetime: Datetime | None = None,
        **kwargs,
    ) -> None:
        """
        
        """
        # time units
        self.time_unit = time_unit
        # if starting datetime not provided use current time
        if starting_datetime is None:
            starting_datetime = datetime.datetime.now()
        # remove microseconds, such accuracy not needed
        starting_datetime = starting_datetime.replace(microsecond=0)
        self.starting_datetime = starting_datetime
        
        super().__init__(time_unit=self.time_unit, datetime0=self.starting_datetime, **kwargs)
        
        # [RESOURCE] infrastructure manager
        self._infstruct_mgr_registered: bool = False
        self._infstruct_mgr: InfrastructureManager | None = None
        
        # [LOAD] job dispatcher
        self._dispatcher_registered: bool = False
        self._dispatcher: Dispatcher | None = None
        
        # [DECISION FLAGS]
        self._signal_allocation: bool = False
        self._signal_sequencing: bool = False
    
    def t_as_dt(self) -> Datetime:
        """return current simulation time as Datetime object

        Returns
        -------
        Datetime
            simulation time in current time unit as Datetime object
        """
        return self.t_to_datetime(t=self.t())
    
    @property
    def infstruct_mgr(self) -> InfrastructureManager:
        """obtain the current registered Infrastructure Manager instance of the environment"""
        if self._infstruct_mgr is None:
            raise ValueError("No Infrastructure Manager instance registered.")
        else:
            return self._infstruct_mgr
    
    @property
    def dispatcher(self) -> Dispatcher:
        """obtain the current registered Dispatcher instance of the environment"""
        if self._dispatcher is None:
            raise ValueError("No Dipsatcher instance registered.")
        else:
            return self._dispatcher
        
    def register_infrastructure_manager(
        self,
        infstruct_mgr: InfrastructureManager,
    ) -> None:
        """
        Registers a dispatcher instance for the environment. Only one instance per environment is allowed.
        returns: EnvID for the dispatcher instance
        """
        if not self._infstruct_mgr_registered and isinstance(infstruct_mgr, InfrastructureManager):
            self._infstruct_mgr = infstruct_mgr
            self._infstruct_mgr_registered = True
            logger_env.info(f"Successfully registered Infrastructure Manager in Env >>{self.name()}<<")
        elif not isinstance(infstruct_mgr, InfrastructureManager):
            raise TypeError(f"The object must be of type >>InfrastructureManager<< but is type >>{type(infstruct_mgr)}<<")
        else:
            raise AttributeError("There is already a registered Infrastructure Manager instance \
                                 Only one instance per environement is allowed.")
    
    def register_dispatcher(
        self,
        dispatcher: Dispatcher,
    ) -> None:
        """
        Registers a dispatcher instance for the environment. Only one instance per environment is allowed.
        returns: EnvID for the dispatcher instance
        """
        if not self._dispatcher_registered and isinstance(dispatcher, Dispatcher):
            self._dispatcher = dispatcher
            self._dispatcher_registered = True
            logger_env.info(f"Successfully registered Dispatcher in Env >>{self.name()}<<")
        elif not isinstance(dispatcher, Dispatcher):
            raise TypeError(f"The object must be of type >>Dispatcher<< but is type >>{type(dispatcher)}<<")
        else:
            raise AttributeError("There is already a registered Dispatcher instance \
                                 Only one instance per environment is allowed.")
    
    # [DISPATCHING SIGNALS]
    @property
    def signal_allocation(self) -> bool:
        return self._signal_allocation
    
    @property
    def signal_sequencing(self) -> bool:
        return self._signal_sequencing
    
    def set_dispatching_signal(
        self,
        sequencing: bool,
        reset: bool = False,
    ) -> None:
        # obtain current value
        if sequencing:
            signal = self._signal_sequencing
            signal_type = 'SEQ'
        else:
            signal = self._signal_allocation
            signal_type = 'ALLOC'
        
        # check flag and determine value
        if not reset:
            # check if already set
            if signal:
                raise RuntimeError(f"Dispatching type >>{signal_type}<<: Flag for Env {self.name()} was already set.")
            else:
                signal = True
        # reset
        else:
            # check if already not set
            if not signal:
                raise RuntimeError(f"Dispatching type >>{signal_type}<<: Flag for Env {self.name()} was already reset.")
            else:
                signal = False
        
        # set flag
        if sequencing:
            self._signal_sequencing = signal
        else:
            self._signal_allocation = signal
        
        logger_env.debug(f"Dispatching type >>{signal_type}<<: Flag for Env {self.name()} was set to >>{signal}<<.")
    
    def build_alloc_feat_vec(
        self,
        exec_system: System | None = None,
    ) -> 'FeatureVector':
        """
        REWORK
        method to generate allocation feature vectors for a given execution system
        currently neither conceptualised nor implemented
        functionality needed:
            - obtain all relevant subsystems of the execution system
            - define relevant Infrastructure Object properties which are considered features
            - build feature vector
        """
        ...
    
    def build_seq_feat_vec(
        self,
        res_obj: InfrastructureObject,
    ) -> 'FeatureVector':
        raise NotImplementedError("Building sequencing feature vectors is not supported yet.")
    
    def check_feasible_agent_alloc(
        self,
        target_station: ProcessingStation,
        op: Operation,
    ) -> bool:
        """
        method which checks for feasibility of agent allocation decisions
        returning True if feasible, False otherwise
        """
        # check if operation has station group identifier (SGI) (CustomID)
        op_SGI = op.target_station_group_identifier
        
        # no station group assigned, choosen station is automatically feasible
        if op_SGI is None:
            return True
        else:
            # lookup SGIs of the target station's station groups
            target_SGIs = target_station.supersystems_custom_ids
            
        if op_SGI in target_SGIs:
            # operation SGI in associated station group IDs found, 
            # target station is feasible for given operation
            return True
        else:
            return False
    
    def check_integrity(self) -> None:
        """
        method to evaluate if certain criteria for the simulation run are satisfied
        checks for:
        - registered dispatcher (min: 1, max: 1)
        - registered sink (min: 1, max: INF)
        """
        # registration of an Infrastructure Manager
        if not self._infstruct_mgr_registered:
            raise ValueError("No Infrastructure Manager instance registered.")
        # registration of a Dispatcher
        elif not self._dispatcher_registered:
            raise ValueError("No Dispatcher instance registered.")
        # registration of sinks
        elif not self._infstruct_mgr.sink_registered:
            raise ValueError("No Sink instance registered.")
        # check if all subsystems are associated to supersystems
        elif not self._infstruct_mgr.verify_system_association():
            raise AssociationError("Non-associated subsystems detected!")
        
        logger_env.info(f"Integrity check for Environment {self.name()} successful.")
    
    def finalise_sim(self) -> None:
        """
        Function which should be executed at the end of the simulation.
        Can be used for finalising data collection, other related tasks or further processing pipelines
        """
        # infrastructure manager instance
        self._infstruct_mgr.finalise()
        
        # dispatcher instance
        self._dispatcher.finalise()


# environment management

class InfrastructureManager:
    
    def __init__(
        self,
        env: SimulationEnvironment,
        **kwargs,
    ) -> None:
        
        # init base class, even if not available
        super().__init__(**kwargs)
        
        # [COMMON]
        self._env = env
        self._env.register_infrastructure_manager(infstruct_mgr=self)
        # subsystem types
        self._subsystem_types: set[str] = set([
            'ProductionArea',
            'StationGroup',
            'Resource',
        ])
        
        # [PRODUCTION AREAS] database as simple Pandas DataFrame
        self._prod_area_prop: dict[str, type] = {
            'prod_area_id': int,
            'custom_id': object,
            'name': str,
            'prod_area': object,
            'containing_proc_stations': bool,
        }
        self._prod_area_db: DataFrame = pd.DataFrame(columns=list(self._prod_area_prop.keys()))
        self._prod_area_db = self._prod_area_db.astype(self._prod_area_prop)
        self._prod_area_db = self._prod_area_db.set_index('prod_area_id')
        self._prod_area_lookup_props: set[str] = set(['prod_area_id', 'custom_id', 'name'])
        # [PRODUCTION AREAS] identifiers
        self._prod_area_counter: ObjectID = 0
        self._prod_area_custom_identifiers: set[CustomID] = set()
        
        # [STATION GROUPS] database as simple Pandas DataFrame
        self._station_group_prop: dict[str, type] = {
            'station_group_id': int,
            'custom_id': object,
            'name': str,
            'station_group': object,
            'prod_area_id': pd.Int64Dtype(),
            'containing_proc_stations': bool,
        }
        self._station_group_db: DataFrame = pd.DataFrame(columns=list(self._station_group_prop.keys()))
        self._station_group_db = self._station_group_db.astype(self._station_group_prop)
        self._station_group_db = self._station_group_db.set_index('station_group_id')
        self._station_group_lookup_props: set[str] = set(['station_group_id', 'custom_id', 'name'])
        # [STATION GROUPS] identifiers
        self._station_group_counter: ObjectID = 0
        self._station_groups_custom_identifiers: set[CustomID] = set()
        
        # [RESOURCES] database as simple Pandas DataFrame
        self._infstruct_prop: dict[str, type] = {
            'res_id': int,
            'custom_id': object,
            'resource': object,
            'name': str,
            'res_type': str,
            'state': str,
            'station_group_id': pd.Int64Dtype(),
        }
        self._res_db: DataFrame = pd.DataFrame(columns=list(self._infstruct_prop.keys()))
        self._res_db = self._res_db.astype(self._infstruct_prop)
        self._res_db = self._res_db.set_index('res_id')
        self._res_lookup_props: set[str] = set(['res_id', 'custom_id', 'name'])
        # [RESOURCES] custom identifiers
        self._res_counter: ObjectID = 0
        self._res_custom_identifiers: set[CustomID] = set()
        # [RESOURCES] sink: pool of sinks possible to allow multiple sinks in one environment
        # [PERHAPS CHANGED LATER] 
        # currently only one sink out of the pool is chosen because jobs do not contain 
        # information about a target sink
        self._sink_registered: bool = False
        self._sinks: list[Sink] = list()
        
        # counter for processing stations (machines, assembly, etc.)
        self.num_proc_stations: int = 0
        
    @property
    def env(self) -> SimulationEnvironment:
        return self._env
    
    # [PRODUCTION AREAS]
    @property
    def prod_area_db(self) -> DataFrame:
        return self._prod_area_db
    
    # [STATION GROUPS]
    @property
    def station_group_db(self) -> DataFrame:
        return self._station_group_db
    
    def verify_system_association(self) -> bool:
        """checks if there are any registered, but non-associated subsystems for each subsystem type

        Returns
        -------
        bool
            indicator if all systems are associated (True) or not (False)
        """
        # check all subsystem types with reference to supersystems if there are
        # any open references (NA values as secondary key)
        relevant_subsystems = ('StationGroup', 'Resource')
        
        for subsystem_type in relevant_subsystems:
            match subsystem_type:
                case 'StationGroup':
                    target_db = self._station_group_db
                    secondary_key: str = 'prod_area_id'
                case 'Resource':
                    target_db = self._res_db
                    secondary_key: str = 'station_group_id'
            # check if there are any NA values as secondary key
            check_val: bool = target_db[secondary_key].isna().any()
            if check_val:
                # there are NA values
                logger_infstrct.error(f"There are non-associated systems for system type >>{subsystem_type}<<. \
                    Please check these systems and add them to a corresponding supersystem.")
                return False
        
        return True
    
    ####################################################################################
    ## REWORK TO WORK WITH DIFFERENT SUBSYSTEMS
    # only one register method by analogy with 'lookup_subsystem_info'
    # currently checking for existence and registration implemented, split into different methods
    # one to check whether such a subsystem already exists
    # another one registers a new subsystem
    # if check positive: return subsystem by 'lookup_subsystem_info'
    ### REWORK TO MULTIPLE SUBSYSTEMS
    def _obtain_system_id(
        self,
        subsystem_type: str,
    ) -> ObjectID:
        """Simple counter function for managing system IDs

        Returns
        -------
        ObjectID
            unique system ID
        """
        if subsystem_type not in self._subsystem_types:
            raise ValueError(f"The subsystem type >>{subsystem_type}<< is not allowed. Choose from {self._subsystem_types}")
        
        match subsystem_type:
            case 'ProductionArea':
                system_id = self._prod_area_counter
                self._prod_area_counter += 1
            case 'StationGroup':
                system_id = self._station_group_counter
                self._station_group_counter += 1
            case 'Resource':
                system_id = self._res_counter
                self._res_counter += 1
        
        return system_id
    
    def register_subsystem(
        self,
        subsystem_type: str,
        obj: System,
        custom_identifier: CustomID,
        name: str | None,
        state: str | None = None,
    ) ->  tuple[ObjectID, str]:
        """
        registers an infrastructure object in the environment by assigning an unique id and 
        adding the object to the associated resources of the environment
        
        obj: env resource = instance of a subclass of InfrastructureObject
        custom_identifier: user defined identifier
        name: custom name of the object, \
            default: None
        returns:
            ObjectID: assigned resource ID
            str: assigned resource's name
        """
        if subsystem_type not in self._subsystem_types:
            raise ValueError(f"The subsystem type >>{subsystem_type}<< is not allowed. Choose from {self._subsystem_types}")
        
        match subsystem_type:
            case 'ProductionArea':
                custom_identifiers = self._prod_area_custom_identifiers
            case 'StationGroup':
                custom_identifiers = self._station_groups_custom_identifiers
            case 'Resource':
                custom_identifiers = self._res_custom_identifiers
        
        # check for uniqueness of custom_identifier
        # type security
        if not isinstance(custom_identifier, (str, int)):
            raise TypeError("Custom identifier must be of type STR or INT")
        # create check value
        if isinstance(custom_identifier, str):
            # remove capital letters for checking
            check_val = custom_identifier.lower()
        else:
            check_val = custom_identifier
        
        # check if value already exists
        if check_val in custom_identifiers:
            raise ValueError(f"The custom identifier {custom_identifier} provided for subsystem type {subsystem_type} \
                already exists, but has to be unique.")
        else:
            custom_identifiers.add(check_val)
        
        # obtain system ID
        system_id = self._obtain_system_id(subsystem_type=subsystem_type)
        
        # [RESOURCES] resource related data
        # register sinks
        if isinstance(obj, Sink):
            if not self._sink_registered:
                self._sink_registered = True
            self._sinks.append(obj)
        # count number of machines
        if isinstance(obj, ProcessingStation):
            self.num_proc_stations += 1
        
        # custom name
        if name is None:
            name = f'{type(obj).__name__}_env_{system_id}'
        
        # new entry for corresponding database
        match subsystem_type:
            case 'ProductionArea':
                new_entry: DataFrame = pd.DataFrame({
                                        'prod_area_id': [system_id],
                                        'custom_id': [custom_identifier],
                                        'name': [name],
                                        'prod_area': [obj],
                                        'containing_proc_stations': [obj.containing_proc_stations]})
                new_entry = new_entry.astype(self._prod_area_prop)
                new_entry = new_entry.set_index('prod_area_id')
                self._prod_area_db = pd.concat([self._prod_area_db, new_entry])
            case 'StationGroup':
                new_entry: DataFrame = pd.DataFrame({
                                        'station_group_id': [system_id],
                                        'custom_id': [custom_identifier],
                                        'name': [name],
                                        'station_group': [obj],
                                        'prod_area_id': [None],
                                        'containing_proc_stations': [obj.containing_proc_stations]})
                new_entry = new_entry.astype(self._station_group_prop)
                new_entry = new_entry.set_index('station_group_id')
                self._station_group_db = pd.concat([self._station_group_db, new_entry])
            case 'Resource':
                new_entry: DataFrame = pd.DataFrame({
                                        'res_id': [system_id],
                                        'custom_id': [custom_identifier],
                                        'resource': [obj],
                                        'name': [name],
                                        'res_type': [obj.res_type],
                                        'state': [state],
                                        'station_group_id': [None]})
                new_entry = new_entry.astype(self._infstruct_prop)
                new_entry = new_entry.set_index('res_id')
                self._res_db = pd.concat([self._res_db, new_entry])
        
        logger_infstrct.info(f"Successfully registered object with SystemID {system_id} and name {name}")
        
        return system_id, name
    
    def register_system_association(
        self,
        supersystem: System,
        subsystem: System,
    ) -> None:
        """associate two system types with each other in the corresponding databases

        Parameters
        ----------
        supersystem : System
            system to which the subsystem is added
        subsystem : System
            system which is added to the supersystem and to whose database the entry is made
        """
        # target subsystem type -> identify appropriate database
        subsystem_type = subsystem.subsystem_type
        
        match subsystem_type:
            case 'StationGroup':
                target_db = self._station_group_db
                target_property: str = 'prod_area_id'
            case 'Resource':
                target_db = self._res_db
                target_property: str = 'station_group_id'
        # system IDs
        supersystem_id = supersystem.system_id
        subsystem_id = subsystem.system_id
        # write supersystem ID to subsystem database entry
        target_db.at[subsystem_id, target_property] = supersystem_id
    
    def set_contain_proc_station(
        self,
        system: System,
    ) -> None:
        
        match system.subsystem_type:
            case 'ProductionArea':
                lookup_db = self._prod_area_db
            case 'StationGroup':
                lookup_db = self._station_group_db

        lookup_db.at[system.system_id, 'containing_proc_stations'] = True
        system.containing_proc_stations = True
        
        # iterate over supersystems
        for supersystem in system.supersystems.values():
            if not supersystem.containing_proc_stations:
                self.set_contain_proc_station(system=supersystem)
    
    def lookup_subsystem_info(
        self,
        subsystem_type: str,
        lookup_val: CustomID,
        lookup_property: str | None = None,
        target_property: str | None = None,
    ) -> tuple[Any, Series | None]:
        """
        obtain a subsystem by its property and corresponding value
        properties: Subsystem ID, Custom ID, Name
        """
        if subsystem_type not in self._subsystem_types:
            raise ValueError(f"The subsystem type >>{subsystem_type}<< is not allowed. Choose from {self._subsystem_types}")
        
        match subsystem_type:
            case 'ProductionArea':
                allowed_lookup_props = self._prod_area_lookup_props
                lookup_db = self._prod_area_db
                if target_property is None:
                    target_property = 'prod_area'
                id_prop: str = 'prod_area_id'
            case 'StationGroup':
                allowed_lookup_props = self._station_group_lookup_props
                lookup_db = self._station_group_db
                if target_property is None:
                    target_property = 'station_group'
                id_prop: str = 'station_group_id'
            case 'Resource':
                allowed_lookup_props = self._res_lookup_props
                lookup_db = self._res_db
                if target_property is None:
                    target_property = 'resource'
                id_prop: str = 'res_id'
        
        # if no lookup property provided use ID
        if lookup_property is None:
            lookup_property = id_prop
        
        # allowed target properties
        allowed_target_props: set[str] = set(lookup_db.columns.to_list())
        # lookup property can not be part of the target properties
        if lookup_property in allowed_target_props:
            allowed_target_props.remove(lookup_property)
        
        # check if property is a filter criterion
        if lookup_property not in allowed_lookup_props:
            raise IndexError(f"Lookup Property '{lookup_property}' is not allowed for subsystem type {subsystem_type}. Choose from {allowed_lookup_props}")
        # check if target property is allowed
        if target_property not in allowed_target_props:
            raise IndexError(f"Target Property >>{target_property}<< is not allowed for subsystem type {subsystem_type}. Choose from {allowed_target_props}")
        # None type value can not be looked for
        if lookup_val is None:
            raise TypeError("The lookup value can not be of type >>None<<.")
        
        # filter resource database for prop-value pair
        if lookup_property == id_prop:
            # direct indexing for ID property: always unique, no need for duplicate check
            try:
                temp1: Any = lookup_db.at[lookup_val, target_property]
                return temp1
            except KeyError:
                raise IndexError(f"There were no subsystems found for the lookup property >>{lookup_property}<< \
                                with the value >>{lookup_val}<<")
        else:
            try:
                temp1: Series = lookup_db.loc[lookup_db[lookup_property] == lookup_val, target_property]
                # check for empty search result, at least one result necessary
                if len(temp1) == 0:
                    raise IndexError(f"There were no subsystems found for the lookup property >>{lookup_property}<< \
                                    with the value >>{lookup_val}<<")
            except KeyError:
                raise IndexError(f"There were no subsystems found for the lookup property >>{lookup_property}<< \
                                with the value >>{lookup_val}<<")
            # check for multiple entries with same prop-value pair
            ########### PERHAPS CHANGE NECESSARY
            ### multiple entries but only one returned --> prone to errors
            if len(temp1) > 1:
                # warn user
                logger_infstrct.warning(f"CAUTION: There are multiple subsystems which share the \
                            same value >>{lookup_val}<< for the lookup property >>{lookup_property}<<. \
                            Only the first entry is returned.")
        
            return temp1.iat[0]
    
    def lookup_custom_ID(
        self,
        subsystem_type: str,
        system_ID: ObjectID,
    ) -> CustomID:
        
        match subsystem_type:
            case 'ProductionArea':
                id_prop: str = 'prod_area_id'
            case 'StationGroup':
                id_prop: str = 'station_group_id'
            case 'Resource':
                id_prop: str = 'res_id'
        
        custom_id = self.lookup_subsystem_info(
            subsystem_type=subsystem_type,
            lookup_val=system_ID,
            lookup_property=id_prop,
            target_property='custom_id',
        )
        
        return custom_id
    
    def lookup_system_ID(
        self,
        subsystem_type: str,
        custom_ID: CustomID,
    ) -> ObjectID:
        
        system = self.lookup_subsystem_info(
            subsystem_type=subsystem_type,
            lookup_val=custom_ID,
            lookup_property='custom_id',
        )
        
        return system.system_id
    
    ####################################################################
    
    # [RESOURCES]
    @property
    def res_db(self) -> DataFrame:
        """obtain a current overview of registered objects in the environment"""
        return self._res_db
    
    @property
    def sinks(self) -> set[Sink]:
        """registered sinks"""
        return self._sinks
    
    @property
    def sink_registered(self) -> bool:
        return self._sink_registered
    
    def update_res_state(
        self,
        obj: InfrastructureObject,
        state: str,
        reset_temp: bool = False,
    ) -> None:
        """method to update the state of a resource object in the resource database"""
        logger_infstrct.debug(f"Set state of {obj} to {state}")
        
        # check if 'TEMP' state should be reset
        if reset_temp:
            # special reset method, calls state setting to previous state
            obj.stat_monitor.reset_temp_state()
            state = obj.stat_monitor.state_current
        else:
            obj.stat_monitor.set_state(state=state)
        
        self._res_db.at[obj.system_id, 'state'] = state
        logger_infstrct.debug(f"Executed state setting of {obj} to {state}")
    
    def res_objs_temp_state(
        self,
        res_objs: Iterable[InfrastructureObject],
        reset_temp: bool,
    ) -> None:
        """Sets/resets given resource objects from the 'TEMP' state

        Parameters
        ----------
        res_objs : Iterable[InfrastructureObject]
            objects for which the TEMP state should be changed
        set_temp : bool
            indicates if the temp state should be set or reset
        """
        for obj in res_objs:
            self.update_res_state(obj=obj, state='TEMP', reset_temp=reset_temp)
            # calculate KPIs if 'TEMP' state is set
            if not reset_temp:
                obj.stat_monitor.calc_KPI()
    
    def finalise(self) -> None:
        
        # set end state for each resource object to calculate the right time amounts
        for res_obj in self._res_db['resource']:
            res_obj.finalise()
        logger_infstrct.info("Successful finalisation of the state information for all resource objects.")

class Dispatcher:
    
    def __init__(
        self,
        env: SimulationEnvironment,
        priority_rule: str = 'FIFO',
        allocation_rule: str = 'RANDOM',
    ) -> None:
        """
        Dispatcher class for given environment (only one dispatcher for each environment)
        - different functions to monitor all jobs in the environment
        - jobs report back their states to the dispatcher
        """
        
        # job data base as simple Pandas DataFrame
        # column data types
        self._job_prop: dict[str, type] = {
            'job_id': int,
            'custom_id': object,
            'job': object,
            'name': str,
            'job_type': str,
            'prio': object,
            #'total_proc_time': float,
            #'creation_date': float,
            #'release_date': float,
            #'entry_date': float,
            #'exit_date': float
            #'lead_time': float,
            'total_proc_time': object,
            'creation_date': object,
            'release_date': object,
            'planned_starting_date': object,
            #'entry_date': object,
            'actual_starting_date': object,
            'starting_date_deviation': object,
            #'exit_date': object,
            'planned_ending_date': object,
            'actual_ending_date': object,
            'ending_date_deviation': object,
            'lead_time': object,
            'state': str,
        }
        self._job_db: DataFrame = pd.DataFrame(columns=list(self._job_prop.keys()))
        self._job_db: DataFrame = self._job_db.astype(self._job_prop)
        self._job_db: DataFrame = self._job_db.set_index('job_id')
        # properties by which a object can be obtained from the job database
        self._job_lookup_props: set[str] = set(['job_id', 'custom_id', 'name'])
        # properties which can be updated after creation
        self._job_update_props: set[str] = set([
            'prio',
            'creation_date',
            'release_date',
            #'entry_date',
            'planned_starting_date',
            'actual_starting_date',
            'starting_date_deviation',
            #'exit_date',
            'planned_ending_date',
            'actual_ending_date',
            'ending_date_deviation',
            'lead_time',
            'state',
        ])
        
        # operation data base as simple Pandas DataFrame
        # column data types
        self._op_prop: dict[str, type] = {
            'op_id': int,
            'job_id': int,
            'job_name': str,
            'custom_id': object,
            'op': object,
            'name': str,
            'prio': object,
            'execution_system': object,
            'execution_system_custom_id': object,
            'execution_system_name': str,
            'execution_system_type': str,
            'target_station_custom_id': object,
            'target_station_name': str,
            'proc_time': object,
            'setup_time': object,
            'order_time': object,
            'creation_date': object,
            'release_date': object,
            #'entry_date': object,
            'planned_starting_date': object,
            'actual_starting_date': object,
            'starting_date_deviation': object,
            #'exit_date': object,
            'planned_ending_date': object,
            'actual_ending_date': object,
            'ending_date_deviation': object,
            'lead_time': object,
            'state': str,
        }
        self._op_db: DataFrame = pd.DataFrame(columns=list(self._op_prop.keys()))
        self._op_db: DataFrame = self._op_db.astype(self._op_prop)
        self._op_db: DataFrame = self._op_db.set_index('op_id')
        # properties by which a object can be obtained from the operation database
        self._op_lookup_props: set[str] = set(['op_id', 'job_id', 'custom_id', 'name', 'machine'])
        # properties which can be updated after creation
        self._op_update_props: set[str] = set([
            'prio',
            'target_station_custom_id',
            'target_station_name',
            'creation_date',
            'release_date',
            #'entry_date',
            #'planned_starting_date',
            'actual_starting_date',
            'starting_date_deviation',
            #'exit_date',
            #'planned_ending_date',
            'actual_ending_date',
            'ending_date_deviation',
            'lead_time',
            'state',
        ])
        
        # register in environment and get EnvID
        self._env = env
        self._env.register_dispatcher(self)
        
        ####################################
        # managing IDs
        self._id_types = set(['job', 'op'])
        self._job_id_counter: ObjectID = 0
        self._op_id_counter: ObjectID = 0
        
        # priority rules
        self._priority_rules: set[str] = set([
            'FIFO',
            'LIFO',
            'SPT',
            'LPT',
            'SST',
            'LST',
            'PRIO',
        ])
        # set current priority rule
        if priority_rule not in self._priority_rules:
            raise ValueError(f"Priority rule {priority_rule} unknown. Must be one of {self._priority_rules}")
        else:
            self._curr_prio_rule = priority_rule
            
        # allocation rule
        self._allocation_rules: set[str] = set([
            'RANDOM',
            'UTILISATION',
            'WIP_LOAD_TIME',
            'WIP_LOAD_JOBS',
            'AGENT',
        ])
        # set current allocation rule
        if allocation_rule not in self._allocation_rules:
            raise ValueError(f"Allocation rule {allocation_rule} unknown. Must be one of {self._allocation_rules}")
        else:
            self._curr_alloc_rule = allocation_rule
            
        # [STATS] cycle time
        self._cycle_time: Timedelta = Timedelta()
        
        self.np_rnd_gen: NPRandomGenerator = np.random.default_rng(seed=42)
    
    ### DATA MANAGEMENT
    def __repr__(self) -> str:
        return f"Dispatcher(env: {self.env.name()})"
    
    @property
    def env(self) -> SimulationEnvironment:
        return self._env
    
    @property
    def curr_prio_rule(self) -> str:
        return self._curr_prio_rule
    
    @curr_prio_rule.setter
    def curr_prio_rule(
        self,
        rule: str,
    ) -> None:
        if rule not in self._priority_rules:
            raise ValueError(f"Priority rule {rule} unknown. Must be one of {self._priority_rules}")
        else:
            self._curr_prio_rule = rule
            logger_dispatcher.info(f"Changed priority rule to {rule}")
    
    def possible_prio_rules(self) -> set[str]:
        return self._priority_rules
    
    @property
    def curr_alloc_rule(self) -> str:
        return self._curr_alloc_rule
    
    @curr_alloc_rule.setter
    def curr_alloc_rule(
        self,
        rule: str,
    ) -> None:
        if rule not in self._allocation_rules:
            raise ValueError(f"Allocation rule {rule} unknown. Must be one of {self._allocation_rules}")
        else:
            self._curr_alloc_rule = rule
            logger_dispatcher.info(f"Changed allocation rule to {rule}")
            
    def possible_alloc_rules(self) -> set[str]:
        return self._allocation_rules
    
    def _obtain_load_obj_id(
        self,
        load_type: str,
    ) -> ObjectID:
        """Simple counter function for managing operation IDs"""
        # assign id and set counter up
        
        if load_type not in self._id_types:
            raise ValueError(f"Given type {type} not valid. Choose from '{self._id_types}'")
        
        match load_type:
            case 'job':
                ident_no = self._job_id_counter
                self._job_id_counter += 1
            case 'op':
                ident_no = self._op_id_counter
                self._op_id_counter += 1
        
        return ident_no
    
    @property
    def cycle_time(self) -> Timedelta:
        return self._cycle_time
    
    def _calc_cycle_time(self) -> None:
        """
        Obtaining the current cycle time of all operations
        """
        self._cycle_time: Timedelta = self._op_db['actual_ending_date'].max() - self._env.starting_datetime
    
    ### JOBS ###
    def register_job(
        self,
        obj: Job,
        custom_identifier: CustomID | None,
        name: str | None,
        state: str,
    ) -> tuple[SimulationEnvironment, ObjectID, str]:
        """
        registers an job object in the dispatcher instance by assigning an unique id and 
        adding the object to the associated jobs
        """
        # obtain id
        job_id = self._obtain_load_obj_id(load_type='job')
        
        # custom name
        if name is None:
            name = f'J_gen_{job_id}'
        
        # time of creation
        #creation_date = self.env.now()
        creation_date = self.env.t_as_dt()
        
        # new entry for job data base
        new_entry: DataFrame = pd.DataFrame({
                                'job_id': [job_id],
                                'custom_id': [custom_identifier],
                                'job': [obj],
                                'name': [name],
                                'job_type': [obj.job_type],
                                'prio': [obj.prio],
                                'total_proc_time': [obj.total_proc_time],
                                'creation_date': [creation_date],
                                'release_date': [obj.time_release],
                                #'entry_date': [obj.time_entry],
                                'planned_starting_date': [obj.time_planned_starting],
                                'actual_starting_date': [obj.time_actual_starting],
                                'starting_date_deviation': [obj.starting_date_deviation],
                                #'exit_date': [obj.time_exit],
                                'planned_ending_date': [obj.time_planned_ending],
                                'actual_ending_date': [obj.time_actual_ending],
                                'ending_date_deviation': [obj.ending_date_deviation],
                                'lead_time': [obj.lead_time],
                                'state': [state]})
        new_entry = new_entry.astype(self._job_prop)
        new_entry = new_entry.set_index('job_id')
        self._job_db = pd.concat([self._job_db, new_entry])
        
        logger_dispatcher.info(f"Successfully registered job with JobID {job_id} and name {name}")
        
        # write job information directly
        obj.time_creation = creation_date
        
        # return current env, job ID, job name
        return self._env, job_id, name
    
    def update_job_db(
        self,
        job: Job,
        property: str,
        val: float | str,
    ) -> None:
        """
        updates the information of a job for a given property
        """
        # check if property is a filter criterion
        if property not in self._job_update_props:
            raise IndexError(f"Property '{property}' is not allowed. Choose from {self._job_update_props}")
        # None type value can not be set
        if val is None:
            raise TypeError("The set value can not be of type >>None<<.")
        
        self._job_db.at[job.job_id, property] = val
    
    def release_job(
        self,
        job: Job,
    ) -> None:
        """
        used to signal the release of the given job
        necessary for time statistics
        """
        #current_time = self.env.now()
        current_time = self.env.t_as_dt()
        job.time_release = current_time
        job.is_released = True
        self.update_job_db(job=job, property='release_date', val=job.time_release)
    
    def enter_job(
        self,
        job: Job,
    ) -> None:
        """
        used to signal the start of the given job on the first Processing Station
        necessary for time statistics
        """
        #current_time = self.env.now()
        current_time = self.env.t_as_dt()
        # starting time processing
        job.time_actual_starting = current_time
        
        # starting times
        if job.time_planned_starting is not None:
            job.starting_date_deviation = job.time_actual_starting - job.time_planned_starting
            self.update_job_db(job=job, property='starting_date_deviation', val=job.starting_date_deviation)
        
        # update operation database
        self.update_job_db(job=job, property='actual_starting_date', val=job.time_actual_starting)
    
    def finish_job(
        self,
        job: Job,
    ) -> None:
        """
        used to signal the exit of the given job
        necessary for time statistics
        """
        # [STATS]
        #current_time = self.env.now()
        current_time = self.env.t_as_dt()
        #job.time_exit = current_time
        job.time_actual_ending = current_time
        job.is_finished = True
        #job.lead_time = job.time_exit - job.time_release
        job.lead_time = job.time_actual_ending - job.time_release
        
        # ending times
        if job.time_planned_ending is not None:
            job.ending_date_deviation = job.time_actual_ending - job.time_planned_ending
            self.update_job_db(job=job, property='ending_date_deviation', val=job.ending_date_deviation)
        
        # update databases
        self.update_job_state(job=job, state='FINISH')
        #self.update_job_db(job=job, property='exit_date', val=job.time_exit)
        self.update_job_db(job=job, property='actual_ending_date', val=job.time_actual_ending)
        self.update_job_db(job=job, property='lead_time', val=job.lead_time)
        # [MONITOR] finalise stats
        job.stat_monitor.finalise_stats()
    
    def update_job_process_info(
        self,
        job: Job,
        preprocess: bool,
    ) -> None:
        """
        method to write necessary information of the job and its current operation before and after processing,
        invoked by Infrastructure Objects
        """
        # get current operation of the job instance
        current_op = job.current_op
        # before processing
        if preprocess:
            # operation enters Processing Station
            #self.release_operation(op=current_op)
            
            # if first operation if given job add job's starting information
            if job.num_finished_ops == 0:
                self.enter_job(job=job)
            
            self.enter_operation(op=current_op)
            ############# ENTRY OF JOB
            #current_op.start_time = self.env.now()
        # after processing
        else:
            # finalise current op
            #logger_dispatcher.debug(f"OP {current_op} is finalised")
            self.finish_operation(op=current_op)
            #current_op.finalise()
            job.num_finished_ops += 1
    
    def update_job_state(
        self,
        job: Job,
        state: str,
    ) -> None:
        """method to update the state of a job in the job database"""
        # update state tracking of the job instance
        job.stat_monitor.set_state(state=state)
        # update job database
        self.update_job_db(job=job, property='state', val=state)
        # only update operation state if it is not finished
        # operations are finished by post-process call to their 'finalise' method
        
        # update state of the corresponding operation
        if job.current_op is not None:
            self.update_operation_state(op=job.current_op, state=state)
    
    def get_next_operation(
        self,
        job: Job,
        ) -> Operation | None:
        """
        get next operation of given job
        """
        # last operation information
        job.last_op = job.current_op
        job.last_proc_time = job.current_proc_time
        job.last_setup_time = job.current_setup_time
        job.last_order_time = job.current_order_time
        # current operation information
        if job.open_operations:
            op = job.open_operations.popleft()
            job.current_proc_time = op.proc_time
            job.current_setup_time = op.setup_time
            job.current_order_time = op.order_time
            # only reset job prio if there are OP-wise defined priorities
            if job.op_wise_prio:
                job.prio = op.prio # use setter function to catch possible errors
                self.update_job_db(job=job, property='prio', val=job.prio)
            if job.op_wise_starting_date:
                job.time_planned_starting = op.time_planned_starting
                self.update_job_db(job=job, property='planned_starting_date', val=job.time_planned_starting)
            if job.op_wise_ending_date:
                job.time_planned_ending = op.time_planned_ending
                self.update_job_db(job=job, property='planned_ending_date', val=job.time_planned_ending)
        else:
            op = None
            job.current_proc_time = None
            job.current_setup_time = None
            job.current_order_time = None
        
        job.current_op = op
        
        return op
    
    ### OPERATIONS ###
    def register_operation(
        self,
        obj: Operation,
        exec_system_identifier: CustomID,
        target_station_group_identifier: CustomID | None,
        custom_identifier: CustomID | None,
        name: str | None,
        state: str,
    ) -> ObjectID:
        """
        registers an operation object in the dispatcher instance by assigning an unique id and 
        adding the object to the associated operations
        
        obj: operation to register
        machine_identifier: custom ID of the associated machine (user interface)
        custom_identifier: custom identifier of the operation 
            (kept for consistency reasons, perhaps remove later)
        name: assigned name the operation
        status: for future features if status of operations is tracked
        
        outputs:
        op_id: assigned operation ID
        name: assigned name
        machine: corresponding machine infrastructure object
        """
        # infrastructure manager
        infstruct_mgr = self.env.infstruct_mgr
        
        # obtain id
        op_id = self._obtain_load_obj_id(load_type='op')
        # time of creation
        #creation_date = self.env.now()
        creation_date = self.env.t_as_dt()
        
        # custom name
        if name is None:
            name = f'O_gen_{op_id}'
            
        # setup time
        setup_time: float = 0.
        if obj.setup_time is not None:
            setup_time = obj.setup_time
        
        # corresponding execution system in which the operation is performed
        # no pre-determined assignment of processing stations
        global EXEC_SYSTEM_TYPE
        exec_system = infstruct_mgr.lookup_subsystem_info(
                                                    subsystem_type=EXEC_SYSTEM_TYPE,
                                                    lookup_property='custom_id',
                                                    lookup_val=exec_system_identifier)
        # if target station group is specified, get instance
        target_station_group: StationGroup | None = None
        if target_station_group_identifier is not None:
            target_station_group = infstruct_mgr.lookup_subsystem_info(
                                                    subsystem_type='StationGroup',
                                                    lookup_property='custom_id',
                                                    lookup_val= target_station_group_identifier)
            # validity check: only target stations allowed which are 
            # part of the current execution system
            if target_station_group.system_id not in exec_system:
                raise ValueError(f"{target_station_group} is not part of {exec_system}. \
                    Mismatch between execution system and associated station groups.")
        
        # new entry for operation data base
        new_entry: DataFrame = pd.DataFrame({
                                'op_id': [op_id],
                                'job_id': [obj.job_id],
                                'job_name': [obj.job.name()],
                                'custom_id': [custom_identifier],
                                'op': [obj],
                                'name': [name],
                                'prio': [obj.prio],
                                'execution_system': [exec_system],
                                'execution_system_custom_id': [exec_system.custom_identifier],
                                'execution_system_name': [exec_system.name],
                                'execution_system_type': [exec_system.subsystem_type],
                                'target_station_custom_id': [None],
                                'target_station_name': [None],
                                'proc_time': [obj.proc_time],
                                'setup_time': [setup_time],
                                'order_time': [obj.order_time],
                                'creation_date': [creation_date],
                                'release_date': [obj.time_release],
                                #'entry_date': [obj.time_entry],
                                'planned_starting_date': [obj.time_planned_starting],
                                'actual_starting_date': [obj.time_actual_starting],
                                'starting_date_deviation': [obj.starting_date_deviation],
                                #'exit_date': [obj.time_exit],
                                'planned_ending_date': [obj.time_planned_ending],
                                'actual_ending_date': [obj.time_actual_ending],
                                'ending_date_deviation': [obj.ending_date_deviation],
                                'lead_time': [obj.lead_time],
                                'state': [state]})
        new_entry: DataFrame = new_entry.astype(self._op_prop)
        new_entry = new_entry.set_index('op_id')
        self._op_db = pd.concat([self._op_db, new_entry])
        
        logger_dispatcher.info(f"Successfully registered operation with OpID {op_id} and name {name}")
        
        # write operation information directly
        obj.name = name
        obj.target_exec_system = exec_system
        obj.target_station_group = target_station_group
        obj.time_creation = creation_date
        
        # return operation ID
        return op_id
    
    def update_operation_db(
        self,
        op: Operation,
        property: str,
        val: float | str,
    ) -> None:
        """
        updates the information of a job for a given property
        """
        # check if property is a filter criterion
        if property not in self._op_update_props:
            raise IndexError(f"Property '{property}' is not allowed. Choose from {self._op_update_props}")
        # None type value can not be looked for
        if val is None:
            raise TypeError("The lookup value can not be of type 'None'.")
        
        self._op_db.at[op.op_id, property] = val
    
    def update_operation_state(
        self,
        op: Operation,
        state: str,
    ) -> None:
        """method to update the state of a operation in the operation database"""
        # update state tracking of the operation instance
        op.stat_monitor.set_state(state=state)
        # update operation database
        self.update_operation_db(op=op, property='state', val=state)

    def release_operation(
        self,
        op: Operation,
        target_station: ProcessingStation,
    ) -> None:
        """
        used to signal the release of the given operation
        necessary for time statistics
        """
        #current_time = self.env.now()
        current_time = self.env.t_as_dt()
        # release time
        op.time_release = current_time
        op.is_released = True
        # update operation database
        # release date
        self.update_operation_db(op=op, property='release_date', val=op.time_release)
        # target station: custom identifier + name
        self.update_operation_db(
                    op=op, property='target_station_custom_id', 
                    val=target_station.custom_identifier)
        self.update_operation_db(
                    op=op, property='target_station_name', 
                    val=target_station.name())
    
    def enter_operation(
        self,
        op: Operation,
    ) -> None:
        """
        used to signal the start of the given operation on a Processing Station
        necessary for time statistics
        """
        #current_time = self.env.now()
        current_time = self.env.t_as_dt()
        # starting time processing
        #op.time_entry = current_time
        op.time_actual_starting = current_time
        
        # starting times
        if op.time_planned_starting is not None:
            op.starting_date_deviation = op.time_actual_starting - op.time_planned_starting
            self.update_operation_db(op=op, property='starting_date_deviation', val=op.starting_date_deviation)
        
        # update operation database
        #self.update_operation_db(op=op, property='entry_date', val=op.time_entry)
        self.update_operation_db(op=op, property='actual_starting_date', val=op.time_actual_starting)
     
    def finish_operation(
        self,
        op: Operation,
    ) -> None:
        """
        used to signal the finalisation of the given operation
        necessary for time statistics
        """
        #current_time = self.env.now()
        current_time = self.env.t_as_dt()
        # [STATE] finished
        op.is_finished = True
        # [STATS] end + lead time
        #op.time_exit = current_time
        op.time_actual_ending = current_time
        #op.lead_time = op.time_exit - op.time_release
        op.lead_time = op.time_actual_ending - op.time_release
        
        # ending times
        if op.time_planned_ending is not None:
            op.ending_date_deviation = op.time_actual_ending - op.time_planned_ending
            self.update_operation_db(op=op, property='ending_date_deviation', val=op.ending_date_deviation)
        
        # update databases
        #logger_dispatcher.debug(f"Update databases for OP {op} ID {op.op_id} with [{op.time_exit, op.lead_time}]")
        logger_dispatcher.debug(f"Update databases for OP {op} ID {op.op_id} with [{op.time_actual_ending, op.lead_time}]")
        self.update_operation_state(op=op, state='FINISH')
        #self.update_operation_db(op=op, property='exit_date', val=op.time_exit)
        self.update_operation_db(op=op, property='actual_ending_date', val=op.time_actual_ending)
        self.update_operation_db(op=op, property='lead_time', val=op.lead_time)
        
        # [MONITOR] finalise stats
        op.stat_monitor.finalise_stats()
    
    ### PROPERTIES ###
    @property
    def job_db(self) -> DataFrame:
        """
        obtain a current overview of registered jobs in the environment
        """
        return self._job_db
    
    @property
    def op_db(self) -> DataFrame:
        """
        obtain a current overview of registered operations in the environment
        """
        return self._op_db

    #@lru_cache(maxsize=200)
    def get_job_obj_by_prop(
        self, 
        val: ObjectID | CustomID | str,
        property: str = 'job_id',
        target_prop: str = 'job',
    ) -> Job:
        """
        obtain a job object from the dispatcher by its property and corresponding value
        properties: job_id, custom_id, name
        """
        # check if property is a filter criterion
        if property not in self._job_lookup_props:
            raise IndexError(f"Property '{property}' is not allowed. Choose from {self._job_lookup_props}")
        # None type value can not be looked for
        if val is None:
            raise TypeError("The lookup value can not be of type 'None'.")
        
        # filter resource database for prop-value pair
        if property == 'job_id':
            # direct indexing for ID property; job_id always unique, no need for duplicate check
            try:
                temp1: Job = self._job_db.at[val, target_prop]
                return temp1
            except KeyError:
                raise IndexError(f"There were no jobs found for the property '{property}' \
                                with the value '{val}'")
        else:
            temp1: Series = self._job_db.loc[self._job_db[property] == val, target_prop]
            # check for empty search result, at least one result necessary
            if len(temp1) == 0:
                raise IndexError(f"There were no jobs found for the property '{property}' \
                                with the value '{val}'")
            # check for multiple entries with same prop-value pair
            ########### PERHAPS CHANGE NECESSARY
            ### multiple entries but only one returned --> prone to errors
            elif len(temp1) > 1:
                # warn user
                logger_dispatcher.warning(f"CAUTION: There are multiple jobs which share the \
                            same value '{val}' for the property '{property}'. \
                            Only the first entry is returned.")
            
            return temp1.iat[0]
    
    ### ROUTING LOGIC ###
    
    
    def request_job_allocation(
        self,
        job: Job,
        agent: bool,
    ) -> InfrastructureObject:
        """
        request an allocation decision for the given job 
        (determine the next processing station on which the job shall be placed)
        
        1. obtaining the target station group
        2. select from target station group (e.g. calling RL agent for that group)
        3. return target station (InfrastructureObject)
        
        requester: output side infrastructure object
        request for: infrastructure object instance
        """
        # SIGNALING ALLOCATION DECISION
        # (ONLY IF PARALLEL PROCESSING STATIONS EXIST)
        ## theoretically: obtaining next operation --> information about machine group -->
        ## based on machine group: choice of corresponding allocation agent -->
        ## preparing feature vectors as input --> trigger agent decision -->
        ## map decision to processing station
        
        logger_dispatcher.info(f"[DISPATCHER: {self}] REQUEST TO DISPATCHER FOR ALLOCATION")
        # set environment signal for ALLOCATION
        self._env.set_dispatching_signal(sequencing=False, reset=False)
        #self._env.main().activate()
        #yield job.hold(0)
        
        ## REWORK: NEW TOP-DOWN-APPROACH
        # routing of jobs is now organized in a hierarchical fashion and can be described
        # for each hierarchy level separately
        # routing in Production Areas --> Station Groups --> Processing Stations
        # so each job must contain information about the production areas and the corresponding station groups
        
        ## choice from station group stays as method
        # routing essentially depending on production areas --> JOB FROM AREA TO AREA
        # NOW DIFFERENTIATE:
        ### ~~(1) choice between station groups of the current area~~
        #           placement on machines outside the station group not possible
        ### (2) choice between processing stations of the current area
        #           placement on machines outside the station group possible, 
        #           but the stations could be filtered by their station group IDs
        ### --> (2) as implemented solution
        
        
        # get the next operation of the job
        #next_op = job.get_next_operation()
        next_op = self.get_next_operation(job=job)
        if next_op is not None:
            # get target execution system ((sub)system type) (defined by the global variable EXEC_SYSTEM_TYPE)
            target_exec_system = next_op.target_exec_system
            # get target station group
            target_station_group = next_op.target_station_group
           
            ##### PROCEDURE AGENT DECISION
            # build feature vector of the target station collection + given job instance
            # go into target station choice...
            ### ADD OBTAINED TARGET STATION GROUP
            logger_dispatcher.debug(f"[DISPATCHER: {self}] Next operation {next_op}")
            target_station = self._choose_target_station_from_exec_system(
                                            exec_system=target_exec_system,
                                            job=job,
                                            target_station_group=target_station_group)
            # check feasibility of the target station
            feasible = self._env.check_feasible_agent_alloc(
                                            target_station=target_station,
                                            op=next_op)
            if not feasible:
                raise RuntimeError((f"The choosen {target_station} is not feasible "
                    f"for {next_op}"))
            
            # ADDITIONAL: check if more than one station possible
            # check if agent available --> CHOICE METHOD
            ## agent registration in corresponding system (prod_area: allocation, resource: sequencing)
            # ...
            
            # with allocation request operation is released
            self.release_operation(op=next_op, target_station=target_station)
        # all operations done, look for sinks
        else:
            infstruct_mgr = self.env.infstruct_mgr
            sinks = infstruct_mgr.sinks
            # [PERHAPS CHANGE IN FUTURE]
            # use first sink of the registered ones
            target_station = sinks[0]
        
        logger_dispatcher.debug(f"[DISPATCHER: {self}] Next operation is {next_op} with machine group (machine) {target_station}")
        # reset environment signal for ALLOCATION
        self._env.set_dispatching_signal(sequencing=False, reset=True)
        
        return target_station
    
    def _choose_target_station_from_exec_system(
        self,
        exec_system: System,
        job: Job,
        target_station_group: StationGroup | None = None,
    ) -> ProcessingStation:
        """REWORK Choosing a target station from a given collection of processing stations

        Parameters
        ----------
        stations : Iterable[ProcessingStation]
            collection of processing stations from which the target station should be obtained

        Returns
        -------
        ProcessingStation
            station object on which the job should be placed
        """
        # 3 options:
        # (1) choice between processing stations of the current area
        #       --> only agent
        # (2) choice between station groups of the current area
        #       --> [OPTIONAL] only agent
        # (3) choice between processing stations of the current station group
        #       --> other allocation rules: the chosen target station 
        #           automatically fulfils feasibility
        
        # obtain the lowest level systems (only ProcessingStations) of 
        # that area or station group
        if target_station_group is not None and self._curr_alloc_rule != 'AGENT':
            # preselection of station group only with allocation rules other than >>AGENT<<
            stations = target_station_group.lowest_level_subsystems(only_processing_stations=True)
        else:
            # choose from whole production area (>>AGENT<< always)
            stations = exec_system.lowest_level_subsystems(only_processing_stations=True)
        
        # infrastructure manager
        infstruct_mgr = self.env.infstruct_mgr
        # [KPIs] calculate necessary information for decision making
        # put all associated processing stations of that group in 'TEMP' state
        infstruct_mgr.res_objs_temp_state(res_objs=stations, reset_temp=False)
        
        # available stations
        ## AGENT can choose from all stations, not only available ones
        ## availability of processing stations is checked in the feasibility method
        if self._curr_alloc_rule != 'AGENT':
            # choose only from available processing stations
            candidates: list[ProcessingStation] = [ps for ps in stations if ps.stat_monitor.is_available]
            # if there are no available ones: use all stations
            if candidates:
                avail_stations = candidates
            else:
                avail_stations = stations
        else:
            # check agent availability
            exec_system.check_alloc_agent()
            # get agent from execution system
            agent = exec_system.alloc_agent
            ### CALL BUILDING FEATURE VECTOR ###
            # procedure:
            # (0) Gym reset (startup phase)
            # (1) build feature vector [in this branch]
            # (2) pause simulation run (by creating an conditional loop)
            # (3) send observation / built feature vector 
            #     to Gym-Env (through agent property --> 'feat_vec')
            #     call: internal or external?
            # (4) get agent decision from Gym-Env (call to step function)
            # (5) write decision back to agent as property
            # (6) continue simulation with written action (should resume in this branch)
            # (7) in same time step:
            # (7.1) check feasible action
            # (7.2) calculate reward (implemented in agent)
            # (7.3) jump to (1) and use this feature vector as observation at (2)
            
            # indicate that request is being done
            # must be blocking
            agent.request_decision(disposable_job=job)
            #agent.activate()
            
            # main problem:
            # execution flow is not interrupted in salabim, thus program
            # execution continues --> env.step() method only finalises if next event is reached
            # to use salabim's internal handling the creation of an event is necessary, which is
            # only possible by using salabim's modelling techniques
            # problem: techniques can be used solely by salabim components
            
            # possible solution: use modelling flow before dispatcher call, placeholder: self.hold(0)
            # tested execution flow with hold(0): creates events at the same point in time
            # --> idea:
            # call dispatcher method to evaluate allocation rule -> ret: True (agent) False (otherwise)
            # if agent: set flags and build feature vector
            # yield self.hold(0) from calling component
            # env.step() should stop after execution of flag setting at point self.hold(0)
            # which allows the external loop to terminate and trigger an agent decision
            ## decision saved in agent properties
            # resume execution: should finally obtain the target station object ('request_job_allocation')
            ## standard way if no agent decision, otherwise use agent properties
            
            # inside method: set flags, build feature vector
            # agent obtains associated stations by the connected ProductionArea and its subsystems
            #agent.build_feat_vec(disposable_job=job)
            print(f'{agent.feat_vec=}')
            #time.sleep(5)
            while not agent.RL_decision_done:
                # wait for agent decision
                # when decision is done in Gym Env: communicate decision by
                # calling agent's method 'set_decision'
                
                # SIMULATE ACTION CHOICE
                # deterministic by user input
                #stat_idx = input("Please enter a target station index:")
                #stat_idx = int(stat_idx)
                # random
                stat_idx = self.np_rnd_gen.integers(0,3)
                
                # set agent decision
                agent.set_decision(action=stat_idx)
                
                #pass
            
            
            # stop simulation at this point
            #self._env.main().activate()
            
            #   --> implement in agent class
            #logger_dispatcher.debug(f"[DISPATCHER: {self}] Build feature vector for allocation agent {agent}")
            # TEST ONLY: make tuple out of all stations
            # all stations are available
            # stations separately not necessary because agent already has information about associated target resources
            avail_stations = tuple(stations)
            # LATER: generation in dedicated method to build feature vector
        
        logger_dispatcher.debug(f"[DISPATCHER: {self}] Available stations at {self.env.now()} are {avail_stations}")
        
        # apply different strategies to select a station out of the station group
        match self._curr_alloc_rule:
            case 'RANDOM':
                # [RANDOM CHOICE]
                target_station: ProcessingStation = random.choice(avail_stations)
            case 'UTILISATION':
                # [UTILISATION]
                # choose the station with the lowest utilisation to time
                target_station: ProcessingStation = min(avail_stations, key=attrgetter('stat_monitor.utilisation'))
                logger_dispatcher.debug(f"[DISPATCHER: {self}] Utilisation of {target_station=} is {target_station.stat_monitor.utilisation:.4f}")
            case 'WIP_LOAD_TIME':
                # WIP as load/processing time, choose station with lowest WIP
                target_station: ProcessingStation = min(avail_stations, key=attrgetter('stat_monitor.WIP_load_time'))
                logger_dispatcher.debug(f"[DISPATCHER: {self}] WIP LOAD TIME of {target_station=} is {target_station.stat_monitor.WIP_load_time}")
            case 'WIP_LOAD_JOBS':
                # WIP as number of associated jobs, choose station with lowest WIP
                target_station: ProcessingStation = min(avail_stations, key=attrgetter('stat_monitor.WIP_load_num_jobs'))
                logger_dispatcher.debug(f"[DISPATCHER: {self}] WIP LOAD NUM JOBS of {target_station=} is {target_station.stat_monitor.WIP_load_time:.2f}")
            case 'AGENT':
                # request decision with previously built feature vector
                # get index value for target station out of tuple
                # [ONLY TEST] simulate agent decision by user input
                # only if more than one station
                """
                if len(avail_stations) > 1:
                    pprint(f"The available stations are:\n {avail_stations}")
                    #stat_idx = input("Please enter a target station index:")
                    #stat_idx = int(stat_idx)
                    stat_idx = agent.action
                    target_station = avail_stations[stat_idx]
                else:
                    target_station = avail_stations[0]
                """
                stat_idx = agent.action
                target_station = avail_stations[stat_idx]
                
        # feasibility check in the request job allocation method after the target station is returned
        
        # [KPIs] reset all associated processing stations of that group to their original state
        infstruct_mgr.res_objs_temp_state(res_objs=stations, reset_temp=True)
        
        return target_station
    
    def _choose_target_station_from_collection(
        self,
        stations: Iterable[ProcessingStation],
    ) -> ProcessingStation:
        """Choosing a target station from a given collection of processing stations

        Parameters
        ----------
        stations : Iterable[ProcessingStation]
            collection of processing stations from which the target station should be obtained

        Returns
        -------
        ProcessingStation
            station object on which the job should be placed
        """
        
        # infrastructure manager
        infstruct_mgr = self.env.infstruct_mgr
        
        # [KPIs] calculate necessary information for decision making
        # put all associated processing stations of that group in 'TEMP' state
        infstruct_mgr.res_objs_temp_state(res_objs=stations, reset_temp=False)
        
        # choose only from available processing stations
        candidates = [ps for ps in stations if ps.stat_monitor.is_available]
        # check if there are available processing stations
        # if not: use all stations
        if candidates:
            avail_stations = candidates
        else:
            avail_stations = stations
        
        logger_dispatcher.debug(f"[DISPATCHER: {self}] Available stations at {self.env.now()} are {avail_stations}")
        
        # apply different strategies to select a station out of the station group
        match self._curr_alloc_rule:
            case 'RANDOM':
                # [RANDOM CHOICE]
                target_station: ProcessingStation = random.choice(avail_stations)
            case 'UTILISATION':
                # [UTILISATION]
                # choose the station with the lowest utilisation to time
                temp = sorted(avail_stations, key=attrgetter('stat_monitor.utilisation'), reverse=True)
                target_station: ProcessingStation = temp.pop()
                logger_dispatcher.debug(f"[DISPATCHER: {self}] Utilisation of {target_station=} is {target_station.stat_monitor.utilisation:.4f}")
            case 'WIP_LOAD_TIME':
                # WIP as load/processing time, choose station with lowest WIP
                temp = sorted(avail_stations, key=attrgetter('stat_monitor.WIP_load_time'), reverse=True)
                target_station: ProcessingStation = temp.pop()
                logger_dispatcher.debug(f"[DISPATCHER: {self}] WIP LOAD TIME of {target_station=} is {target_station.stat_monitor.WIP_load_time:.2f}")
            case 'WIP_LOAD_JOBS':
                # WIP as number of associated jobs, choose station with lowest WIP
                temp = sorted(avail_stations, key=attrgetter('stat_monitor.WIP_load_num_jobs'), reverse=True)
                target_station: ProcessingStation = temp.pop()
                logger_dispatcher.debug(f"[DISPATCHER: {self}] WIP LOAD NUM JOBS of {target_station=} is {target_station.stat_monitor.WIP_load_time:.2f}")
            case 'AGENT':
                # CHECK: agent available
                pass
        
        # [KPIs] reset all associated processing stations of that group to their original state
        infstruct_mgr.res_objs_temp_state(res_objs=stations, reset_temp=True)
        
        return target_station
    
    def request_job_sequencing(
        self,
        req_obj: ProcessingStation,
    ) -> tuple[Job, float, float]:
        """
        request a sequencing decision for a given queue of the requesting resource
        requester: input side processing stations
        request for: job instance
        
        req_obj: requesting object (ProcessingStation)
        """
        # SIGNALING SEQUENCING DECISION
        # (ONLY IF MULTIPLE JOBS IN THE QUEUE EXIST)
        ## theoretically: get logic queue of requesting object --> information about feasible jobs -->
        ## [*] choice of sequencing agent (based on which properties?) --> preparing feature vector as input -->
        ## trigger agent decision --> map decision to feasible jobs
        ## [*] use implemented priority rules as intermediate step
        
        logger_dispatcher.info(f"[DISPATCHER: {self}] REQUEST TO DISPATCHER FOR SEQUENCING")
        # set environment signal for SEQUENCING
        self._env.set_dispatching_signal(sequencing=True, reset=False)
        
        # get logic queue of requesting object
        # contains all feasible jobs for this resource
        logic_queue = req_obj.logic_queue
        # get job from logic queue with currently defined priority rule
        job = self.seq_priority_rule(queue=logic_queue)
        # reset environment signal for SEQUENCING
        self._env.set_dispatching_signal(sequencing=True, reset=True)
        
        return job, job.current_proc_time, job.current_setup_time
    
    def seq_priority_rule(
        self,
        queue: Queue,
    ) -> Job:
        """apply priority rules to a pool of jobs"""
        match self._curr_prio_rule:
            # first in, first out
            case 'FIFO':
                # salabim queue pops first entry if no index is specified, 
                # not last like in Python
                job = queue.pop()
            # last in, last out
            case 'LIFO':
                # salabim queue pops first entry if no index is specified, 
                # not last like in Python
                job = queue.pop(-1)
            # shortest processing time
            case 'SPT':
                # choose job with shortest processing time
                temp = queue.as_list()
                job: Job = min(temp, key=attrgetter('current_proc_time'))
                # remove job from original queue
                queue.remove(job)
            # longest processing time
            case 'LPT':
                # choose job with longest processing time
                temp = queue.as_list()
                job: Job = max(temp, key=attrgetter('current_proc_time'))
                # remove job from original queue
                queue.remove(job)
            # shortest setup time
            case 'SST':
                # choose job with shortest setup time
                temp = queue.as_list()
                job: Job = min(temp, key=attrgetter('current_setup_time'))
                # remove job from original queue
                queue.remove(job)
            # longest setup time
            case 'LST':
                # choose job with longest setup time
                temp = queue.as_list()
                job: Job = max(temp, key=attrgetter('current_setup_time'))
                # remove job from original queue
                queue.remove(job)
            case 'PRIO':
                # choose job with highest priority
                temp = queue.as_list()
                job: Job = max(temp, key=attrgetter('prio'))
                # remove job from original queue
                queue.remove(job)
        
        return job
    
    ### ANALYSE ###
    def draw_gantt_chart(
        self,
        use_custom_proc_station_id: bool = True,
        sort_by_proc_station: bool = False,
        sort_ascending: bool = True,
        group_by_exec_system: bool = False,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'gantt_chart',
    ) -> PlotlyFigure:
        """
        draw a Gantt chart based on the dispatcher's operation database
        use_custom_machine_id: whether to use the custom IDs of the processing station (True) or its name (False)
        sort_by_proc_station: whether to sort by processing station property (True) or by job name (False) \
            default: False
        sort_ascending: whether to sort in ascending (True) or descending order (False) \
            default: True
        use_duration: plot each operation with its scheduled duration instead of the delta time \
            between start and end; if there were no interruptions both methods return the same results \
            default: False
        """
        # filter operation DB for relevant information
        filter_items: list[str] = [
            'job_name',
            'target_station_custom_id',
            'target_station_name',
            'execution_system',
            'execution_system_custom_id',
            'prio',
            #'entry_date',
            'planned_starting_date',
            'actual_starting_date',
            #'exit_date',
            'planned_ending_date',
            'actual_ending_date',
            'proc_time',
            'setup_time',
            'order_time',
        ]
        
        hover_data: dict[str, str | bool] = {
            'job_name': False,
            'target_station_custom_id': True,
            'execution_system_custom_id': True,
            'prio': True,
            #'entry_date': True,
            'planned_starting_date': True,
            'actual_starting_date': True,
            #'exit_date': True,
            'planned_ending_date': True,
            'actual_ending_date': True,
            'proc_time': True,
            'setup_time': True,
            'order_time': True,
        }
        
        #hover_template: str = "proc_time: %{proc_time|%d:%H:%M:%S}"
        
        df = self._op_db.filter(items=filter_items)
        # calculate delta time between start and end
        # Timedelta
        #df['delta'] = df['exit_date'] - df['entry_date']
        df['delta'] = df['actual_ending_date'] - df['actual_starting_date']
        
        # sorting
        sort_key: str = ''
        # choose relevant processing station property
        proc_station_prop: str = ''
        if use_custom_proc_station_id:
            proc_station_prop = 'target_station_custom_id'
        else:
            proc_station_prop = 'target_station_name'
        
        # check if sorting by processing station is wanted and custom ID should be used or not
        if sort_by_proc_station:
            sort_key = proc_station_prop
        else:
            sort_key = 'job_name' 
        
        df = df.sort_values(by=sort_key, ascending=sort_ascending, kind='stable')
        
        # group by value
        if group_by_exec_system:
            group_by_key = 'execution_system_custom_id'
        else:
            group_by_key = 'job_name'
        
        # build Gantt chart with Plotly Timeline
        fig: PlotlyFigure = px.timeline(
            df, 
            #x_start='entry_date',
            x_start='actual_starting_date',
            #x_end='exit_date',
            x_end='actual_ending_date',
            y=proc_station_prop, 
            color=group_by_key,
            hover_name='job_name',
            hover_data=hover_data,
        )
        fig.update_yaxes(type='category', autorange='reversed')
        #fig.update_traces(hovertemplate=hover_template)
        
        """
        fig.update_xaxes(type='linear')

        # reset axis scale for every figure element
        # https://stackoverflow.com/questions/66078893/plotly-express-timeline-for-gantt-chart-with-integer-xaxis
        for d in fig.data:
            try:
                # convert to integer if property is of that type in the database
                filt_val = int(d.name)
            except ValueError:
                filt_val = d.name
            filt = df[group_by_key] == filt_val
            d.x = df.loc[filt, 'delta']
        """

        fig.show()
        
        if save_html:
            file = f'{file_name}.html'
            fig.write_html(file)
        
        if save_img:
            file = f'{file_name}.svg'
            fig.write_image(file)
        
        return fig
    
    def draw_gantt_chart_old(
        self,
        use_custom_proc_station_id: bool = True,
        sort_by_proc_station: bool = False,
        sort_ascending: bool = True,
        group_by_exec_system: bool = False,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'gantt_chart',
    ) -> PlotlyFigure:
        """
        draw a Gantt chart based on the dispatcher's operation database
        use_custom_machine_id: whether to use the custom IDs of the processing station (True) or its name (False)
        sort_by_proc_station: whether to sort by processing station property (True) or by job name (False) \
            default: False
        sort_ascending: whether to sort in ascending (True) or descending order (False) \
            default: True
        use_duration: plot each operation with its scheduled duration instead of the delta time \
            between start and end; if there were no interruptions both methods return the same results \
            default: False
        """
        # filter operation DB for relevant information
        filter_items: list[str] = [
            'job_name',
            'target_station_custom_id',
            'target_station_name',
            'execution_system',
            'execution_system_custom_id',
            'prio',
            'entry_date',
            'exit_date',
            'proc_time',
            'setup_time',
            'order_time',
        ]
        
        hover_data: dict[str, str | bool] = {
            'job_name': False,
            'target_station_custom_id': True,
            'execution_system_custom_id': True,
            'prio': True,
            'entry_date': True,
            'exit_date': True,
            'proc_time': True,
            'setup_time': True,
            'order_time': True,
        }
        
        df = self._op_db.filter(items=filter_items)
        # calculate delta time between start and end
        # Timedelta
        df['delta'] = df['exit_date'] - df['entry_date']
        
        # sorting
        sort_key: str = ''
        # choose relevant processing station property
        proc_station_prop: str = ''
        if use_custom_proc_station_id:
            proc_station_prop = 'target_station_custom_id'
        else:
            proc_station_prop = 'target_station_name'
        
        # check if sorting by processing station is wanted and custom ID should be used or not
        if sort_by_proc_station:
            sort_key = proc_station_prop
        else:
            sort_key = 'job_name' 
        
        df = df.sort_values(by=sort_key, ascending=sort_ascending, kind='stable')
        
        # group by value
        if group_by_exec_system:
            group_by_key = 'execution_system_custom_id'
        else:
            group_by_key = 'job_name'
        
        # build Gantt chart with Plotly Timeline
        fig: PlotlyFigure = px.timeline(
            df, 
            x_start='entry_date', 
            x_end='exit_date', 
            y=proc_station_prop, 
            color=group_by_key,
            hover_name='job_name',
            hover_data=hover_data,
        )
        fig.update_yaxes(type='category', autorange='reversed')
        fig.update_xaxes(type='linear')

        # reset axis scale for every figure element
        # https://stackoverflow.com/questions/66078893/plotly-express-timeline-for-gantt-chart-with-integer-xaxis
        for d in fig.data:
            try:
                # convert to integer if property is of that type in the database
                filt_val = int(d.name)
            except ValueError:
                filt_val = d.name
            filt = df[group_by_key] == filt_val
            d.x = df.loc[filt, 'delta']

        fig.show()
        
        if save_html:
            file = f'{file_name}.html'
            fig.write_html(file)
        
        if save_img:
            file = f'{file_name}.svg'
            fig.write_image(file)
        
        return fig
    
    def finalise(self) -> None:
        """
        method to be called at the end of the simulation run by 
        the environment's "finalise_sim" method
        """
        self._calc_cycle_time()


# systems

class System(OrderedDict):
    
    def __init__(
        self,
        env: SimulationEnvironment,
        subsystem_type: str,
        custom_identifier: CustomID,
        abstraction_level: int,
        name: str | None = None,
        state: str | None = None,
    ) -> None:
        # [BASIC INFO]
        # environment
        self.env = env
        # subsystem information
        self._subsystem_type: str = subsystem_type
        # supersystem information
        self._supersystems: OrderedDict[ObjectID, System] = dict()
        self._supersystems_ids: set[ObjectID] = set()
        self._supersystems_custom_ids: set[CustomID] = set()
        # number of lower levels
        # how many levels of subsystems are possible
        self._abstraction_level = abstraction_level
        # indicator if the system contains processing stations
        self._containing_proc_stations: bool = False
        
        infstruct_mgr = self.env.infstruct_mgr
        self._system_id, self._name = infstruct_mgr.register_subsystem(
                                        subsystem_type=self._subsystem_type,
                                        obj=self, custom_identifier=custom_identifier,
                                        name=name, state=state)
        self._custom_identifier = custom_identifier
        
        # [AGENT] decision agent
        self._agent_types: set[str] = set(['SEQ', 'ALLOC'])
        self._alloc_agent_registered: bool = False
        # assignment
        self._alloc_agent: AllocationAgent | None = None
        
    ### REWORK
    def register_agent(
        self,
        agent: Agent,
        agent_type: str,
    ) -> tuple[Self, SimulationEnvironment]:
        
        if agent_type not in self._agent_types:
            raise ValueError(f"The agent type >>{agent_type}<< is not allowed. Choose from {self._agent_types}")
        
        match agent_type:
            case 'ALLOC':
                # allocation agents on lowest hierarchy level not allowed
                if self._abstraction_level == 0:
                    raise RuntimeError(f"Can not register allocation agents for lowest hierarchy level objects.")
                # registration, type and existence check
                if not self._alloc_agent_registered and isinstance(agent, AllocationAgent):
                    self._alloc_agent = agent
                    self._alloc_agent_registered = True
                    logger_env.info(f"Successfully registered Allocation Agent in {self}")
                elif not isinstance(agent, AllocationAgent):
                    raise TypeError(f"The object must be of type >>AllocationAgent<< but is type >>{type(agent)}<<")
                else:
                    raise AttributeError("There is already a registered AllocationAgent instance \
                                        Only one instance per system is allowed.")
            case 'SEQ':
                raise NotImplementedError(f"Registration of sequencing agents not supported yet!")
            
        return self, self.env
    
    @property
    def alloc_agent(self) -> AllocationAgent:
        if self._alloc_agent is None:
            raise ValueError("No AllocationAgent instance registered.")
        else:
            return self._alloc_agent
    
    def register_alloc_agent(
        self,
        alloc_agent: 'AllocationAgent',
    ) -> None:
        
        if self._abstraction_level == 0:
            raise RuntimeError(f"Can not register allocation agents for lowest hierarchy level objects.")
        
        if not self._alloc_agent_registered and isinstance(alloc_agent, AllocationAgent):
            self._alloc_agent = alloc_agent
            self._alloc_agent_registered = True
            logger_env.info(f"Successfully registered Allocation Agent in Env = {self.name()}")
        elif not isinstance(alloc_agent, AllocationAgent):
            raise TypeError(f"The object must be of type >>AllocationAgent<< but is type >>{type(alloc_agent)}<<")
        else:
            raise AttributeError("There is already a registered AllocationAgent instance \
                                 Only one instance per system is allowed.")
    
    def check_alloc_agent(self) -> None:
        """checks if an allocation agent is registered for the system

        Raises
        ------
        NoAllocationAgentAssignedError
            if no allocation agent was found
        """
        if not self._alloc_agent_registered:
            raise NoAllocationAgentAssignedError(f"The system {self} has no allocation agent assigned.")
        else:
            return None
    
    def __str__(self) -> str:
        return f'System (type: {self._subsystem_type}, custom_id: {self._custom_identifier}, name: {self._name})'
    
    def __repr__(self) -> str:
        return f'System (type: {self._subsystem_type}, custom_id: {self._custom_identifier}, name: {self._name})'
    
    def __key(self) -> tuple[ObjectID, str]:
        return (self._system_id, self._subsystem_type)
    
    def __hash__(self) -> int:
        return hash(self.__key())
    
    @property
    def subsystem_type(self) -> str:
        return self._subsystem_type
    
    @property
    def system_id(self) -> ObjectID:
        return self._system_id
    
    @property
    def custom_identifier(self) -> CustomID:
        return self._custom_identifier
    
    # compatibility to salabim component --> declaration as property not allowed
    def name(self) -> str | None:
        return self._name
    
    @property
    def abstraction_level(self) -> int:
        return self._abstraction_level
    
    @property
    def containing_proc_stations(self) -> bool:
        return self._containing_proc_stations
    
    @containing_proc_stations.setter
    def containing_proc_stations(
        self,
        val: bool,
    ) -> None:
        if not isinstance(val, bool):
            raise TypeError(f"Type of {val} must be boolean, but is {type(val)}")
        
        self._containing_proc_stations = val
    
    @property
    def supersystems(self) -> OrderedDict[ObjectID, System]:
        return self._supersystems
    
    @property
    def supersystems_ids(self) -> set[ObjectID]:
        return self._supersystems_ids
    
    @property
    def supersystems_custom_ids(self) -> set[CustomID]:
        return self._supersystems_custom_ids
    
    def as_list(self) -> list[System]:
        """output the associated subsystems as list

        Returns
        -------
        list[System]
            list of associated subsystems
        """
        return list(self.values())
    
    def as_tuple(self) -> tuple[System, ...]:
        """output the associated subsystems as tuple

        Returns
        -------
        tuple[System, ...]
            tuple of associated subsystems
        """
        return tuple(self.values())
    
    def add_supersystem(
        self,
        supersystem: System,
    ) -> None:
        if supersystem.system_id not in self._supersystems:
            self._supersystems[supersystem.system_id] = supersystem
            self._supersystems_ids.add(supersystem.system_id)
            self._supersystems_custom_ids.add(supersystem.custom_identifier)
    
    def add_subsystem(
        self,
        subsystem: System,
    ) -> None:
        """adding a subsystem to the given supersystem

        Parameters
        ----------
        subsystem : System
            subsystem object which shall be added to the supersystem

        Raises
        ------
        UserWarning
            if a subsystem is already associated with the given supersystem
        """
        # do not allow adding of subsystems for lowest level systems
        if self._abstraction_level == 0:
            raise RuntimeError(f"Tried to add subsystem to {self}, but it is \
                on the lowest hierarchy level. Systems on the lowest level can not contain other systems.")
        
        if subsystem.system_id not in self:
            self[subsystem.system_id] = subsystem
        else:
            raise UserWarning(f"Subsystem {subsystem} was already \
                in supersystem {self}!")
        
        subsystem.add_supersystem(supersystem=self)
        
        # register association in corresponding database
        infstruct_mgr = self.env.infstruct_mgr
        infstruct_mgr.register_system_association(supersystem=self, subsystem=subsystem)
        
        # check if a processing station was added
        if isinstance(subsystem, ProcessingStation):
            # set flag
            self._containing_proc_stations = True
            # update property in database
            infstruct_mgr.set_contain_proc_station(system=self)
        
        logger_infstrct.info(f"Successfully added {subsystem} to {self}.")
    
    #@lru_cache(maxsize=3)
    def lowest_level_subsystems(
        self,
        only_processing_stations: bool = False,
    ) -> tuple[InfrastructureObject, ...]:
        """obtain all associated InfrastructureObjects on the lowest hierarchy level

        Parameters
        ----------
        only_processing_stations : bool, optional
            return all associated InfrastructureObjects (False)
            or only ProcessingStations (True), by default False

        Returns
        -------
        tuple[InfrastructureObject, ...]
            tuple with all associated InfrastructureObjects

        Raises
        ------
        RuntimeError
            if system itself is on the lowest hierarchy level
        """
        
        if self._abstraction_level == 0:
            raise RuntimeError(f"Can not obtain lowest level subsystems from lowest hierarchy level objects.")
        
        remaining_abstraction_level = self._abstraction_level - 1
        subsystems = self.as_list()
        
        while remaining_abstraction_level > 0:
            temp: list[System] = list()
            
            for subsystem in subsystems:
                children = subsystem.as_list()
                temp.append(children)
                
            subsystems = temp.copy()
            remaining_abstraction_level -= 1
        
        # flatten list and remove duplicates by making a set
        low_lev_subsystems_set: set[InfrastructureObject] = set(flatten(subsystems))
        # filter only processing stations if option chosen
        if only_processing_stations:
            low_lev_subsystems_lst = filter_processing_stations(
                                        infstruct_obj_collection=low_lev_subsystems_set)
        else:
            # obtain list and 
            low_lev_subsystems_lst: list[InfrastructureObject] = list(low_lev_subsystems_set)
        
        # sort list by system ID (ascending), so that the order is always the same
        low_lev_subsystems_lst = sorted(low_lev_subsystems_lst, 
                                        key=attrgetter('system_id'), reverse=False)
        
        return tuple(low_lev_subsystems_lst)

class ProductionArea(System):
    
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Group of processing stations which are considered parallel machines
        """
        
        # initiliase base class
        super().__init__(subsystem_type='ProductionArea', abstraction_level=2, **kwargs)
    
    def add_subsystem(
        self,
        subsystem: System,
    ) -> None:
        """adding a subsystem to the given supersystem

        Parameters
        ----------
        subsystem : System
            subsystem object which shall be added to the supersystem

        Raises
        ------
        TypeError
            if a subsystem is not the type this system contains
        """
        # type check: only certain subsystems are allowed for each supersystem
        if not isinstance(subsystem, StationGroup):
            raise TypeError(f"The provided subsystem muste be of type >>StationGroup<<, \
                but it is {type(subsystem)}.")
            
        super().add_subsystem(subsystem=subsystem)

class StationGroup(System):
    
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Group of processing stations which are considered parallel machines
        """
        
        # initiliase base class
        super().__init__(subsystem_type='StationGroup', abstraction_level=1, **kwargs)
        
        return None
    
    def add_subsystem(
        self,
        subsystem: System,
    ) -> None:
        """adding a subsystem to the given supersystem

        Parameters
        ----------
        subsystem : System
            subsystem object which shall be added to the supersystem

        Raises
        ------
        TypeError
            if a subsystem is not the type this system contains
        """
        # type check: only certain subsystems are allowed for each supersystem
        if not isinstance(subsystem, InfrastructureObject):
            raise TypeError(f"The provided subsystem muste be of type >>InfrastructureObject<<, \
                but it is {type(subsystem)}.")
            
        super().add_subsystem(subsystem=subsystem)


# monitors

class Monitor:
    
    def __init__(
        self,
        env: SimulationEnvironment,
        obj: InfrastructureObject | Job | Operation,
        init_state: str = 'INIT',
        possible_states: Iterable[str] = (
            'INIT',
            'FINISH',
            'TEMP',
            'WAITING', 
            'PROCESSING',
            'SETUP', 
            'BLOCKED', 
            'FAILED', 
            'PAUSED',
        ),
        **kwargs,
    ) -> None:
        """
        Class to monitor associated objects (load and resource)
        """
        # initialise parent class if available
        super().__init__(**kwargs)
        
        # [REGISTRATION]
        self._env = env
        self._target_object = obj
        
        # [STATE] state parameters
        # all possible/allowed states
        self.states_possible: set[str] = set(possible_states)
        # always add states 'INIT', 'FINISH', 'TEMP' for control flow
        if not 'INIT' in self.states_possible:
            self.states_possible.add('INIT')
        if not 'FINISH' in self.states_possible:
            self.states_possible.add('FINISH')
        if not 'TEMP' in self.states_possible:
            self.states_possible.add('TEMP')
            
        # check integrity of the given state
        if init_state in self.states_possible:
            self.state_current: str = init_state
        else:
            raise ValueError(f"The state {state} is not allowed. Must be one of {self.states_possible}")
        
        # boolean indicator if a state is set
        self.state_status: dict[str, bool] = dict()
        # time counter for each state
        #self.state_times: dict[str, float] = dict()
        self.state_times: dict[str, Timedelta] = dict()
        # starting time variable indicating when the last state assignment took place
        #self.state_starting_time: float = self._env.t()
        self.state_starting_time: Datetime = self._env.t_as_dt()
        
        for state in self.states_possible:
            # init state time dictionary
            #self.state_times[state] = 0.
            self.state_times[state] = Timedelta()
            # init state is set to True
            if state == self.state_current:
                self.state_status[state] = True
            else:
                self.state_status[state] = False
                
        # DataFrame to further analyse state durations
        self.state_durations: DataFrame | None = None

        # availability indicator
        self._availability_states: set[str] = set([
            'WAITING',
        ])
        if self.state_current in self._availability_states:
            self.is_available: bool = True
        else:
            self.is_available: bool = False
        
        # additional 'TEMP' state information
        # indicator if state was 'TEMP'
        self._is_temp: bool = False
        # state before 'TEMP' was set
        self._state_before_temp: str = self.state_current
        # time components
        self.time_active: float = 0.
        #self.time_active: Timedelta = Timedelta()
        
        # time handling
        self._dt_parser: DTParser = DTParser()
    
    def __repr__(self) -> str:
        return f"Monitor instance of {self._target_object}"
    
    @property
    def env(self) -> SimulationEnvironment:
        return self._env
        
    def get_current_state(self) -> str:
        """get the current state of the associated resource"""
        return self.state_current
        
    def set_state(
        self,
        state: str,
    ) -> None:
        """
        function to set the object in the given state
        state: name of the state in which the object should be placed, must be part \
            of the object's possible states
        """
        # eliminate lower-case letters
        target_state = state.upper()
        
        # check if state is allowed
        if target_state not in self.states_possible:
            raise ValueError(f"The state {target_state} is not allowed. Must be one of {self.states_possible}")
        
        # check if state is already set
        if self.state_status[target_state] == True and target_state != 'TEMP':
            logger_monitors.info(f"Tried to set state of {self._target_object} to >>{target_state}<<, but this state was already set.\
                The object's state was not changed.")
        # check if the 'TEMP' state was already set, this should never happen
        # if it happens raise an error to catch wrong behaviour
        elif self.state_status[target_state] == True and target_state == 'TEMP':
            raise RuntimeError(f"Tried to set state of {self._target_object} to >>TEMP<<, but this state was already set.")
        
        # calculate time for which the object was in the current state before changing it
        current_state_start = self.state_starting_time
        #current_time = self._env.now()
        current_time = self._env.t_as_dt()
        current_state_duration: Timedelta = current_time - current_state_start
        # add time to the time counter for the current state
        current_state = self.state_current
        self.state_times[current_state] += current_state_duration
        
        # check if 'TEMP' state shall be set
        if target_state == 'TEMP':
            # set 'TEMP' state indicator to true
            self._is_temp = True
            # save current state for the state reset
            self._state_before_temp = current_state
        
        # set old state to False and new state to True
        self.state_status[current_state] = False
        self.state_status[target_state] = True
        # assign new state as current one
        self.state_current = target_state
        # set state starting time to current time
        self.state_starting_time = current_time
        # availability
        if self.state_current in self._availability_states:
            self.is_available: bool = True
        elif self.state_current == 'TEMP':
            # 'TEMP' state shall not change the availability indicator
            pass
        else:
            self.is_available: bool = False
        
        logger_monitors.debug(f"Duration for state {current_state} on {self._target_object} was {current_state_duration}")
    
    def reset_temp_state(self) -> None:
        """Reset from 'TEMP' state
        """
        # check if object was in TEMP state, raise error if not
        if not self._is_temp:
            raise RuntimeError(f"Tried to reset {self._target_object} from 'TEMP' state but \
                the current state is >>{self.state_current}<<")
        else:
            self._is_temp = False
            self.set_state(state=self._state_before_temp)
    
    def calc_KPI(
        self,
        is_finalise: bool = False,
    ) -> None:
        """calculates different KPIs at any point in time
        """
        
        # state durations for analysis
        if not is_finalise:
            self.state_durations = self.state_durations_as_df()
        
        # [TOTAL ACTIVE TIME]
        self.time_active = self.state_durations.loc[:, 'abs [seconds]'].sum()
    
    def state_durations_as_df(self) -> DataFrame:
        """Calculates absolute and relative state durations at the current time

        Returns
        -------
        DataFrame
            State duration table with absolute and relative values
        """
        # build state duration table
        temp1: Series = pd.Series(data=self.state_times)
        temp2: DataFrame = temp1.to_frame()
        temp2.columns = ['abs [Timedelta]']
        temp2['abs [seconds]'] = temp2['abs [Timedelta]'].apply(func= lambda x: x.total_seconds())
        temp2['rel [%]'] = temp2['abs [seconds]'] / temp2.sum(axis=0)['abs [seconds]'] * 100
        temp2 = temp2.drop(labels=['INIT', 'FINISH', 'TEMP'], axis=0)
        temp2 = temp2.sort_index(axis=0, ascending=True, kind='stable')
        state_durations_df = temp2.copy()
        
        return state_durations_df
    
    def finalise_stats(self) -> None:
        """finalisation of stats gathering"""
        
        # assign state duration table
        self.state_durations = self.state_durations_as_df()
        
        # calculate KPIs
        self.calc_KPI(is_finalise=True)
    
    ### ANALYSE AND CHARTS ###
    def draw_state_bar_chart(        
        self,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'state_distribution_bar',
        time_unit: str = 'hours',
    ) -> PlotlyFigure:
        """draws the collected state times of the object as bar chart"""
        data = pd.DataFrame.from_dict(data=self.state_times, orient='index', columns=['total time'])
        data.index = data.index.rename('state')
        # change time from Timedelta to any time unit possible --> float
        # Plotly can not handle Timedelta objects properly, only Datetimes
        calc_td = self._dt_parser.timedelta_from_val(val=1., time_unit=time_unit)
        calc_col: str = f'total time [{time_unit}]'
        data[calc_col] = data['total time'] / calc_td
        data = data.sort_index(axis=0, kind='stable')
        
        fig: PlotlyFigure = px.bar(data, y=calc_col, text_auto='.2f')
        fig.update_layout(title=f'State Time Distribution of {self._target_object}', showlegend=False)
        fig.update_yaxes(title=dict({'text': calc_col}))
        
        fig.show()
        
        file_name = file_name + f'_{self}'
        
        if save_html:
            file = f'{file_name}.html'
            fig.write_html(file)
        
        if save_img:
            file = f'{file_name}.svg'
            fig.write_image(file)
        
        return fig
    
    def draw_state_pie_chart(        
        self,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'state_distribution_pie',
        time_unit: str = 'hours',
    ) -> PlotlyFigure:
        """draws the collected state times of the object as bar chart"""
        data = pd.DataFrame.from_dict(data=self.state_times, orient='index', columns=['total time'])
        data.index = data.index.rename('state')
        # change time from Timedelta to any time unit possible --> float
        # Plotly can not handle Timedelta objects properly, only Datetimes
        calc_td = self._dt_parser.timedelta_from_val(val=1., time_unit=time_unit)
        calc_col: str = f'total time [{time_unit}]'
        data[calc_col] = data['total time'] / calc_td
        data = data.sort_index(axis=0, kind='stable')
        data = data.loc[data[calc_col] > 0., :]
        
        fig: PlotlyFigure = px.pie(data, values=calc_col, names=data.index)
        fig.update_layout(title=f'State Time Distribution of {self._target_object}')
        
        fig.show()
        
        file_name = file_name + f'_{self}'
        
        if save_html:
            file = f'{file_name}.html'
            fig.write_html(file)
        
        if save_img:
            file = f'{file_name}.svg'
            fig.write_image(file)
        
        return fig

class BufferMonitor(Monitor):
    
    def __init__(
        self,
        obj: Buffer,
        **kwargs,
    ) -> None:
        # initialise parent class
        super().__init__(obj=obj, **kwargs)
        
        # fill level tracking
        """
        self._level_db_types = {
            'sim_time': float,
            'duration': float,
            'level': int,
        }
        """
        self._level_db_types = {
            'sim_time': object,
            'duration': object,
            'level': int,
        }
        """
        self._level_db: DataFrame = pd.DataFrame(
                                        columns=['sim_time', 'duration', 'level'], 
                                        data=[[0., 0., obj.start_fill_level]])
        """
        self._level_db: DataFrame = pd.DataFrame(
                                        columns=['sim_time', 'duration', 'level'], 
                                        data=[[self.env.t_as_dt(), Timedelta(), obj.start_fill_level]])
        self._level_db = self._level_db.astype(self._level_db_types)
        
        self._current_fill_level = obj.start_fill_level
        #self._fill_level_starting_time: float = self.env.now()
        self._fill_level_starting_time: Datetime = self.env.t_as_dt()
        self._wei_avg_fill_level: float | None = None
        
    @property
    def wei_avg_fill_level(self) -> float:
        return self._wei_avg_fill_level
    
    @property
    def level_db(self) -> DataFrame:
        return self._level_db
    
    def set_state(
        self,
        state: str,
    ) -> None:
        """additional level tracking functionality"""
        super().set_state(state=state)
        
        is_finalise: bool = False
        if self.state_current == 'FINISH':
            is_finalise: bool = True
        self.track_fill_level(is_finalise=is_finalise)
        
    # Buffer fill level tracking
    def track_fill_level(
        self,
        is_finalise: bool = False,
    ) -> None:
        """adds an entry to the fill level database"""
        # only calculate duration if buffer level changes
        #current_time = self.env.now()
        current_time = self.env.t_as_dt()
        duration: Timedelta = current_time - self._fill_level_starting_time
        logger_buffers.debug(f"[BUFFER: {self._target_object}] Current time is {current_time} with level {len(self._target_object)} and old level {self._current_fill_level}")
        #if ((self._current_fill_level != len(self)) and (duration > 0.0)) or is_finalise:
        if (self._current_fill_level != len(self._target_object)) or is_finalise:
            temp1: Series = pd.Series(
                                    index=['sim_time', 'duration', 'level'],
                                    data=[current_time, duration, self._current_fill_level])
            temp2: DataFrame = temp1.to_frame().T.astype(self._level_db_types)
            self._level_db = pd.concat([self._level_db, temp2], ignore_index=True)
            self._current_fill_level = len(self._target_object)
            self._fill_level_starting_time = current_time
        
    def finalise_stats(self) -> None:
        """finalisation of stats gathering"""
        # execute parent class function
        super().finalise_stats()
        
        # finalise fill level tracking
        self.track_fill_level(is_finalise=True)
        
        # weighted average fill level
        self._level_db = self._level_db.loc[self._level_db['duration'] > Timedelta(), :].copy()
        self._level_db = self._level_db.reset_index(drop=True)
        temp1: DataFrame = self._level_db.copy()
        temp1['duration_seconds'] = temp1['duration'].apply(func= lambda x: x.total_seconds())
        temp1['mul'] = temp1['duration_seconds'] * temp1['level']
        sums: Series = temp1.filter(items=['duration_seconds', 'mul']).sum(axis=0)
        #sums: Series = temp1.sum(axis=0)
        self._wei_avg_fill_level: float = sums['mul'] / sums['duration_seconds']
        
        
    ### ANALYSE AND CHARTS ###
    def draw_fill_level(
        self,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'fill_level',
    ) -> PlotlyFigure:
        """
        method to draw and display the fill level expansion of the corresponding buffer
        """
        # add starting point to start chart at t = init time
        data = self._level_db.copy()
        val1: float = data.at[0, 'sim_time'] - data.at[0, 'duration']
        val2: float = 0.
        val3: int = data.at[0, 'level']
        temp1: DataFrame = pd.DataFrame(columns=data.columns, data=[[val1, val2, val3]])
        temp1 = pd.concat([temp1, data], ignore_index=True)
        
        fig: PlotlyFigure = px.line(x=temp1['sim_time'], y=temp1['level'], line_shape="vh")
        fig.update_traces(line=dict(width=3))
        fig.update_layout(title=f'Fill Level of {self._target_object}')
        fig.update_yaxes(title=dict({'text': 'fill level [-]'}))
        fig.update_xaxes(title=dict({'text': 'time'}))
        # weighted average fill level
        fig.add_hline(
                    y=self.wei_avg_fill_level, line_width=3, 
                    line_dash='dot', line_color='orange')
        # capacity
        cap = self._target_object.capacity()
        if cap < INF:
            fig.add_hline(
                        y=cap, line_width=3, 
                        line_dash='dash', line_color='red')
        
        fig.show()
        
        file_name = file_name + f'_{self}'
        
        if save_html:
            file = f'{file_name}.html'
            fig.write_html(file)
        
        if save_img:
            file = f'{file_name}.svg'
            fig.write_image(file)
        
        return fig

class InfStructMonitor(Monitor):
    
    def __init__(
        self,
        obj: ProcessingStation,
        **kwargs,
    ) -> None:
        # initialise parent class
        super().__init__(obj=obj, **kwargs)
        
        # WIP tracking time load
        """
        self._WIP_time_db_types = {
            'sim_time': float,
            'duration': float,
            'level': float,
        }
        """
        self._WIP_time_db_types = {
            'sim_time': object,
            'duration': object,
            'level': object,
        }
        ###################### PERHAPS ADD STARTING LEVEL LATER
        """
        self._WIP_time_db: DataFrame = pd.DataFrame(
                                        columns=['sim_time', 'duration', 'level'], 
                                        data=[[0., 0., 0.]])
        """
        self._WIP_time_db: DataFrame = pd.DataFrame(
                                        columns=['sim_time', 'duration', 'level'], 
                                        data=[[self.env.t_as_dt(), Timedelta(), Timedelta()]])
        self._WIP_time_db = self._WIP_time_db.astype(self._WIP_time_db_types)
        
        # WIP tracking number of jobs
        """
        self._WIP_num_db_types = {
            'sim_time': float,
            'duration': float,
            'level': int,
        }
        """
        self._WIP_num_db_types = {
            'sim_time': object,
            'duration': object,
            'level': int,
        }
        ###################### PERHAPS ADD STARTING LEVEL LATER
        """
        self._WIP_num_db: DataFrame = pd.DataFrame(
                                        columns=['sim_time', 'duration', 'level'], 
                                        data=[[0., 0., 0]])
        """
        self._WIP_num_db: DataFrame = pd.DataFrame(
                                        columns=['sim_time', 'duration', 'level'], 
                                        data=[[self.env.t_as_dt(), Timedelta(), 0]])
        self._WIP_num_db = self._WIP_num_db.astype(self._WIP_num_db_types)
        
        #self._current_WIP_time: float = 0.
        #self._last_WIP_time: float = 0.
        #self._current_WIP_num: int = 0
        #self._last_WIP_num: int = 0
        
        #self._WIP_time_starting_time: float = self.env.now()
        self._WIP_time_starting_time: Datetime = self.env.t_as_dt()
        #self._WIP_num_starting_time: float = self.env.now()
        self._WIP_num_starting_time: Datetime = self.env.t_as_dt()
        self._wei_avg_WIP_level_time: Timedelta | None = None
        self._wei_avg_WIP_level_num: float | None = None
        
        # time components
        self.time_occupied: float = 0.
        
        # resource KPIs
        self.utilisation: float = 0.
        
        # logistic objective values
        #self.WIP_load_time: float = 0.
        #self._WIP_load_time_last: float = 0.
        self.WIP_load_time: Timedelta = Timedelta()
        self._WIP_load_time_last: Timedelta = Timedelta()
        self.WIP_load_num_jobs: int = 0
        self._WIP_load_num_jobs_last: int = 0
    
    @property
    def wei_avg_WIP_level_time(self) -> float:
        return self._wei_avg_WIP_level_time
    
    @property
    def wei_avg_WIP_level_num(self) -> float:
        return self._wei_avg_WIP_level_num
    
    @property
    def WIP_time_db(self) -> DataFrame:
        return self._WIP_time_db
    
    @property
    def WIP_num_db(self) -> DataFrame:
        return self._WIP_num_db
    
    def _track_WIP_level(
        self,
        is_finalise: bool = False,
    ) -> None:
        """adds an entry to the fill level database"""
        # only calculate duration if level changes
        #current_time = self.env.now()
        current_time = self.env.t_as_dt()
        
        if (self._WIP_load_time_last != self.WIP_load_time) or is_finalise:
            
            # if updates occur at an already set time, just update the level
            if self._WIP_time_starting_time == current_time:
                self._WIP_time_db.iat[-1,2] = self.WIP_load_time
                self._WIP_load_time_last = self.WIP_load_time
            # else new entry
            else:
                duration = current_time - self._WIP_time_starting_time
                temp1: Series = pd.Series(
                                        index=['sim_time', 'duration', 'level'],
                                        data=[current_time, duration, self.WIP_load_time])
                temp2: DataFrame = temp1.to_frame().T.astype(self._WIP_time_db_types)
                self._WIP_time_db = pd.concat([self._WIP_time_db, temp2], ignore_index=True)
                self._WIP_load_time_last = self.WIP_load_time
                self._WIP_time_starting_time = current_time
            
        if (self._WIP_load_num_jobs_last != self.WIP_load_num_jobs) or is_finalise:
            
            # if updates occur at an already set time, just update the level
            if self._WIP_num_starting_time == current_time:
                self._WIP_num_db.iat[-1,2] = self.WIP_load_num_jobs
                self._WIP_load_num_jobs_last = self.WIP_load_num_jobs
            # else new entry
            else:
                duration = current_time - self._WIP_num_starting_time
                temp1: Series = pd.Series(
                                        index=['sim_time', 'duration', 'level'],
                                        data=[current_time, duration, self.WIP_load_num_jobs])
                temp2: DataFrame = temp1.to_frame().T.astype(self._WIP_num_db_types)
                self._WIP_num_db = pd.concat([self._WIP_num_db, temp2], ignore_index=True)
                self._WIP_load_num_jobs_last = self.WIP_load_num_jobs
                self._WIP_num_starting_time = current_time
    
    def calc_KPI(
        self,
        is_finalise: bool = False,
    ) -> None:
        
        super().calc_KPI()
        
        # [OCCUPATION]
        # properties which count as occupied
        # paused counts in because pausing the processing station is an external factor
        util_props = ['PROCESSING', 'PAUSED']
        self.time_occupied = self.state_durations.loc[util_props, 'abs [seconds]'].sum()
        
        # [UTILISATION]
        # avoid division by 0
        if self.time_active > 0.:
            self.utilisation: float = self.time_occupied / self.time_active
    
    def change_WIP(
        self,
        job: Job,
        remove: bool,
    ) -> None:
        # removing WIP
        if remove:
            # next operation of the job already assigned
            #self.WIP_load_time -= job.last_proc_time
            self.WIP_load_time -= job.last_order_time
            self.WIP_load_num_jobs -= 1
        else:
            #self.WIP_load_time += job.current_proc_time
            self.WIP_load_time += job.current_order_time
            self.WIP_load_num_jobs += 1
        
        self._track_WIP_level()
    
    def finalise_stats(self) -> None:
        """finalisation of stats gathering"""
        # execute parent class function
        super().finalise_stats()
        
        # finalise WIP level tracking
        self._track_WIP_level(is_finalise=True)
        
        # post-process WIP time level databases
        #print(f'I AM {self}')
        self._WIP_time_db['level'] = self._WIP_time_db['level'].shift(periods=1, fill_value=Timedelta())
        self._WIP_time_db = self._WIP_time_db.loc[self._WIP_time_db['duration'] > Timedelta(), :].copy()
        self._WIP_time_db = self._WIP_time_db.reset_index(drop=True)
        
        # weighted average WIP time level
        temp1: DataFrame = self._WIP_time_db.copy()
        temp1['level_seconds'] = temp1['level'].apply(func= lambda x: x.total_seconds())
        temp1['duration_seconds'] = temp1['duration'].apply(func= lambda x: x.total_seconds())
        temp1['mul'] = temp1['duration_seconds'] * temp1['level_seconds']
        sums: Series = temp1.filter(items=['duration_seconds', 'mul']).sum(axis=0)
        wei_avg_time_sec: float = sums['mul'] / sums['duration_seconds']
        self._wei_avg_WIP_level_time: Timedelta = Timedelta(seconds=wei_avg_time_sec)
        
        # post-process WIP num level databases
        self._WIP_num_db['level'] = self._WIP_num_db['level'].shift(periods=1, fill_value=Timedelta())
        self._WIP_num_db = self._WIP_num_db.loc[self._WIP_num_db['duration'] > Timedelta(), :].copy()
        self._WIP_num_db = self._WIP_num_db.reset_index(drop=True)
        # weighted average WIP num level
        temp1: DataFrame = self._WIP_num_db.copy()
        temp1['duration_seconds'] = temp1['duration'].apply(func= lambda x: x.total_seconds())
        temp1['mul'] = temp1['duration_seconds'] * temp1['level']
        sums: Series = temp1.filter(items=['duration_seconds', 'mul']).sum(axis=0)
        self._wei_avg_WIP_level_num: float = sums['mul'] / sums['duration_seconds']
    
    ### ANALYSE AND CHARTS ###
    def draw_WIP_level(
        self,
        use_num_jobs_metric: bool = False,
        save_img: bool = False,
        save_html: bool = False,
        file_name: str = 'fill_level',
        time_unit_load_time: str = 'hours',
    ) -> PlotlyFigure:
        """
        method to draw and display the fill level expansion of the corresponding buffer
        """
        # add starting point to start chart at t = init time
        if use_num_jobs_metric:
            data = self._WIP_num_db.copy()
            title = f'WIP Level Num Jobs of {self._target_object}'
            yaxis = 'WIP Level Number of Jobs [-]'
            avg_WIP_level = self._wei_avg_WIP_level_num
            #last_WIP_level = self.WIP_load_time
            last_WIP_level = self.WIP_load_num_jobs
        else:
            data = self._WIP_time_db.copy()
            # change WIP load time from Timedelta to any time unit possible --> float
            # Plotly can not handle Timedelta objects properly, only Datetimes
            calc_td = self._dt_parser.timedelta_from_val(val=1., time_unit=time_unit_load_time)
            data['level'] = data['level'] / calc_td
            title = f'WIP Level Time of {self._target_object}'
            yaxis = 'WIP Level Time [time units]'
            avg_WIP_level: float = self._wei_avg_WIP_level_time / calc_td
            last_WIP_level: float = self.WIP_load_time / calc_td
        f_val1: Datetime = data.at[0, 'sim_time'] - data.at[0, 'duration']
        f_val2: Timedelta = Timedelta()
        f_val3: float = data.at[0, 'level']
        first_entry: DataFrame = pd.DataFrame(columns=data.columns, data=[[f_val1, f_val2, f_val3]])
        l_val1: Datetime = data.iat[-1, 0]
        l_val2: Timedelta = Timedelta()
        l_val3: float = last_WIP_level # REWORK type hint
        last_entry: DataFrame = pd.DataFrame(columns=data.columns, data=[[l_val1, l_val2, l_val3]])
        temp1: DataFrame = pd.concat([first_entry, data, last_entry], ignore_index=True)
        
        fig: PlotlyFigure = px.line(x=temp1['sim_time'], y=temp1['level'], line_shape="vh")
        fig.update_traces(line=dict(width=3))
        fig.update_layout(title=title)
        fig.update_yaxes(title=dict({'text': yaxis}))
        fig.update_xaxes(title=dict({'text': 'time'}))
        # weighted average WIP level
        fig.add_hline(
                    y=avg_WIP_level, line_width=3, 
                    line_dash='dot', line_color='orange')
        
        fig.show()
        
        file_name = file_name + f'_{self}'
        
        if save_html:
            file = f'{file_name}.html'
            fig.write_html(file)
        
        if save_img:
            file = f'{file_name}.svg'
            fig.write_image(file)
        
        return fig


# infrastructure components

class InfrastructureObject(System, sim.Component):
    
    def __init__(
        self,
        env: SimulationEnvironment,
        custom_identifier: CustomID,
        name: str | None = None,
        setup_time: Timedelta | None = None,
        capacity: float = INF,
        state: str = 'INIT',
        possible_states: Iterable[str] = (
            'INIT',
            'FINISH',
            'TEMP',
            'WAITING', 
            'PROCESSING',
            'SETUP', 
            'BLOCKED', 
            'FAILED', 
            'PAUSED',
        ),
        **kwargs,
    ) -> None:
        """
        env: simulation environment in which the infrastructure object is embedded
        custom_identifier: unique user-defined custom ID of the given object \
            necessary for user interfaces
        capacity: capacity of the infrastructure object, if multiple processing \
            slots available at the same time > 1, default=1
        """
        # [STATS] Monitoring
        # special monitors for some classes
        if isinstance(self, Buffer):
            self._stat_monitor = BufferMonitor(
                                env=env, obj=self, init_state=state, 
                                possible_states=possible_states, **kwargs)
        elif isinstance(self, InfrastructureObject) and not isinstance(self, Buffer):
            self._stat_monitor = InfStructMonitor(
                                env=env, obj=self, init_state=state, 
                                possible_states=possible_states, **kwargs)
        else:
            self._stat_monitor = Monitor(env=env, obj=self, init_state=state, 
                                possible_states=possible_states, **kwargs)
        
        # assert machine information and register object in the environment
        current_state = self._stat_monitor.get_current_state()
        
        # [HIERARCHICAL SYSTEM INFORMATION]
        # contrary to other system types no bucket because a processing station 
        # is the smallest unit in the system view/analysis
        # initiliase base class >>System<<
        # calls to Infrastructure Manager to register object
        System.__init__(
            self,
            env=env,
            subsystem_type='Resource',
            custom_identifier=custom_identifier,
            abstraction_level=0,
            name=name,
            state=current_state,
            **kwargs
        )
        
        self.cap = capacity
        
        # [SALABIM COMPONENT] intialise base class
        process: str = 'main_logic'
        sim.Component.__init__(self, env=env, name=self._name, 
                               process=process, suppress_trace=True, **kwargs)
        
        # add logic queues
        # each resource uses one associated logic queue, logic queues are not physically available
        queue_name: str = f"queue_{self.name()}"
        self.logic_queue: Queue = sim.Queue(name=queue_name, env=self.env)
        
        # currently available jobs on that resource
        self.contents: OrderedDict[ObjectID, Job] = OrderedDict()
        
        # [STATS] additional information
        # number of inputs/outputs
        self.num_inputs: int = 0
        self.num_outputs: int = 0
        
        # time characteristics
        #self._proc_time: float = 0.
        self._proc_time: Timedelta = Timedelta.min
        # setup time: if a setup time is provided use always this time and ignore job-related setup times
        #self._setup_time = setup_time
        self._setup_time = setup_time
        if self._setup_time is not None:
            self._use_const_setup_time: bool = True
        else:
            self._use_const_setup_time: bool = False
    
    @property
    def stat_monitor(self) -> Monitor:
        return self._stat_monitor
    
    def td_to_simtime(
        self,
        timedelta: Timedelta,
    ) -> float:
        return self.env.timedelta_to_duration(timedelta=timedelta)
    
    @property
    def use_const_setup_time(self) -> bool:
        return self._use_const_setup_time
    
    @property
    def proc_time(self) -> Timedelta:
        return self._proc_time
    
    @proc_time.setter
    def proc_time(
        self,
        new_proc_time: Timedelta,
    ) -> None:
        if isinstance(new_proc_time, Timedelta):
            self._proc_time = new_proc_time
        else:
            raise TypeError(f"The processing time must be of type >>Timedelta<<, \
                but it is >>{type(new_proc_time)}<<")
    
    @property
    def setup_time(self) -> float:
        return self._setup_time
    
    @setup_time.setter
    def setup_time(
        self, 
        new_setup_time: float,
    ) -> None:
        if self._use_const_setup_time:
            raise RuntimeError(f"Tried to change setup time of >>{self}<<, but it is \
                configured to use a constant time of >>{self._setup_time}<<")
        
        if isinstance(new_setup_time, Timedelta):
            self._setup_time = new_setup_time
        else:
            raise TypeError(f"The setup time must be of type >>Timedelta<<, \
                but it is >>{type(new_proc_time)}<<")
    
    def add_content(
        self,
        job: Job,
    ) -> None:
        """add contents to the InfrastructureObject"""
        job_id = job.job_id
        if job_id not in self.contents:
            self.contents[job_id] = job
        else:
            raise KeyError(f"Job {job} already in contents of {self}")
    
    def remove_content(
        self,
        job: Job,
    ) -> None:
        """remove contents from the InfrastructureObject"""
        job_id = job.job_id
        if job_id in self.contents:
            del self.contents[job_id]
        else:
            raise KeyError(f"Job {job} not in contents of {self}")
    
    def put_job(
        self,
        job: Job,
    ) -> Generator[InfrastructureObject, None, InfrastructureObject]:
        """
        placing
        """
        # ALLOCATION REQUEST
        ## call dispatcher --> request for allocation
        ## self._dispatcher.request_allocation ...
        ### input job
        #### LATER: LOGIC FOR RESOURCE ALLOCATION (AGENT)
        ### - Dispatcher calls "get_next_operation"
        ### - Dispatcher returns target_machine
        ## ret: obtaining target machine
        # ++++++++++ add later ++++++++++++
        ## time component: given start date of operation
        ## returning release date, waiting for release date or release early
        dispatcher = self.env.dispatcher
        infstruct_mgr = self.env.infstruct_mgr
        # call dispatcher to check for allocation rule
        # if agent is set, do set flags and calculate feature vector
        print(f'--------------- DEBUG: call before hold(0) at {self.env.t()}')
        yield self.hold(0)
        print(f'--------------- DEBUG: call after hold(0) at {self.env.t()}')
        target_station = dispatcher.request_job_allocation(job=job)
        #target_station = yield from dispatcher.request_job_allocation(job=job)
        
        ### UPDATE JOB PROCESS INFO IN REQUEST FUNCTION???
        
        # get logic queue
        logic_queue = target_station.logic_queue
        # check if the target is a sink
        if isinstance(target_station, Sink):
            pass
        else:
            # check if associated buffers exist
            logger_prodStations.debug(f"[{self}] Check for buffers")
            buffers = target_station.buffers
            
            if buffers:
                #logger_prodStations.debug(f"[{self}] Buffer found")
                # [STATE:InfrStructObj] BLOCKED
                infstruct_mgr.update_res_state(obj=self, state='BLOCKED')
                # [STATE:Job] BLOCKED
                dispatcher.update_job_state(job=job, state='BLOCKED')
                yield self.to_store(store=buffers, item=job, fail_delay=FAIL_DELAY, fail_priority=1)
                if self.failed():
                    raise UserWarning(f"Store placement failed after {FAIL_DELAY} time steps. \
                        There seems to be deadlock.")
                # [STATE:Buffer] trigger state setting for target buffer
                buffer = self.to_store_store()
                if not isinstance(buffer, Buffer):
                    #logger_prodStations.debug(f"To store store object: {buffer}")
                    raise TypeError(f"From {self}: Job {job} Obj {buffer} is no buffer type at {self.env.now()}")
                buffer.activate()
                # [CONTENT:Buffer] add content
                buffer.add_content(job=job)
                # [STATS:Buffer] count number of inputs
                buffer.num_inputs += 1
                logger_prodStations.debug(f"obj = {self} \t type of buffer >>{buffer}<< = {type(buffer)} at {self.env.now()}")
            else:
                # adding request to machine
                # currently not possible because machines are components,
                # but resources which could be requested are not
                pass
        
        # [Job] enter logic queue after physical placement
        job.enter(logic_queue)
        # [STATS:WIP] REMOVING WIP FROM CURRENT STATION
        # remove only if it was added before, only case if the last operation exists
        if job.last_op is not None:
            self.stat_monitor.change_WIP(job=job, remove=True)
        # [STATS:WIP] ADDING WIP TO TARGET STATION
        # add only if there is a next operation, only case if the current operation exists
        if job.current_op is not None:
            target_station.stat_monitor.change_WIP(job=job, remove=False)
        
        # activate target processing station if passive
        if target_station.ispassive():
            target_station.activate()
        
        logger_prodStations.debug(f"[{self}] Put Job {job} in queue {logic_queue}")
    
        # [STATE:InfrStructObj] WAITING
        infstruct_mgr.update_res_state(obj=self, state='WAITING')
        # [STATE:Job] successfully placed --> WAITING
        dispatcher.update_job_state(job=job, state='WAITING')
        # [STATS:InfrStructObj] count number of outputs
        self.num_outputs += 1
        
        return target_station
    
    def get_job(self) -> Generator[None | InfrastructureObject, None, Job]:
        """
        getting jobs from associated predecessor resources
        """
        # entering target machine (logic_buffer)
        ## logic_buffer: job queue regardless of physical buffers
        ### entity physically on machine, but no true holding resource object (violates load-resource model)
        ### no capacity restrictions between resources, e.g. source can endlessly produce entities
        ## --- logic ---
        ## job enters logic queue of machine with unrestricted capacity
        ## each machine can have an associated physical buffer
        dispatcher = self.env.dispatcher
        infstruct_mgr = self.env.infstruct_mgr
        # request job and its time characteristics from associated queue
        job, job_proc_time, job_setup_time = dispatcher.request_job_sequencing(req_obj=self)
        
        ### UPDATE JOB PROCESS INFO IN REQUEST FUNCTION???
        
        # update time characteristics of the infrastructure object
        # contains additonal checks if the target values are allowed
        self.proc_time = job_proc_time
        if job_setup_time is not None:
            logger_prodStations.debug(f"-------->>>>> [SETUP TIME DETECTED] job ID {job.job_id} at {self.env.now()} on machine ID {self.custom_identifier} \
                with setup time {self.setup_time}")
            self.setup_time = job_setup_time
        
        # request and get job from associated buffer if it exists
        if self._buffers:
            yield self.from_store(store=self._buffers, filter=lambda item: item.job_id == job.job_id)
            buffer = self.from_store_store()
            # [STATS:Buffer] count number of outputs
            buffer.num_outputs += 1
            # [CONTENT:Buffer] remove content
            buffer.remove_content(job=job)
            # [STATE:Buffer] trigger state setting for target buffer
            buffer.activate()
        else:
            pass
        
        # RELEVANT INFORMATION BEFORE PROCESSING
        dispatcher.update_job_process_info(job=job, preprocess=True)
        # [STATS] count number of inputs
        self.num_inputs += 1
        # [CONTENT] add content
        self.add_content(job=job)
        
        # SETUP
        if self.setup_time is not None:
            # special state setting only for setup times
            # [STATE:InfrStructObj]
            infstruct_mgr.update_res_state(obj=self, state='SETUP')
            # [STATE:Job]
            dispatcher.update_job_state(job=job, state='SETUP')
            logger_prodStations.debug(f"[START SETUP] job ID {job.job_id} at {self.env.now()} on machine ID {self.custom_identifier} \
                with setup time {self.setup_time}")
            sim_time = self.td_to_simtime(timedelta=self.setup_time)
            #yield self.hold(self.setup_time)
            yield self.hold(sim_time)
        
        # [STATE:InfrStructObj] set state to processing
        infstruct_mgr.update_res_state(obj=self, state='PROCESSING')
        # [STATE:Job] successfully taken --> PROCESSING
        dispatcher.update_job_state(job=job, state='PROCESSING')
        
        return job
    
    ### PROCESS LOGIC
    # each method of 'pre_process', 'sim_control', 'post_process' must be implemented in the child classes
    def pre_process(self) -> None:
        """return type: tuple with parameters or None"""
        raise NotImplementedError(f"No pre-process method for {self} of type {self.__class__.__name__} defined.")
    
    def sim_control(self) -> None:
        """return type: tuple with parameters or None"""
        raise NotImplementedError(f"No sim-control method for {self} of type {self.__class__.__name__} defined.")
    
    def post_process(self) -> None:
        """return type: tuple with parameters or None"""
        raise NotImplementedError(f"No post-process method for {self} of type {self.__class__.__name__} defined.")
    
    def main_logic(self) -> Generator[Any, None, None]:
        """main logic loop for all resources in the simulation environment"""
        logger.debug(f"----> Process logic of {self}")
        # pre control logic
        ret = self.pre_process()
        # main control logic
        if ret is not None:
            ret = yield from self.sim_control(*ret)
        else:
            ret = yield from self.sim_control()
        # post control logic
        if ret is not None:
            ret = self.post_process(*ret)
        else:
            ret = self.post_process()
            
    def finalise(self) -> None:
        """
        method to be called at the end of the simulation run by 
        the environment's "finalise_sim" method
        """
        infstruct_mgr = self.env.infstruct_mgr
        # set finish state for each infrastructure object no matter of which child class
        infstruct_mgr.update_res_state(obj=self, state='FINISH')
        # finalise stat gathering
        self._stat_monitor.finalise_stats()

class ProcessingStation(InfrastructureObject):
    
    def __init__(
        self,
        buffers: Iterable[Buffer] | None = None,
        **kwargs,
    ) -> None:
        """
        env: simulation environment in which the infrastructure object is embedded
        capacity: capacity of the infrastructure object, if multiple processing \
            slots available at the same time > 1, default=1
        """
        # intialize base class
        super().__init__(**kwargs)
        
        # add physical buffers, more than one allowed
        # contrary to logic queues buffers are infrastructure objects and exist physically
        if buffers is None:
            self._buffers: set[Buffer] = set()
        else:
            self._buffers: set[Buffer] = set(buffers).copy()
        
        # add processing station to the associated ones of each buffer
        # necessary because if the number of resources for one buffer exceeds its capacity
        # deadlocks are possible
        for buffer in self._buffers:
            buffer.add_prod_station(prod_station=self)
    
    @property
    def station_group_id(self) -> ObjectID:
        return self._station_group_id
    
    @property
    def station_group(self) -> StationGroup:
        return self._station_group
    
    @property
    def buffers(self) -> set[Buffer]:
        return self._buffers
    
    def add_buffer(
        self,
        buffer: Buffer,
    ) -> None:
        """
        adding buffer to the current associated ones
        """
        # only buffer types allowed
        if not isinstance(buffer, Buffer):
            raise TypeError(f"Object is no Buffer type. Only objects of type Buffer can be added as buffers.")
        # check if already present
        if buffer not in self._buffers:
            self._buffers.add(buffer)
            buffer.add_prod_station(prod_station=self)
        else:
            logger_prodStations.warning(f"The Buffer >>{buffer}<< is already associated with the resource >>{self}<<. \
                Buffer was not added to the resource.")

    def remove_buffer(
        self,
        buffer: Buffer,
    ) -> None:
        """
        removing buffer from the current associated ones
        """
        if buffer in self._buffers:
            self._buffers.remove(buffer)
            buffer.remove_prod_station(prod_station=self)
        else:
            raise KeyError(f"The buffer >>{buffer}<< is not associated with the resource >>{self}<< and \
                therefore could not be removed.")
    
    ### PROCESS LOGIC
    def pre_process(self) -> None:
        infstruct_mgr = self.env.infstruct_mgr
        infstruct_mgr.update_res_state(obj=self, state='WAITING')
    
    def sim_control(self) -> Generator[None | Job, None, None]:
        dispatcher = self.env.dispatcher
        while True:
            # initialise state by passivating machines
            # resources are activated by other resources
            if len(self.logic_queue) == 0:
                yield self.passivate()
            logger_prodStations.debug(f"[MACHINE: {self}] is getting job from queue")
            
            # get job function from PARENT CLASS
            # ONLY PROCESSING STATIONS ARE ASKING FOR SEQUENCING
            # state setting --> 'PROCESSING'
            job = yield from self.get_job()
            
            logger_prodStations.debug(f"[START] job ID {job.job_id} at {self.env.now()} on machine ID {self.custom_identifier} \
                with proc time {self.proc_time}")
            # PROCESSING
            sim_time = self.td_to_simtime(timedelta=self.proc_time)
            yield self.hold(sim_time)
            # RELEVANT INFORMATION AFTER PROCESSING
            dispatcher.update_job_process_info(job=job, preprocess=False)
            
            logger_prodStations.debug(f"[END] job ID {job.job_id} at {self.env.now()} on machine ID {self.custom_identifier}")
            # only place job if there are open operations left
            # maybe add to 'put_job' method
            target_proc_station = yield from self.put_job(job=job)
            # [CONTENT:ProdStation] remove content
            self.remove_content(job=job)
            
    def post_process(self) -> None:
        pass
    
    def finalise(self) -> None:
        """
        method to be called at the end of the simulation run by 
        the environment's "finalise_sim" method
        """
        # each resource object class has dedicated finalise methods which 
        # must be called by children
        super().finalise()

class Machine(ProcessingStation):
    
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        ADD LATER
        """
        # assert object information
        self.res_type = 'Machine'
        
        # intialize base class
        super().__init__(**kwargs)

class Buffer(sim.Store, InfrastructureObject):
    
    def __init__(
        self,
        capacity: float,
        possible_states: Iterable[str] = (
            'INIT',
            'FINISH',
            'TEMP',
            'FULL',
            'EMPTY',
            'INTERMEDIATE',
            'FAILED',
            'PAUSED',
        ),
        fill_level: int = 0,
        **kwargs,
    ) -> None:
        """
        capacity: capacity of the buffer, can be infinite
        """
        # assert object information
        self.res_type = 'Buffer'
        self.start_fill_level = fill_level
        
        # intialize base classes
        # using hard-coded classes because salabim does not provide 
        # interfaces for multiple inheritance
        sim.Store.__init__(self, capacity=capacity, env=kwargs['env'])
        InfrastructureObject.__init__(
                            self, capacity=capacity, 
                            possible_states=possible_states, **kwargs)
        
        # material flow relationships
        self._associated_prod_stations: set[ProcessingStation] = set()
        self._count_associated_prod_stations: int = 0
    
    @property
    def level_db(self) -> DataFrame:
        return self._stat_monitor.level_db
    
    @property
    def wei_avg_fill_level(self) -> float:
        return self._stat_monitor.wei_avg_fill_level
    
    
    ### MATERIAL FLOW RELATIONSHIP
    def add_prod_station(
        self,
        prod_station: ProcessingStation
    ) -> None:
        """
        function to add processing stations which are associated with 
        """
        if not isinstance(prod_station, ProcessingStation):
            raise TypeError(f"Object is no ProcessingStation type. Only objects of type ProcessingStation can be added to a buffer.")
        
        # check if adding a new resource exceeds the given capacity
        # each associated processing station needs one storage place in the buffer
        # else deadlocks are possible
        if (self._count_associated_prod_stations + 1) > self.cap:
            raise UserWarning(f"Tried to add a new resource to buffer {self}, but the number of associated \
                resources exceeds its capacity which could result in deadlocks.")
        
        # check if processing station can be added
        if prod_station not in self._associated_prod_stations:
            self._associated_prod_stations.add(prod_station)
            self._count_associated_prod_stations += 1
        else:
            logger_buffers.warning(f"The Processing Station >>{prod_station}<< is already associated with the resource >>{self}<<. \
                Processing Station was not added to the resource.")
        
    def remove_prod_station(
        self,
        prod_station: ProcessingStation
    ) -> None:
        """
        removing a processing station from the current associated ones
        """
        if prod_station in self._associated_prod_stations:
            self._associated_prod_stations.remove(prod_station)
            self._count_associated_prod_stations -= 1
        else:
            raise KeyError(f"The processing station >>{prod_station}<< is not associated with the resource >>{self}<< and \
                therefore could not be removed.")
    
    ### PROCESS LOGIC
    def pre_process(self) -> None:
        infstruct_mgr = self.env.infstruct_mgr
        infstruct_mgr.update_res_state(obj=self, state='EMPTY')
    
    def sim_control(self) -> Generator[None, None, None]:
        infstruct_mgr = self.env.infstruct_mgr
        while True:
            logger_prodStations.debug(f"[BUFFER: {self}] Invoking at {self.env.now()}")
            # full
            if self.available_quantity() == 0:
                # [STATE] FULL
                infstruct_mgr.update_res_state(obj=self, state='FULL')
                logger_prodStations.debug(f"[BUFFER: {self}] Set to 'FULL' at {self.env.now()}")
            # empty
            elif self.available_quantity() == self.capacity():
                # [STATE] EMPTY
                infstruct_mgr.update_res_state(obj=self, state='EMPTY')
                logger_prodStations.debug(f"[BUFFER: {self}] Set to 'EMPTY' at {self.env.now()}")
            else:
                # [STATE] INTERMEDIATE
                infstruct_mgr.update_res_state(obj=self, state='INTERMEDIATE')
                logger_prodStations.debug(f"[BUFFER: {self}] Neither 'EMPTY' nor 'FULL' at {self.env.now()}")
            
            yield self.passivate()
        
        return None
            
    def post_process(self) -> None:
        pass
    
    def finalise(self) -> None:
        """
        method to be called at the end of the simulation run by 
        the environment's "finalise_sim" method
        """
        # each resource object class has dedicated finalise methods which 
        # must be called by children
        super().finalise()

class Source(InfrastructureObject):
    
    def __init__(
        self,
        proc_time: Timedelta = Timedelta(hours=2),
        random_generation: bool = False,
        job_generator: RandomJobGenerator | None = None,
        num_gen_jobs: int = 5,
        **kwargs,
    ) -> None:
        """
        num_gen_jobs: total number of jobs to be generated
        """
        # assert object information and register object in the environment
        self.res_type = 'Source'
        
        # random generation
        if random_generation and job_generator is None:
            raise ValueError("Random job generator instance needed for random job generation")
        
        self.random_generation = random_generation
        self.job_generator = job_generator
        
        ################## REWORK
        # initialize component with necessary process function
        random.seed(42)
        super().__init__(**kwargs)
        
        # parameters
        self.proc_time = proc_time
        self.num_gen_jobs = num_gen_jobs
    
    def _obtain_proc_time(self) -> float:
        """
        function to generate a constant or random processing time
        """
        proc_time = self.td_to_simtime(timedelta=self.proc_time)
        if self.random_generation:
            # random generation, add later
            return proc_time
        else:
            return proc_time
    
    ### PROCESS LOGIC
    def pre_process(self) -> None:
        infstruct_mgr = self.env.infstruct_mgr
        infstruct_mgr.update_res_state(obj=self, state='PROCESSING')
    
    def sim_control(self) -> Generator[None, None, None]:
        # id counter for debugging, else endless generation
        count = 0
        infstruct_mgr = self.env.infstruct_mgr
        dispatcher = self.env.dispatcher
        
        # use machine custom identifiers for generation
        machines = infstruct_mgr.res_db.loc[infstruct_mgr.res_db['res_type']=='Machine']
        machines_custom_ids = machines['custom_id'].to_list()
        
        # use station group custom identifiers for generation
        station_groups_custom_ids = infstruct_mgr.station_group_db['custom_id'].to_list()
        
        # use production area custom identifiers for generation
        prod_areas = infstruct_mgr.prod_area_db.copy()
        # prod_area_custom_ids = prod_areas.loc[prod_areas['containing_proc_stations'] == True,'custom_id'].to_list()
        prod_area_custom_ids = prod_areas.loc[prod_areas['containing_proc_stations'] == True,'custom_id']
        #prod_area_system_ids = prod_areas.loc[prod_areas['containing_proc_stations'] == True,:].index.to_list()
        # get station group custom identifiers which are associated with 
        # the relevant production areas
        stat_groups = infstruct_mgr.station_group_db.copy()
        stat_group_ids: dict[CustomID, list[CustomID]] = dict()
        for PA_sys_id, PA_custom_id in prod_area_custom_ids.items():
            # get associated station group custom IDs by their corresponding production area system ID
            candidates = stat_groups.loc[(stat_groups['prod_area_id'] == PA_sys_id), 'custom_id'].to_list()
            # map production area custom ID to the associated station group custom IDs
            stat_group_ids[PA_custom_id] = candidates
        
        while count < self.num_gen_jobs:
            # start at t=0 with generation
            # generate object
            ## random job properties
            ## currently: each job passes each machine, only one machine of each operation type
            #mat_ProcTimes, mat_JobMachID = self.job_generator.gen_rnd_job(n_machines=self.env.num_proc_stations)
            #job = Job(dispatcher=dispatcher, proc_times=mat_ProcTimes.tolist(), 
            #          machine_order=mat_JobMachID.tolist())
            #mat_ProcTimes, mat_JobMachID = self.job_generator.gen_rnd_job_by_ids(ids=machines_custom_ids)
            #mat_ProcTimes, mat_JobExOrder = self.job_generator.gen_rnd_job_by_ids(ids=station_groups_custom_ids, min_proc_time=5)
            (job_ex_order, job_target_station_groups, 
             proc_times, setup_times) = self.job_generator.gen_rnd_job_by_ids(
                exec_system_ids=prod_area_custom_ids,
                target_station_group_ids=stat_group_ids,
                #target_station_group_ids=None,
                min_proc_time=5,
                gen_setup_times=True,
            )
            logger_sources.debug(f"[SOURCE: {self}] ProcTimes {proc_times} at {self.env.now()}")
            
            # assign random priority
            prio = self.job_generator.gen_prio() + count
            #prio = [2,8]
            # assign starting and ending dates
            start_date_init = Datetime(2023, 11, 20, hour=6)
            end_date_init = Datetime(2023, 12, 1, hour=10)
            #start_date_init = [Datetime(2023, 11, 20, hour=6), Datetime(2023, 11, 21, hour=2)]
            #end_date_init = [Datetime(2023, 12, 1, hour=10), Datetime(2023, 12, 2, hour=2)]
            
            logger_sources.debug(f"{job_ex_order=}")
            logger_sources.debug(f"{job_target_station_groups=}")
            job = Job(dispatcher=dispatcher,
                      exec_systems_order=job_ex_order,
                      target_stations_order=job_target_station_groups,
                      proc_times=proc_times,
                      setup_times=setup_times,
                      prio=prio,
                      planned_starting_date=start_date_init,
                      planned_ending_date=end_date_init)
            # [Call:DISPATCHER]
            dispatcher.release_job(job=job)
            # [STATS:Source] count number of inputs (source: generation of jobs or entry in pipeline)
            # implemented in 'get_job' method which is not executed by source objects
            self.num_inputs += 1
            logger_sources.debug(f"[SOURCE: {self}] Generated {job} at {self.env.now()}")
            
            logger_sources.debug(f"[SOURCE: {self}] Request allocation...")
            # put job via 'put_job' function, implemented in parent class 'InfrastructureObject'
            target_proc_station = yield from self.put_job(job=job)
            logger_sources.debug(f"[SOURCE: {self}] PUT JOB with ret = {target_proc_station}")
            # [STATE:Source] put in 'WAITING' by 'put_job' method but still processing
            # only 'WAITING' if all jobs are generated
            infstruct_mgr.update_res_state(obj=self, state='PROCESSING')
            
            # hold for defined generation time (constant or statistically distributed)
            # if hold time elapsed start new generation
            proc_time = self._obtain_proc_time()
            logger_sources.debug(f"[SOURCE: {self}] Hold for >>{proc_time}<< at {self.env.now()}")
            yield self.hold(proc_time)
            # set counter up
            count += 1
        
        # [STATE:Source] WAITING
        infstruct_mgr.update_res_state(obj=self, state='WAITING')
        
    def post_process(self) -> None:
        pass

class Sink(InfrastructureObject):
    
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        num_gen_jobs: total number of jobs to be generated
        """
        # assert object information and register object in the environment
        self.res_type = 'Sink'
        
        # initialize parent class
        super().__init__(**kwargs)
    
    ### PROCESS LOGIC
    def pre_process(self) -> None:
        # currently sinks are 'PROCESSING' the whole time
        infstruct_mgr = self.env.infstruct_mgr
        infstruct_mgr.update_res_state(obj=self, state='PROCESSING')
    
    def sim_control(self) -> Generator[None, None, None]:
        dispatcher = self.env.dispatcher
        infstruct_mgr = self.env.infstruct_mgr
        while True:
            # in analogy to ProcessingStations
            if len(self.logic_queue) == 0:
                yield self.passivate()
            logger_sinks.debug(f"[SINK: {self}] is getting job from queue")
            # get job, simple FIFO
            job: Job = self.logic_queue.pop()
            # [Call:DISPATCHER] data collection: finalise job
            dispatcher.finish_job(job=job)
            #job.finalise()
            # destroy job object ???
            # if job object destroyed, unsaved information is lost
            # if not destroyed memory usage could increase
            
    def post_process(self) -> None:
        pass


# load components

class Operation:
    
    def __init__(
        self,
        dispatcher: Dispatcher,
        job: Job,
        exec_system_identifier: CustomID,
        proc_time: Timedelta,
        setup_time: Timedelta | None = None,
        target_station_group_identifier: CustomID | None = None,
        prio: int | None = None,
        planned_starting_date: Datetime | None = None,
        planned_ending_date: Datetime | None = None,
        custom_identifier: CustomID | None = None,
        name: str | None = None,
        state: str = 'INIT',
        possible_states: Iterable[str] = (
            'INIT',
            'FINISH',
            'WAITING',
            'SETUP',
            'PROCESSING',
            'BLOCKED',
            'FAILED',
            'PAUSED',
        ),
        **kwargs,
    ) -> None:
        """
        ADD DESCRIPTION
        """
        # !!!!!!!!! perhaps processing times in future multiple entries depending on associated machine
        # change of input format necessary, currently only one machine for each operation
        # no differing processing times for different machines or groups
        
        # initialise parent class if available
        super().__init__(**kwargs)
        
        # assert operation information
        self._dispatcher = dispatcher
        self._job = job
        self._job_id = job.job_id
        self._exec_system_identifier = exec_system_identifier
        self._target_station_group_identifier = target_station_group_identifier
        
        # [STATS] Monitoring
        self._stat_monitor = Monitor(env=self._dispatcher.env, obj=self, init_state=state, 
                                possible_states=possible_states, **kwargs)
        
        # process information
        # [STATS]
        # time characteristics
        self.proc_time = proc_time
        self.setup_time = setup_time
        if self.setup_time is not None:
            self.order_time = self.proc_time + self.setup_time
        else:
            self.order_time = self.proc_time
        # inter-process time characteristics
        # time of release
        #self.time_release: float = 0.
        self.time_release: Datetime = Datetime.min
        # time of first operation starting point
        self.time_actual_starting: Datetime = Datetime.min
        # starting date deviation
        self.starting_date_deviation: Timedelta | None = None
        # time of last operation ending point
        self.time_actual_ending: Datetime = Datetime.min
        # ending date deviation
        self.ending_date_deviation: Timedelta | None = None
        # lead time
        #self.lead_time: float = 0.
        self.lead_time: Datetime = Datetime.min
        # starting and end dates
        self.time_planned_starting = planned_starting_date
        self.time_planned_ending = planned_ending_date
        # in future setting starting points in advance possible
        self.is_finished: bool = False
        self.is_released: bool = False
        # priority, default: -1 --> no prio set
        self._prio = prio
        
        ########### adding machine instances
        ### perhaps adding machine sets if multiple machines possible (machine groups)
        # assignment of machine instance by dispatcher
        # from dispatcher: op_id, name, target_machine
        # register operation instance
        current_state = self._stat_monitor.get_current_state()
        
        # REWORK: only return OpID, other properties directly written by dispatcher method
        # add target station group by station group identifier
        self.name: str | None = None
        self.target_exec_system: System | None = None
        self.target_station_group: StationGroup | None = None
        self.time_creation: Datetime | None = None
        
        self._op_id = self.dispatcher.register_operation(
                                        obj=self, 
                                        exec_system_identifier=self._exec_system_identifier,
                                        target_station_group_identifier=target_station_group_identifier,
                                        custom_identifier=custom_identifier, name=name, 
                                        state=current_state)
        
        """
        (self._op_id, self.name,
         self.target_exec_system,
         self.time_creation) = self.dispatcher.register_operation(
                                                        obj=self, 
                                                        exec_system_identifier=self._exec_system_identifier,
                                                        custom_identifier=custom_identifier, name=name, 
                                                        state=current_state)
        """
    
    def __repr__(self) -> str:
        return (f'Operation(ProcTime: {self.proc_time}, ExecutionSystemID: {self._exec_system_identifier}, '
                f'SGI: {self._target_station_group_identifier})')    
    
    @property   
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher
    
    @property
    def stat_monitor(self) -> Monitor:
        return self._stat_monitor
    
    @property
    def op_id(self) -> ObjectID:
        return self._op_id
    
    @property
    def job(self) -> Job:
        return self._job
    
    @property
    def job_id(self) -> ObjectID:
        return self._job_id
    
    @property
    def exec_system_identifier(self) -> CustomID:
        return self._exec_system_identifier
    
    @property
    def target_station_group_identifier(self) -> CustomID | None:
        return self._target_station_group_identifier
    
    @property
    def prio(self) -> int:
        return self._prio
    
    @prio.setter
    def prio(
        self,
        new_prio: int,
    ) -> None:
        """setting of priority
        prio can be initialized as None, 
        but every change has to be an integer value

        Parameters
        ----------
        new_prio : int
            new priority which should be set

        Raises
        ------
        TypeError
            if other types are provided
        """
        if not isinstance(new_prio, int):
            raise TypeError((f"The type of {new_prio} must be >>int<<, but "
                             f"it is {type(new_prio)}"))
        else:
            self._prio = new_prio
            # REWORK changing OP prio must change job prio but only if op is job's current one

class Job(sim.Component):
    
    def __init__(
        self,
        dispatcher: Dispatcher,
        exec_systems_order: Sequence[CustomID],
        proc_times: Sequence[Timedelta],
        target_stations_order: Sequence[CustomID | None] | None = None,
        setup_times: Sequence[Timedelta | None] | None = None,
        prio: int | Sequence[int] | None = None,
        planned_starting_date: Datetime | Sequence[Datetime] | None = None,
        planned_ending_date: Datetime | Sequence[Datetime] | None = None,
        custom_identifier: CustomID | None = None,
        name: str | None = None,
        state: str = 'INIT',
        possible_states: Iterable[str] = (
            'INIT',
            'FINISH',
            'WAITING',
            'SETUP',
            'PROCESSING', 
            'BLOCKED', 
            'FAILED', 
            'PAUSED',
        ),
        additional_info: dict[str, CustomID] | None = None,
        **kwargs,
    ) -> None:
        """
        ADD DESCRIPTION
        """
        # add not provided information
        # target station identifiers
        #if target_stations_order is None:
        if isinstance(target_stations_order, Sequence):
            op_target_stations = target_stations_order
        else:
            op_target_stations: list[None] = [None] * len(exec_systems_order)
        # setup times
        if setup_times is None:
            setup_times: list[None] = [None] * len(proc_times)
        
        # prio
        self.op_wise_prio: bool = False
        self._prio: int | None = None
        if isinstance(prio, Sequence):
            op_prios = prio
            # job prio later set by 'get_next_operation' method
            self.op_wise_prio = True
        else:
            # job priority applies to all operations
            # priority, default: None --> no prio set
            op_prios: list[None] = [None] * len(proc_times)
            # set job priority as a whole
            self._prio = prio
        
        # planned dates
        self.op_wise_starting_date: bool = False
        self.op_wise_ending_date: bool = False
        self.time_planned_starting: Datetime | None = None
        self.time_planned_ending: Datetime | None = None
        if isinstance(planned_starting_date, Sequence):
            # operation-wise defined starting dates
            op_starting_dates = planned_starting_date
            # job starting date later set by 'get_next_operation' method
            self.op_starting_dates = True
        else:
            # only job-wise defined starting date
            op_starting_dates: list[None] = [None] * len(proc_times)
            self.time_planned_starting = planned_starting_date
        if isinstance(planned_ending_date, Sequence):
            # operation-wise defined starting dates
            op_ending_dates = planned_ending_date
            # job ending date later set by 'get_next_operation' method
            self.op_wise_ending_date = True
        else:
            # only job-wise defined starting date
            op_ending_dates: list[None] = [None] * len(proc_times)
            self.time_planned_ending = planned_ending_date
        
        ### VALIDITY CHECK ###
        # length of provided identifiers and lists must match
        if len(target_stations_order) != len(exec_systems_order):
            raise ValueError(("The number of target stations must match "
                "the number of execution systems."))
        if len(proc_times) != len(exec_systems_order):
            raise ValueError(("The number of processing times must match "
                "the number of execution systems."))
        if len(setup_times) != len(proc_times):
            raise ValueError((f"The number of setup times must match "
                "the number of processing times."))
        if len(op_prios) != len(proc_times):
            raise ValueError(("The number of operation priorities must match "
                "the number of processing times."))
        if len(op_starting_dates) != len(proc_times):
            raise ValueError(("The number of operation starting dates must match "
                "the number of processing times."))
        if len(op_ending_dates) != len(proc_times):
            raise ValueError(("The number of operation ending dates must match "
                "the number of processing times."))
        
        ### BASIC INFORMATION ###
        # assert job information
        self.custom_identifier = custom_identifier
        self.job_type: str = 'Job'
        self._dispatcher = dispatcher
        # sum of the proc times of each operation
        #self.total_proc_time: float = sum(proc_times)
        self.total_proc_time: Timedelta = sum(proc_times, Timedelta())
        
        # inter-process job state parameters
        # first operation scheduled --> released job
        self.is_released: bool = False
        # job's next operation is disposable
        # true for each new job, maybe reworked in future for jobs with
        # a start date later than creation date
        self.is_disposable: bool = True
        # add job to disposable ones
        #ret = self.dispatcher.add_disposable_job(self)
        # last operation ended --> finished job
        self.is_finished: bool = False
        
        # inter-process time characteristics
        # time of release
        #self.time_release: float = 0.
        self.time_release: Datetime = Datetime.min
        # time of first operation starting point
        self.time_actual_starting: Datetime = Datetime.min
        # starting date deviation
        self.starting_date_deviation: Timedelta | None = None
        # time of last operation ending point
        self.time_actual_ending: Datetime = Datetime.min
        # ending date deviation
        self.ending_date_deviation: Timedelta | None = None
        # lead time
        #self.lead_time: float = 0.
        self.lead_time: Datetime = Datetime.min
        # time of creation
        #self.time_creation: float = 0.
        self.time_creation: Datetime = Datetime.min
        
        # current resource location
        self._current_resource: InfrastructureObject | None = None
        
        # [STATS] Monitoring
        self._stat_monitor = Monitor(env=self._dispatcher.env, obj=self, init_state=state, 
                                possible_states=possible_states, **kwargs)
        
        # register job instance
        current_state = self._stat_monitor.get_current_state()
        
        env, self._job_id, name = self._dispatcher.register_job(
                                                    obj=self, custom_identifier=self.custom_identifier,
                                                    name=name, state=current_state)
        
        # intialize base class
        super().__init__(env=env, name=name, process='', **kwargs)
        
        ### OPERATIONS ##
        self.operations: deque[Operation] = deque()
        
        for idx, op_proc_time in enumerate(proc_times):
            op = Operation(
                dispatcher=self._dispatcher,
                job=self,
                proc_time=op_proc_time,
                setup_time=setup_times[idx],
                exec_system_identifier=exec_systems_order[idx],
                target_station_group_identifier=op_target_stations[idx],
                prio=op_prios[idx],
                planned_starting_date=op_starting_dates[idx],
                planned_ending_date=op_ending_dates[idx],
            )
            self.operations.append(op)
        
        self.open_operations = self.operations.copy()
        self.total_num_ops: int = len(self.operations)
        self.num_finished_ops: int = 0
        # current and last OP: properties set by method "get_next_operation"
        """
        self._last_op: Operation | None = None
        self._last_proc_time: Timedelta | None = None
        self._last_setup_time: Timedelta | None = None
        self._last_order_time: Timedelta | None = None
        self._current_op: Operation | None = None
        self._current_proc_time: Timedelta | None = None
        self._current_setup_time: Timedelta | None = None
        self._current_order_time: Timedelta | None = None
        """
        self.last_op: Operation | None = None
        self.last_proc_time: Timedelta | None = None
        self.last_setup_time: Timedelta | None = None
        self.last_order_time: Timedelta | None = None
        self.current_op: Operation | None = None
        self.current_proc_time: Timedelta | None = None
        self.current_setup_time: Timedelta | None = None
        self.current_order_time: Timedelta | None = None
        
        # ------- NOT IMPLEMENTED YET -------
        # rank-like property, set if job enters the infrastructure object
        # acts like a counter to allow easy sorting even if queue order is not maintained
        self._obj_entry_idx: int | None = None
        
        ### ADDITIONAL INFORMATION ###
        self.additional_info = additional_info
    
    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher
    
    @property
    def stat_monitor(self) -> Monitor:
        return self._stat_monitor
    
    @property
    def job_id(self) -> ObjectID:
        return self._job_id
    
    @property
    def prio(self) -> int:
        return self._prio
    
    @prio.setter
    def prio(
        self,
        new_prio: int,
    ) -> None:
        """setting of priority
        prio can be initialized as None, 
        but every change has to be an integer value

        Parameters
        ----------
        new_prio : int
            new priority which should be set

        Raises
        ------
        TypeError
            if other types are provided
        """
        if not isinstance(new_prio, int):
            raise TypeError((f"The type of {new_prio} must be >>int<<, but "
                             f"it is {type(new_prio)}"))
        else:
            self._prio = new_prio
    
    @property
    def obj_entry_idx(self) -> int | None:
        """
        returns the entry index which is set by each infrastructure object
        """
        return self._obj_entry_idx
    
    @property
    def current_resource(self) -> InfrastructureObject | None:
        """
        returns the current resource on which the job lies
        """
        return self._current_resource

    @current_resource.setter
    def current_resource(
        self,
        obj: InfrastructureObject
    ) -> None:
        """setting the current resource object which must be of type InfrastructureObject"""
        if not isinstance(obj, InfrastructureObject):
            raise TypeError(f"From {self}: Object >>{obj}<< muste be of type 'InfrastructureObject'")
        else:
            self._current_resource = obj
    """
    def get_next_operation(self) -> Operation | None:
        
        get next operation
        
        # last operation information
        self._last_op = self._current_op
        self._last_proc_time = self._current_proc_time
        self._last_setup_time = self._current_setup_time
        self._last_order_time = self._current_order_time
        # current operation information
        if self.open_operations:
            op = self.open_operations.popleft()
            self._current_proc_time = op.proc_time
            self._current_setup_time = op.setup_time
            self._current_order_time = op.order_time
            # only reset job prio if there are OP-wise defined priorities
            if self.op_wise_prio:
                self.prio = op.prio # use setter function to catch possible errors
                
            if self.op_wise_starting_date:
                self.time_planned_starting = op.time_planned_starting
            if self.op_wise_ending_date:
                self.time_planned_ending = op.time_planned_ending
        else:
            op = None
            self._current_proc_time = None
            self._current_setup_time = None
            self._current_order_time = None
        
        self._current_op = op
        
        return op
    """
    """
    def has_job_id(
        self,
        job_id: ObjectID,
    ) -> bool:
        
        checks whether the current job has the given id
        
        if self._job_id == job_id:
            return True
        else:
            return False
    """

