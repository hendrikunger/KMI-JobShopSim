from __future__ import annotations
from typing import TypeAlias, Self, Any
import datetime
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta


# utility functions
def flatten(
    lst_tpl: list[Any] | tuple[Any, ...]
) -> Generator[Any, None, None]:
    """flattens an arbitrarily nested list or tuple
    https://stackoverflow.com/questions/2158395/flatten-an-irregular-arbitrarily-nested-list-of-lists 

    Parameters
    ----------
    lst_tpl : list[Any] | tuple[Any, ...]
        arbitrarily nested list or tuple

    Yields
    ------
    Generator[Any, None, None]
        non-nested list or tuple
    """
    #
    for x in lst_tpl:
        # only flatten lists and tuples
        if isinstance(x, (list, tuple)):
            yield from flatten(x)
        else:
            yield x


### date and time handling ###

class DTParser:
    
    def __init__(
        self
    ) -> None:
        """
        date and time parser with convenient methods 
        to parse time units as timedelta and datetime objects
        """
        
        self._time_units_datetime: set[str] = set([
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'second',
            'microsecond',
        ])
        
        self._time_units_timedelta: set[str] = set([
            'weeks',
            'days',
            'hours',
            'minutes',
            'seconds',
            'milliseconds',
            'microseconds',
        ])
        
    def timedelta_from_val(
        self,
        val: float,
        time_unit: str,
    ) -> Timedelta:
        """create Python timedelta object by choosing time value and time unit

        Parameters
        ----------
        val : float
            duration
        time_unit : str
            target time unit

        Returns
        -------
        Timedelta
            timedelta object corresponding to the given values

        Raises
        ------
        ValueError
            if chosen time unit not implemented
        """
        
        if time_unit not in self._time_units_timedelta:
            raise ValueError(f"Time unit >>{time_unit}<< not supported. Choose from {self._time_units_timedelta}")
        
        match time_unit:
            case 'weeks':
                ret = datetime.timedelta(weeks=val)
            case 'days':
                ret = datetime.timedelta(days=val)
            case 'hours':
                ret = datetime.timedelta(hours=val)
            case 'minutes':
                ret = datetime.timedelta(minutes=val)
            case 'seconds':
                ret = datetime.timedelta(seconds=val)
            case 'milliseconds':
                ret = datetime.timedelta(milliseconds=val)
            case 'microseconds':
                ret = datetime.timedelta(microseconds=val)
                
        return ret