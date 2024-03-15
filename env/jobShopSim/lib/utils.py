from __future__ import annotations
from typing import TypeAlias, Self, Any
import datetime
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from datetime import date as Date
from datetime import tzinfo as TZInfo
from datetime import timezone
from zoneinfo import ZoneInfo
from pandas import DataFrame

# local time-zone, currently static
TIMEZONE_CEST = ZoneInfo('Europe/Berlin')
TIMEZONE_UTC = timezone.utc
DEFAULT_DATETIME: Datetime = Datetime(datetime.MINYEAR, 1, 1, tzinfo=TIMEZONE_UTC)


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

def current_time_tz(
    tz: TZInfo = TIMEZONE_UTC,
    cut_microseconds: bool = False,
) -> Datetime:
    """current time as datetime object with
    associated time zone information (UTC by default)

    Parameters
    ----------
    tz : TZInfo, optional
        time zone information, by default TIMEZONE_UTC

    Returns
    -------
    Datetime
        datetime object with corresponding time zone
    """
    if cut_microseconds:
        return Datetime.now(tz=tz).replace(microsecond=0)
    else:
        return Datetime.now(tz=tz)

def add_timedelta_with_tz(
    starting_dt: Datetime,
    td: Timedelta,
) -> Datetime:
    """time-zone-aware calculation of an end point in time 
    with a given timedelta

    Parameters
    ----------
    starting_dt : Datetime
        starting point in time
    td : Timedelta
        duration as timedelta object

    Returns
    -------
    Datetime
        time-zone-aware end point
    """
    
    if starting_dt.tzinfo is None:
        # no time zone information
        raise RuntimeError("The provided starting date does not contain time zone information.")
    else:
        # obtain time zone information from starting datetime object
        tz_info = starting_dt.tzinfo

    # transform starting point in time to utc
    dt_utc = starting_dt.astimezone(TIMEZONE_UTC)
    # all calculations are done in UTC
    # add duration
    ending_dt_utc = dt_utc + td
    # transform back to previous time zone
    ending_dt = ending_dt_utc.astimezone(tz=tz_info)
    
    return ending_dt

def validate_dt_UTC(
    dt: Datetime,
) -> bool:
    """_summary_

    Parameters
    ----------
    dt : Datetime
        datetime object to be checked for available UTC time zone
        information

    Returns
    -------
    bool
        True if UTC information is available

    Raises
    ------
    ValueError
        if no UTC time zone information is found
    """
    
    if dt.tzinfo != TIMEZONE_UTC:
        raise ValueError(f"Datetime object {dt} does not contain "
                         "necessary UTC time zone information")
    else:
        return True

def dt_to_timezone(
    dt: Datetime,
    target_tz: TZInfo = TIMEZONE_CEST,
) -> Datetime:
    """_summary_

    Parameters
    ----------
    dt : Datetime
        datetime with time zone information
    target_tz : TZInfo, optional
        target time zone information, by default TIMEZONE_CEST

    Returns
    -------
    Datetime
        datetime object adjusted to given local time zone

    Raises
    ------
    RuntimeError
        if datetime object does not contain time zone information
    """
    
    if dt.tzinfo is None:
        # no time zone information
        raise RuntimeError("The provided starting date does not contain time zone information.")
    # transform to given target time zone
    dt_local_tz = dt.astimezone(tz=target_tz)
    
    return dt_local_tz

# data wrangling

def get_date_cols_from_db(
    db: DataFrame,
) -> list[str]:
    
    target_cols: list[str] = list()
    
    for col in db.columns:
        if 'date' in col and 'deviation' not in col:
            target_cols.append(col)
    
    return target_cols.copy()

def adjust_db_dates_local_tz(
    db: DataFrame,
    tz: TZInfo = TIMEZONE_CEST,
) -> DataFrame:
    
    db = db.copy()
    # obtain date columns from database
    date_cols = get_date_cols_from_db(db=db)
    # adjust UTC times to local time zone provided
    temp1 = db[date_cols]
    temp1 = temp1.applymap(dt_to_timezone, na_action='ignore')
    db[date_cols] = temp1
    
    return db.copy()