import datetime
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from datetime import date as Date
from datetime import tzinfo as TZInfo
from datetime import timezone
from zoneinfo import ZoneInfo

TIMEZONE_CEST = ZoneInfo('Europe/Berlin')
TIMEZONE_UTC = timezone.utc


def current_time_tz(
    tz: TZInfo = TIMEZONE_UTC,
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