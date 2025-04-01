from datetime import datetime, timedelta

from dateutil.tz import tzutc


def nlx_timestamp_to_float(timestamp: int) -> float:
    """
    Convert a Neuralynx timestamp to a float.

    Parameters
    ----------
    timestamp: int
        Neuralynx timestamp

    Returns
    -------
    float
        Timestamp as a float
    """
    return (
        datetime.fromtimestamp(0, tzutc()) + timedelta(microseconds=timestamp)
    ).timestamp()


SESSION_METADATA_FILENAME = "metadata.yml"
