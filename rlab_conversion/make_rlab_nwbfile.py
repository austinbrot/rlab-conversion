from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pandas as pd
import yaml
from dateutil.tz import tzutc
from pynwb import NWBFile
from pynwb.file import Subject

from rlab_conversion.utils import SESSION_METADATA_FILENAME


def _load_metadata_file(metadata_file: Path):
    """
    Load the metadata file for the session.

    Parameters
    ----------
    metadata_file: Path
        Path to the metadata file

    Returns
    -------
    dict
        Metadata for the session
    """
    with metadata_file.open("r") as stream:
        metadata = yaml.safe_load(stream)
        return metadata


def _extract_start_time_from_events(events_path: Path) -> datetime:
    """
    Extract the start time of the session from the events file.

    Parameters
    ----------
    events_path: Path
        Path to the .nev file containing the events

    Returns
    -------
    datetime
        Start time of the session
    """
    # Extract the start time from the events file
    ttls = pd.read_csv(str(events_path), header=None)
    ttls.rename(columns={0: "timestamps", 1: "event"}, inplace=True)
    start_time = ttls.loc[ttls["event"] == 1, "timestamps"].values[0]
    sess_start_datetime = datetime.fromtimestamp(0, tzutc()) + timedelta(
        microseconds=start_time
    )
    print("Session start time is:", sess_start_datetime)
    return sess_start_datetime


def _safe_handle_optional_path(
    path: Optional[Path],
    default_path: Path,
    error_message_template: str,
) -> Path:
    """
    Safely handle an optional path.

    Parameters
    ----------
    path: Optional[Path]
        Path to handle
    default_path: Path
        Default path to use if path is None
    error_message_template: str
        Error message template to use if path (or default path) does not exist. Must accept a Path object.
    """
    if path is None:
        path = default_path
    if not path.exists():
        raise FileNotFoundError(error_message_template.format(str(path)))
    return path


def make_rlab_nwbfile(
    session_dir: Path,
    metadata_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
) -> NWBFile:
    """
    Create an NWB file for a session in the Rlab.

    Parameters
    ----------
    metadata_path: Path
        Path to the metadata file for the session
    events_path: Path
        Path to the events file for the session

    Returns
    -------
    NWBFile
        NWB file for the session
    """
    # Points to Nlx events from .csv file
    events_path = _safe_handle_optional_path(
        events_path,
        session_dir / "raw" / "Events.csv",
        "Events file not found: {}",
    )

    # points to session-specific metadata from .yml file
    metadata_path = _safe_handle_optional_path(
        metadata_path,
        session_dir / SESSION_METADATA_FILENAME,
        "Metadata file not found: {}",
    )
    session_start_time = _extract_start_time_from_events(events_path)
    metadata = _load_metadata_file(metadata_path)

    subject = Subject(
        subject_id=metadata["subject_id"],
        species="Homo sapiens",
    )

    # Default file construction for epilepsy subject
    nwbfile = NWBFile(
        session_description=metadata["session_description"],
        identifier=str(uuid4()),
        session_start_time=session_start_time,
        experimenter=metadata["experimenter"],
        lab=metadata["lab"],
        institution=metadata["institution"],
        session_id=str(metadata["session_id"]),
        subject=subject,
    )

    return nwbfile
