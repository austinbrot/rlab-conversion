import argparse as ap
from pathlib import Path

import yaml
from pynwb import NWBHDF5IO

from rlab_conversion import (
    FaceTraitBehaviorConverter,
    OsortSortingInterface,
    make_rlab_nwbfile,
)


def handle_args():
    parser = ap.ArgumentParser(
        description="Curate NWB file for a session after spike sorting."
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Path to the session directory",
    )
    parser.add_argument(
        "--behavior-file",
        type=Path,
        required=True,
        help="Path to the behavior file",
    )
    parser.add_argument(
        "--events-file",
        type=Path,
        required=False,
        help="Path to the events file",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        required=False,
        help="Path to the metadata file",
    )
    parser.add_argument(
        "--sort-dir",
        type=Path,
        required=False,
        help="Path to the sorting folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NWB file",
    )
    return parser.parse_args()


def main():
    #######################
    # Handle arguments
    #######################
    args = handle_args()
    session_dir = args.session_dir

    if not session_dir.exists() or not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory not found: {str(session_dir)}")

    metadata_file = (
        args.metadata_file
        if args.metadata_file is not None
        else session_dir / "metadata.yml"
    )
    if not metadata_file.exists() or not metadata_file.is_file():
        raise FileNotFoundError(f"Metadata file not found: {str(metadata_file)}")

    with metadata_file.open("r") as stream:
        metadata = yaml.safe_load(stream)

    events_file = (
        args.events_file
        if args.events_file is not None
        else session_dir / "raw" / "events.csv"
    )
    if not events_file.exists() or not events_file.is_file():
        raise FileNotFoundError(f"Events file not found: {str(events_file)}")
    
    sort_dir = (
        args.sort_dir
        if args.sort_dir is not None
        else session_dir / "sort" / "final"
    )
    if not sort_dir.exists() or not sort_dir.is_dir():
        raise FileNotFoundError(f"Sorting directory not found: {str(sort_dir)}")


    #######################
    # Create NWB file
    #######################
    nwbfile = make_rlab_nwbfile(session_dir)

    behavior_interface = FaceTraitBehaviorConverter(
        events_file=events_file,
        behavior_file=args.behavior_file,
        metadata_file=metadata_file,
    )
    behavior_interface.add_to_nwbfile(nwbfile)

    sorting_interface = OsortSortingInterface(
        sort_folder=sort_dir, metadata_file=metadata_file
    )
    sorting_interface.add_to_nwbfile(nwbfile)

    nwbfile_path = (
        session_dir
        / f"sub-{metadata['subject_id']}_ses-{metadata['session_id']}_task-faceTraitRating_ieeg.nwb"
    )
    if nwbfile_path.exists() and not args.overwrite:
        raise FileExistsError(f"NWB file already exists: {nwbfile_path}")
    with NWBHDF5IO(path=nwbfile_path, mode="w") as io:
        io.write(container=nwbfile)


if __name__ == "__main__":
    main()
