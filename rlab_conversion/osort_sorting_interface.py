import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from dateutil.tz import tzutc
from pymatreader import read_mat


class OsortSortingInterface:
    """Data interface class for OSort sorting extractor."""

    def __init__(
        self,
        sort_folder: Union[str, Path],
        metadata_file: Union[str, Path],
    ):
        """
        Parameters
        ----------
        osort_folder: str or Path
            Path to the OSort output folder
        metadata_file: str or Path
            Path to the metadata file
        sampling_rate: int
            Sampling rate of the recording
        """
        self.osort_folder = Path(sort_folder)
        self.metadata_file = Path(metadata_file)

        with self.metadata_file.open("r") as stream:
            self.electrode_config = yaml.safe_load(stream)["electrode_config"]

    def _collect_cell_file_paths(self, sort_folder):
        """
        Collect paths to all cell files in the OSort output folder.

        Parameters
        ----------
        sort_folder: Path
            Path to the OSort output folder

        Returns
        -------
        list
            List of paths to cell files
        """
        cell_files = list(sort_folder.glob("A*_cells.mat"))
        return cell_files

    def _process_cell_file(self, cell_file: Path):
        """
        Process a single cell file.

        Parameters
        ----------
        cell_file: Path
            Path to the cell file

        Returns
        -------
        dict
            Dictionary containing spike data
        """
        search_result = re.search(r"\d+", cell_file.name)
        if search_result is None:
            raise ValueError(f"Could not extract channel number from {cell_file.name}")
        channel_num = int(search_result.group())

        channel_data = read_mat(cell_file)
        processing_timestamp = channel_data["processedOn"]
        spike_data = channel_data["spikes"]
        unit_nums = spike_data.T[0]
        cluster_nums = spike_data.T[1]
        spike_times = spike_data.T[2]

        units = []
        for unit_num in np.unique(unit_nums):
            unit_mask = unit_nums == unit_num
            cluster_nums_unit = cluster_nums[unit_mask]
            spike_times_unit = spike_times[unit_mask]
            spike_datetimes_unit = [
                (
                    datetime.fromtimestamp(0, tzutc()) + timedelta(microseconds=st)
                ).timestamp()
                for st in spike_times_unit
            ]

            # TODO: Check whether or not it can have multiple clusters
            assert len(np.unique(cluster_nums_unit)) == 1
            cluster = int(np.unique(cluster_nums_unit)[0])
            units.append(
                {
                    "channel": channel_num,
                    "cluster": cluster,
                    "spike_times": spike_datetimes_unit,
                    "spike_time_raw": spike_times_unit[0],
                    "processing_timestamp": processing_timestamp,
                }
            )
        return units

    def _extract_units(self, cell_files: list[Path]):
        """
        Extract units from all cell files.

        Parameters
        ----------
        cell_files: list
            List of paths to cell files

        Returns
        -------
        list
            List of dictionaries containing spike data
        """
        units = []
        for cell_file in cell_files:
            units.extend(self._process_cell_file(cell_file))

        # Sort units by channel and cluster number
        units = sorted(units, key=lambda x: (x["channel"], x["cluster"]))
        return units

    def _get_brain_area(self, channel_num: int):
        """
        Get brain area for a given channel number.

        Parameters
        ----------
        channel_num: int
            Channel number

        Returns
        -------
        str
            Brain area
        """
        for probe_num, probe in self.electrode_config.items():
            if channel_num in probe["channels"]:
                return probe["brain_area"]
        return None

    def _add_units_to_nwbfile(self, nwbfile, units):
        """
        Add units to NWB file.

        Parameters
        ----------
        nwbfile: NWBFile
            NWB file object
        units: list
            List of dictionaries containing spike data
        """
        electrodes = nwbfile.electrodes.to_dataframe()
        nwbfile.add_unit_column("cluster", "Cluster number")
        nwbfile.add_unit_column(
            "channel", "Channel number, for convenience as info is in electrodes"
        )
        nwbfile.add_unit_column("processing_timestamp", "Processing timestamp")
        nwbfile.add_unit_column("brain_area", "Brain area")
        nwbfile.add_unit_column("brain_area_hemisphere", "Brain area hemisphere")
        nwbfile.add_unit_column("spike_time_raw", "Raw spike time")
        for unit in units:
            electrode_idx = np.searchsorted(electrodes.index, unit["channel"])
            brain_area = self._get_brain_area(unit["channel"])
            if brain_area is None:
                raise ValueError(f"Brain area not found for channel {unit['channel']}")
            nwbfile.add_unit(
                spike_times=unit["spike_times"],
                electrodes=[electrode_idx],
                channel=unit["channel"],
                cluster=unit["cluster"],
                processing_timestamp=unit["processing_timestamp"],
                brain_area=brain_area[1:],
                brain_area_hemisphere=brain_area,
                spike_time_raw=unit["spike_time_raw"],
            )
        return nwbfile

    def _maybe_init_nwb_electrodes(self, nwbfile, electrode_config):
        """
        Initialize NWB electrodes table.

        Parameters
        ----------
        nwbfile: NWBFile
            NWB file object
        electrode_config: dict
            Electrode configuration dictionary
        """
        if not hasattr(nwbfile, "devices") or not hasattr(nwbfile.devices, "Neuralynx"):
            print("Initializing devices table")
            device = nwbfile.create_device(
                name="Neuralynx",
                description="Neuralynx acquisition system",
                manufacturer="Neuralynx",
            )
        else:
            device = nwbfile.devices["Neuralynx"]

        if not hasattr(nwbfile, "electrodes") or nwbfile.electrodes is None:
            print("Initializing electrodes table")
            nwbfile.add_electrode_column("label", "Electrode label")
            for probe_num, probe in electrode_config.items():
                brain_area = probe["brain_area"]
                electrode_group = nwbfile.create_electrode_group(
                    name=f"probe-{probe_num}",
                    description=f'Electrode group for probe {probe_num}. Referred to as "channel" in the OSort output.',
                    device=device,
                    location=brain_area,
                )
                for channel_num in probe["channels"]:
                    nwbfile.add_electrode(
                        id=channel_num,
                        x=0.0,
                        y=0.0,
                        z=0.0,
                        imp=float("nan"),
                        location=brain_area,
                        filtering="unknown",
                        group=electrode_group,
                        label=f"probe-{probe_num}_channel-{channel_num}",
                    )
        return nwbfile

    def add_to_nwbfile(self, nwbfile):
        """
        Add spike sorting data to NWB file.

        Parameters
        ----------
        nwbfile: NWBFile
            NWB file object
        """

        cell_files = self._collect_cell_file_paths(self.osort_folder)
        units = self._extract_units(cell_files)

        nwbfile = self._maybe_init_nwb_electrodes(nwbfile, self.electrode_config)

        nwbfile = self._add_units_to_nwbfile(nwbfile, units)

        return nwbfile
