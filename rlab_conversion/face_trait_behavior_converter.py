from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import yaml

from rlab_conversion.utils import nlx_timestamp_to_float


def _handle_provided_path(
    file_path: Union[str, Path],
    expected_suffix: Optional[list[str]] = None,
    is_file: bool = True,
):
    """
    Handle the provided file path (str or Path) and return a validated Path.
    """
    file_path = Path(file_path)
    assert (
        file_path.exists()
    ), f"{'File' if is_file else 'Directory'} not found: {file_path}"
    if is_file:
        assert file_path.is_file(), f"Invalid file: {file_path}"
    else:
        assert file_path.is_dir(), f"Invalid directory: {file_path}"
    if expected_suffix is not None:
        assert (
            file_path.suffix in expected_suffix
        ), f"Invalid file type: {file_path}. Must be one of {expected_suffix}"
    return file_path


def _safe_load_yaml(file_path: Path):
    with file_path.open("r") as stream:
        metadata = yaml.safe_load(stream)
        return metadata


def _load_events_file(events_file: Path):
    if events_file.suffix == ".csv":
        events = pd.read_csv(events_file, header=None)
        events.columns = ["timestamp", "ttl"]
        events = events[events["ttl"] != 0]
        events.loc[:, "timestamp"] = events["timestamp"].map(nlx_timestamp_to_float)
    else:
        raise NotImplementedError("Only CSV files are supported for now.")
    return events.reset_index(drop=True)


def _load_behavior_file(behavior_file: Path):
    behavior = pd.read_csv(behavior_file)

    # drop if is fixation cross immediately following fixation cross
    behavior = behavior.drop(
        behavior[
            (behavior["event"] == "fixation_cross")
            & (behavior["event"].shift(-1) == "fixation_cross")
        ].index
    )

    # add context column if not present
    if "context" not in behavior.columns:
        behavior["context"] = "none"

    return behavior.reset_index(drop=True)


class FaceTraitBehaviorConverter:
    def __init__(
        self,
        events_file: Union[str, Path],
        behavior_file: Union[str, Path],
        metadata_file: Union[str, Path],
    ) -> None:
        self.events_file = _handle_provided_path(events_file, expected_suffix=[".csv"])

        self.behavior_file = _handle_provided_path(
            behavior_file, expected_suffix=[".csv"]
        )

        self.metadata_file = _handle_provided_path(
            metadata_file, expected_suffix=[".yml", ".yaml"]
        )
        self.trial_metadata = _safe_load_yaml(self.metadata_file)

    def _align_events_to_behavior(self, behavior_df, events_df):
        """
        Finds the contiguous subset of events_df that aligns with behavior_df.

        Parameters
        ----------
        behavior_df: pd.DataFrame
            DataFrame containing the behavior data
        events_df: pd.DataFrame
            DataFrame containing the events data

        Returns
        -------
        pd.DataFrame
            Subset of events_df that aligns with behavior_df
        """
        unique_ttls = behavior_df["ttl"].unique()
        events_df = events_df[events_df["ttl"].isin(unique_ttls)].reset_index(drop=True)

        behavior_ttls = behavior_df["ttl"].values
        event_ttls = events_df["ttl"].values

        import pdb

        pdb.set_trace()

        candidate_start_idxs = np.where(event_ttls == behavior_ttls[0])[0]
        for start_idx in candidate_start_idxs:
            # How many elements can we compare from this start index?
            length_to_compare = min(len(behavior_ttls), len(event_ttls) - start_idx)

            # Compare slices of whichever length is valid for both arrays
            if (
                event_ttls[start_idx : start_idx + length_to_compare]
                == behavior_ttls[:length_to_compare]
            ).all():
                # Return the corresponding rows from events_df
                events_df = events_df.iloc[
                    start_idx : start_idx + length_to_compare
                ].reset_index(drop=True)
                behavior_df = behavior_df.iloc[:length_to_compare].reset_index(
                    drop=True
                )
                return events_df, behavior_df

        raise ValueError(
            "No matching TTL sequence found between behavior and events data."
        )

    def _merge_behavior_and_events(self, behavior_df, events_df):
        # check ttls align for automatic merge
        assert "ttl" in behavior_df.columns, "TTL column not found in behavior data."
        assert "ttl" in events_df.columns, "TTL column not found in events data."

        events_df, behavior_df = self._align_events_to_behavior(behavior_df, events_df)

        assert (
            behavior_df["ttl"].values == events_df["ttl"].values
        ).all(), "TTL codes do not match between behavior and events. Inspect the data and try again."

        # merge the two dataframes on the TTL column
        merged_df = events_df.join(
            behavior_df.drop(columns=["ttl"]),
            how="left",
        )
        return merged_df

    def _check_uniform_values(self, df, columns, trial):
        for col in columns:
            if not len(df[col].unique()) == 1:
                print(df)
            assert (
                len(df[col].unique()) == 1
            ), f"Multiple values found in column {col} during trial {trial}: {df[col].unique()}"

    def _process_context_trials(self, events):
        raise NotImplementedError("Context trials not yet implemented.")

    def _process_rating_trials(self, events):
        events = events[
            ~events["event"].isin(
                [
                    "face_rating_begin",
                    "face_rating_end",
                    "test_retest_begin",
                    "test_retest_end",
                    "training_begin",
                    "training_end",
                ]
            )
        ].reset_index(drop=True)
        trials = []
        for trial_num, trial_events in events.groupby("trial_number"):
            # skip pause block
            if "experiment_paused" in trial_events["event"].values:
                continue_time = trial_events.loc[
                    trial_events["event"] == "experiment_continue", "timestamp"
                ].values[0]
                trial_events = trial_events[trial_events["timestamp"] > continue_time]

            # skip trials with no events
            if len(trial_events) == 0:
                print(f"Skipping trial {trial_num} with no events.")
                continue

            # extract response and response time
            response_mask = trial_events["event"].str.contains(
                "response|missed_trial", regex=True
            )
            if response_mask.any():
                response_time = trial_events.loc[response_mask, "timestamp"].values[0]
                response = trial_events.loc[response_mask, "event"].values[0]
                if not response == "missed_trial":
                    response = response.split("_")[-1]
            else:
                response_time = np.nan
                response = "none"

            self._check_uniform_values(
                trial_events, ["task", "condition", "context", "training"], trial_num
            )

            trials.append(
                {
                    "start_time": trial_events["timestamp"].values[0],
                    "task": trial_events["task"].values[0],
                    "condition": trial_events["condition"].values[0],
                    "context": trial_events["context"].values[0],
                    "training": trial_events["training"].values[0],
                    "image": trial_events.loc[
                        trial_events["event"] == "image_on", "image"
                    ].values[0],
                    "context_time": np.nan,
                    "fixation_time": trial_events.loc[
                        trial_events["event"] == "fixation_cross", "timestamp"
                    ].values[0],
                    "image_on_time": trial_events.loc[
                        trial_events["event"] == "image_on", "timestamp"
                    ].values[0],
                    "image_and_options_time": trial_events.loc[
                        trial_events["event"] == "image_and_options", "timestamp"
                    ].values[0],
                    "response": response,
                    "response_time": response_time,
                }
            )

        return pd.DataFrame(trials)

    def _process_nback_trials(self, events):
        events = events[
            ~events["event"].isin(["n_back_begin", "n_back_end"])
        ].reset_index(drop=True)
        trials = []
        for trial_num, trial_events in events.groupby("trial_number"):
            # skip pause block
            if "experiment_paused" in trial_events["event"].values:
                continue_time = trial_events.loc[
                    trial_events["event"] == "experiment_continue", "timestamp"
                ].values[0]
                trial_events = trial_events[trial_events["timestamp"] > continue_time]

            # skip trials with no events
            if len(trial_events) == 0:
                print(f"Skipping trial {trial_num} with no events.")
                continue

            response_mask = trial_events["event"].str.contains(
                "response|missed_trial", regex=True
            )
            if response_mask.any():
                response_time = trial_events.loc[response_mask, "timestamp"].values[0]
                response = trial_events.loc[response_mask, "event"].values[0]
                response = response.split("_")[-1]
            else:
                response_time = np.nan
                response = "none"

            self._check_uniform_values(
                trial_events, ["task", "condition", "context", "training"], trial_num
            )

            trials.append(
                {
                    "start_time": trial_events["timestamp"].values[0],
                    "task": trial_events["task"].values[0],
                    "condition": trial_events["condition"].values[0],
                    "context": trial_events["context"].values[0],
                    "training": trial_events["training"].values[0],
                    "image": trial_events.loc[
                        trial_events["event"] == "image_on", "image"
                    ].values[0],
                    "context_time": np.nan,
                    "fixation_time": trial_events.loc[
                        trial_events["event"] == "fixation_cross", "timestamp"
                    ].values[0],
                    "image_on_time": trial_events.loc[
                        trial_events["event"] == "image_on", "timestamp"
                    ].values[0],
                    "image_and_options_time": np.nan,
                    "response": response,
                    "response_time": response_time,
                }
            )

        return pd.DataFrame(trials)

    def _process_task_trials(self, events, task):
        if task == "face_rating" or task == "test_retest":
            return self._process_rating_trials(events)
        elif task == "n_back":
            return self._process_nback_trials(events)
        elif task == "context_rating":
            return self._process_context_trials(events)
        else:
            raise ValueError(f"Invalid task type: {task}")

    def _process_trials(self, events):
        events = events[events["event"] != "none"]

        trials = []
        for task, task_events in events.groupby("task"):
            if task != "none":
                task_trials = self._process_task_trials(task_events, task)
                trials.append(task_trials)

        # sort tasks by start time
        trials = sorted(trials, key=lambda x: x.loc[0, "start_time"])

        return pd.concat(trials, ignore_index=True)

    def add_to_nwbfile(self, nwbfile):
        # Load Nlx events (as .csv) and behavior file written by task
        behavior = _load_behavior_file(self.behavior_file)
        events = _load_events_file(self.events_file)

        # Merge behavior and events dataframes on TTL column
        behavior_events = self._merge_behavior_and_events(behavior, events)

        # process merge trial data into final DataFrame
        trials = self._process_trials(behavior_events)
        trials["stop_time"] = trials["start_time"].shift(
            -1, fill_value=events.iloc[-1]["timestamp"]
        )

        # add trial columns so trial df has structure in NWB file
        nwbfile.add_trial_column(
            name="image", description="Image shown during the trial"
        )
        nwbfile.add_trial_column(
            name="context_time", description="Time the context was shown in seconds"
        )
        nwbfile.add_trial_column(
            name="fixation_time",
            description="Time the fixation cross appears on the screen in seconds",
        )
        nwbfile.add_trial_column(
            name="image_on_time",
            description="Time the image appears on the screen in seconds",
        )
        nwbfile.add_trial_column(
            name="image_and_options_time",
            description="Time the image and options appear on the screen in seconds",
        )
        nwbfile.add_trial_column(
            name="response", description="Patient's response to the image"
        )
        nwbfile.add_trial_column(
            name="response_time", description="Time of the response in seconds"
        )
        nwbfile.add_trial_column(
            name="condition",
            description='Task condition during the trial. One of "warm", "competent", "youthful", "femenine", "0_back", "1_back", or "none".',
        )
        nwbfile.add_trial_column(
            name="context",
            description="Context of the trial. One of 'positive', 'negative', or 'none'.",
        )
        nwbfile.add_trial_column(name="task", description="Task type")
        nwbfile.add_trial_column(
            name="training",
            description="Whether the trial is part of the training session",
        )

        # add trials to nwbfile as rows
        for _, trial in trials.iterrows():
            nwbfile.add_trial(
                start_time=trial["start_time"],
                stop_time=trial["stop_time"],
                image=trial["image"],
                context_time=trial["context_time"],
                fixation_time=trial["fixation_time"],
                image_on_time=trial["image_on_time"],
                image_and_options_time=trial["image_and_options_time"],
                response=trial["response"],
                response_time=trial["response_time"],
                condition=trial["condition"],
                context=trial["context"],
                task=trial["task"],
                training=trial["training"],
            )
            
        return nwbfile

