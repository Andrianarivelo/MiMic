# -*- coding: utf-8 -*-
"""VocalPy visualization helpers."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import math
import os

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D

from vocalpy.errors import ValidationError
from vocalpy.utils.io import load_recording_data


SUPPORTED_PLOT_TYPES = {"group", "individual"}
SUPPORTED_DATANAMES = {
    "raster",
    "rate",
    "min_freq",
    "max_freq",
    "avg_freq",
    "duration",
    "bandwidth",
    "min_intensity",
    "max_intensity",
    "avg_intensity",
    "area",
}
NONNEGATIVE_DATANAMES = {"rate", "duration", "bandwidth", "area"}


def _recording_name_from_path(recording_path):
    output_dir = os.path.basename(os.path.dirname(recording_path))
    return output_dir[:-8] if output_dir.endswith("_outputs") else output_dir


def _resolve_group_names(list_of_groups, group_names=None):
    if group_names is not None:
        resolved_names = np.asarray(group_names)
        if resolved_names.shape[0] != len(list_of_groups):
            raise ValidationError("group_names must match the number of groups")
        return resolved_names

    derived_names = []
    for index, group in enumerate(list_of_groups, start=1):
        if len(group) == 1:
            derived_names.append(_recording_name_from_path(group[0]))
        else:
            derived_names.append(f"group_{index}")

    return np.asarray(derived_names)


def _resolve_duration_minutes(recording_data):
    duration_seconds = getattr(recording_data, "recording_duration", None)
    if duration_seconds is None:
        duration_seconds = getattr(getattr(recording_data, "audio", None), "audio_duration", None)

    if duration_seconds is None:
        raise ValidationError("recording data does not include a duration")

    return duration_seconds / 60


def _calculate_bins(duration_minutes, bin_size):
    if bin_size <= 0:
        raise ValidationError("bin_size must be greater than zero")
    return max(1, int(math.ceil(duration_minutes / bin_size)))


def _coerce_group_label(recording_data, recording_path):
    group_name = getattr(recording_data, "group_name", None)
    if group_name and group_name != "not set":
        return group_name
    return _recording_name_from_path(recording_path)


class _BaseViz(object):
    """Shared dataframe and plotting helpers."""

    def __init__(self, label, recording_df, bin_size, bins, duration_minutes):
        self._label = label
        self._recording_df = recording_df.sort_values(["bin_number", "start"], na_position="last").copy()
        self._bin_size = bin_size
        self._bins = bins
        self._duration = duration_minutes

    @property
    def label(self):
        return self._label

    def _full_bin_numbers(self):
        return list(range(1, self._bins + 1))

    def _recording_df_with_plot_bins(self):
        dataframe = self._recording_df.copy()
        if dataframe.empty:
            dataframe["plot_bin"] = pd.Series(dtype=int)
            return dataframe

        dataframe["plot_bin"] = np.floor((dataframe["bin_number"].to_numpy() - 1) / self._bin_size).astype(int) + 1
        return dataframe

    def _build_binned_dataframe(self, dataname):
        if dataname == "rate":
            counts = pd.Series(0.0, index=self._full_bin_numbers(), dtype=float)
            if not self._recording_df.empty:
                binned_counts = (
                    self._recording_df_with_plot_bins()["plot_bin"].value_counts().sort_index().astype(float)
                )
                counts.loc[binned_counts.index.to_list()] = binned_counts.to_numpy()
            return pd.DataFrame([counts.to_numpy()], columns=counts.index)

        binned_values = {}
        grouped_values = pd.Series(dtype=object)
        if not self._recording_df.empty:
            grouped_values = self._recording_df_with_plot_bins().groupby("plot_bin")[dataname].apply(list)

        for bin_number in self._full_bin_numbers():
            binned_values[bin_number] = pd.Series(grouped_values.get(bin_number, []), dtype=float)

        data_values = pd.DataFrame(binned_values)
        if data_values.empty:
            data_values = pd.DataFrame({bin_number: pd.Series([np.nan]) for bin_number in self._full_bin_numbers()})

        data_values.index = np.arange(1, len(data_values) + 1)
        data_values.columns = self._full_bin_numbers()
        return data_values

    def _configure_value_axis(self, ax, dataname):
        ax.set_ylabel(dataname)
        ax.set_xlabel("Bin")
        if dataname in NONNEGATIVE_DATANAMES:
            ax.set_ylim(bottom=0)

    def _configure_bin_axis(self, ax, bins=None):
        total_bins = self._bins if bins is None else bins
        bin_positions = list(range(total_bins))
        ax.set_xlim(-0.5, total_bins - 0.5)
        ax.set_xticks(bin_positions)
        ax.set_xticklabels(self._full_bin_numbers()[:total_bins])

    def get_datapoints(self, dataname):
        return self._build_binned_dataframe(dataname)

    def rugplot(self, dataname="start"):
        data_values = self._recording_df[[dataname]] / 60

        sns.set(style="white", palette="muted", color_codes=True)
        _, ax = plt.subplots(1, 1, figsize=(12, 2), dpi=300)

        if not data_values.empty:
            sns.rugplot(data_values.squeeze(axis=1), height=1, color="coral", alpha=0.5)

        ax.set_title(self.label)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(dataname)
        ax.set_yticks([])
        plt.tight_layout()

        return data_values

    def pointplot(self, dataname="avg_freq"):
        data_values = self.get_datapoints(dataname)

        sns.set(style="whitegrid", palette="muted", color_codes=True)
        _, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=300)
        sns.pointplot(data=data_values, color="coral", capsize=0.1, ax=ax)

        self._configure_value_axis(ax, dataname)
        self._configure_bin_axis(ax)
        plt.tight_layout()

        return data_values

    def violinplot(self, dataname="avg_freq", inner="box"):
        data_values = self.get_datapoints(dataname)

        sns.set(style="whitegrid", palette="muted", color_codes=True)
        _, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=300)
        sns.violinplot(data=data_values, color="coral", inner=inner, ax=ax)

        self._configure_value_axis(ax, dataname)
        self._configure_bin_axis(ax)
        plt.tight_layout()

        return data_values


class Viz(object):
    """Visualization object for plotting data from one or more recordings."""

    def __init__(self, list_of_groups, group_names=None, bin_size=1):
        if len(list_of_groups) < 1:
            raise ValidationError("please provide at least one recording for analysis.")

        self._list_of_groups = list_of_groups
        self._bin_size = bin_size
        self._group_names = _resolve_group_names(list_of_groups, group_names)
        self._number_of_groups = self._group_names.shape[0]
        self._list_of_viz = None
        self._list_of_group_viz = None

        self.create_viz_for_each_recording()
        self.create_group_viz()

    @property
    def list_of_groups(self):
        return self._list_of_groups

    def generate_group_names(self):
        return _resolve_group_names(self.list_of_groups)

    def combine_viz_dataframes(self, list_of_viz=None):
        if list_of_viz is None or len(list_of_viz) == 0:
            raise ValidationError("please provide a non-empty list_of_viz")

        return pd.concat([viz._recording_df for viz in list_of_viz], axis=0).sort_values(
            ["bin_number", "start"], na_position="last"
        )

    def combine_viz(self, list_of_viz, groupname):
        if not list_of_viz:
            return None
        return GroupViz(list_of_viz, groupname, self._bin_size)

    def create_viz_for_each_recording(self):
        self._list_of_viz = [
            [SingleViz(recording_path=recording_path, bin_size=self._bin_size) for recording_path in group]
            for group in self._list_of_groups
        ]
        return 0

    def create_group_viz(self):
        self._list_of_group_viz = [
            self.combine_viz(group, self._group_names[index]) for index, group in enumerate(self._list_of_viz)
        ]
        return 0

    def plot(self, plot_type="group", dataname="avg_freq"):
        if plot_type not in SUPPORTED_PLOT_TYPES:
            raise ValidationError('type must be "group" or "individual".')

        if dataname not in SUPPORTED_DATANAMES:
            raise ValidationError(
                "possible datapoints are: "
                "raster, rate, min_freq, max_freq, avg_freq, duration, "
                "bandwidth, min_intensity, max_intensity, avg_intensity, area."
            )

        if plot_type == "group":
            self.plot_group(dataname)
        else:
            self.plot_individual(dataname)

        return 0

    def plot_individual(self, dataname):
        for group in self._list_of_viz:
            for viz in group:
                if dataname == "raster":
                    viz.rugplot()
                else:
                    viz.pointplot(dataname=dataname)
        return 0

    def plot_group(self, dataname):
        if dataname == "raster":
            for group in self._list_of_group_viz:
                if group is not None:
                    group.rugplot()
        else:
            self.group_pointplot(dataname=dataname)
        return 0

    def group_pointplot(self, dataname="avg_freq"):
        sns.set(style="whitegrid", palette="muted", color_codes=True)
        _, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=150)
        cmap = sns.color_palette()
        labels = []
        custom_lines = []
        max_bins = 1

        for index, group in enumerate(self._list_of_group_viz):
            if group is None:
                continue

            data_values = group.get_datapoints(dataname)
            sns.pointplot(data=data_values, color=cmap[index], capsize=0.1, ax=ax)
            custom_lines.append(Line2D([0], [0], color=cmap[index], lw=2))
            labels.append(group.label)
            max_bins = max(max_bins, group._bins)

        if labels:
            ax.legend(custom_lines, labels, fontsize=16)

        if self._list_of_group_viz:
            self._list_of_group_viz[0]._configure_value_axis(ax, dataname)
        ax.tick_params(labelsize=16)
        ax.set_ylabel(dataname, fontsize=16)
        ax.set_xlabel("Bin", fontsize=16)
        ax.set_xlim(-0.5, max_bins - 0.5)
        ax.set_xticks(list(range(max_bins)))
        ax.set_xticklabels(list(range(1, max_bins + 1)))
        plt.tight_layout()


class GroupViz(_BaseViz):
    """Aggregate visualization object for a group of recordings."""

    def __init__(self, list_of_viz, group_name, bin_size):
        self._recording_path = None
        self._list_of_vocals = None
        self._recording_data = SimpleNamespace(
            group_name=group_name,
            _list_of_vocals=SimpleNamespace(
                number_of_vocals=sum(viz._recording_data._list_of_vocals.number_of_vocals for viz in list_of_viz)
            ),
        )
        combined_df = pd.concat([viz._recording_df for viz in list_of_viz], axis=0)
        duration_minutes = max(viz._duration for viz in list_of_viz)
        bins = max(viz._bins for viz in list_of_viz)
        super().__init__(group_name, combined_df, bin_size, bins, duration_minutes)


class SingleViz(_BaseViz):
    """Visualization object for a single recording."""

    def __init__(self, recording_path=None, bin_size=1):
        if recording_path is None:
            raise ValidationError("please provide a recording_path")

        self._recording_path = recording_path
        self._recording_data = load_recording_data(self._recording_path)
        self._list_of_vocals = self.create_list_of_vocals()
        self._bin_size = bin_size
        self._duration = _resolve_duration_minutes(self._recording_data)
        self._bins = _calculate_bins(self._duration, self._bin_size)
        recording_df = self.create_object_dataframe()
        super().__init__(
            _coerce_group_label(self._recording_data, self._recording_path),
            recording_df,
            self._bin_size,
            self._bins,
            self._duration,
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n "
            f"recording path: {self._recording_path}\n "
            f"recording duration: {self._duration:.2f}\n "
            f"bin size: {self._bin_size}\n "
            f"bins: {self._bins}\n "
            f"vocals in recording: {self._recording_data._list_of_vocals.number_of_vocals}"
        )

    def create_list_of_vocals(self):
        return self._recording_data._list_of_vocals

    def create_object_dataframe(self):
        columns = [
            "bin_number",
            "start",
            "end",
            "duration",
            "interval",
            "min_freq",
            "max_freq",
            "avg_freq",
            "bandwidth",
            "min_intensity",
            "max_intensity",
            "avg_intensity",
            "bg_intensity",
            "area",
            "centroid_y",
        ]
        records = []

        for vocal in self._list_of_vocals.vocals_in_recording:
            records.append(
                {
                    "bin_number": vocal.bin_number,
                    "start": vocal.start,
                    "end": vocal.end,
                    "duration": vocal.duration,
                    "interval": vocal.interval,
                    "min_freq": vocal.min_freq,
                    "max_freq": vocal.max_freq,
                    "avg_freq": vocal.avg_freq,
                    "bandwidth": vocal.bandwidth,
                    "min_intensity": vocal.min_intensity,
                    "max_intensity": vocal.max_intensity,
                    "avg_intensity": vocal.avg_intensity,
                    "bg_intensity": vocal.bg_intensity,
                    "area": vocal.area,
                    "centroid_y": vocal.centroid[0] if vocal.centroid is not None else None,
                }
            )

        recording_df = pd.DataFrame.from_records(records, columns=columns)
        recording_df.index = np.arange(1, len(recording_df) + 1)
        return recording_df
