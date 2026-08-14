# -*- coding: utf-8 -*-
"""Regression tests for modern pandas compatibility."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from types import SimpleNamespace

from vocalpy.modules.viz import SingleViz
from vocalpy.utils.misc import create_dataframe_from_list_of_vocals


def make_vocal(bin_number, start, end):
    return SimpleNamespace(
        bin_number=bin_number,
        start=start,
        end=end,
        duration=(end - start) * 1000,
        interval=0.0,
        min_freq=40,
        max_freq=80,
        avg_freq=60,
        bandwidth=40,
        min_intensity=-20,
        max_intensity=-10,
        avg_intensity=-15,
        bg_intensity=-30,
        area=10,
        centroid=(3, 4),
        top1="flat",
        top2="up_fm",
    )


def test_create_dataframe_from_list_of_vocals_builds_rows_without_dataframe_append():
    list_of_vocals = SimpleNamespace(vocals_in_recording=[make_vocal(2, 1.5, 1.7), make_vocal(1, 0.2, 0.4)])

    dataframe = create_dataframe_from_list_of_vocals(list_of_vocals)

    assert dataframe["start(s)"].tolist() == [0.2, 1.5]
    assert dataframe["class_top1"].tolist() == ["flat", "flat"]


def test_single_viz_dataframe_builder_works_on_modern_pandas():
    single_viz = SingleViz.__new__(SingleViz)
    single_viz._list_of_vocals = SimpleNamespace(vocals_in_recording=[make_vocal(1, 0.2, 0.4)])

    dataframe = single_viz.create_object_dataframe()

    assert dataframe.index.tolist() == [1]
    assert dataframe.loc[1, "avg_freq"] == 60
