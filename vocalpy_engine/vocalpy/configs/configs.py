# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"


from os import cpu_count
from os.path import join, dirname

from vocalpy.errors import ConfigurationError
from vocalpy.utils.io import read_yaml, write_yaml


def read_default_parameters(args):
    """
    Reads the default YAML configuration and pipeline parameters files

    Parameters
    ----------
    args : dict
        Struct with arguments that customizes the execution with parameters provided by the user

    """
    default_params_pipeline = read_yaml(join(dirname(__file__), "pipelines_parameters.yml"))
    default_params_run = read_yaml(join(dirname(__file__), "run_parameters.yml"))

    try:
        default_params_pipeline = default_params_pipeline["pipelines"][args.animal.lower()]
    except KeyError as ke:
        raise ConfigurationError(f"could not find animal pipeline with name {ke}") from ke

    default_params = {"animal": args.animal.lower(), **default_params_run, **default_params_pipeline}
    return default_params


def combine_default_and_user_parameters(default_params, args):
    """
    Combines user provided parameters with the default YAML configuration and pipeline parameters

    Parameters
    ----------
    default_params : dict
        YAML dictionary with configuration and parameters definitions
    args : dict
        Struct with arguments that customizes the execution with parameters provided by the user

    """
    # -- update default thread count
    if default_params["threads"] == -1:
        default_params["threads"] = cpu_count() // 2 if cpu_count() is not None else 2

    # -- iterate over user defined parameters
    for arg in vars(args):
        # -- overwrite dict with updated parameter
        if arg in default_params.keys():
            arg_value = getattr(args, arg)
            if arg_value in ["default", -1]:
                continue
            default_params[arg] = arg_value
    return default_params


def load_user_parameters(args):
    """
    Combines user provided parameters with the default YAML configuration and pipeline parameters

    Parameters
    ----------
    args : dict
        Dictionary with arguments that customizes the execution with parameters provided by the user

    Returns
    -------
    user_parameters : dict
        Dictionary with execution and pipeline parameters
    """
    default_params = read_default_parameters(args)
    user_parameters = combine_default_and_user_parameters(default_params, args)
    return user_parameters


def write_user_parameters(data, output_path):
    """
    Writes a YAML file with analysis parameters in the output path for reproducibility

    Parameters
    ----------
    data : dict
        Dictionary with arguments that customizes the execution with parameters provided by the user
    output_path : str
        Path to write YAML file
    """
    write_yaml(data, join(output_path, "parameters.yml"))
