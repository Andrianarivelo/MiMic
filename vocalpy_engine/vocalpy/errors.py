# -*- coding: utf-8 -*-
"""Shared exception types for VocalPy."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"


class VocalPyError(Exception):
    """Base exception for package-specific failures."""


class ValidationError(VocalPyError, ValueError):
    """Raised when user-provided values are invalid."""


class ConfigurationError(VocalPyError):
    """Raised when the configured pipeline or model surface is invalid."""


class RecordingStateError(VocalPyError):
    """Raised when a recording does not have the required state for an action."""


class InputPathError(ValidationError):
    """Raised when the requested audio input path is missing or invalid."""


class SerializationError(VocalPyError):
    """Raised when a serialized VocalPy object is malformed or incompatible."""
