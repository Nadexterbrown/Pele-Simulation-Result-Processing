"""
Custom exceptions for the Pele processing system.

Provides domain-specific exceptions that enable better error handling
and debugging throughout the system.
"""
from typing import Optional, Any, Dict


class PeleProcessingError(Exception):
    """Base exception for all Pele processing errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {detail_str})"
        return self.message


# =============================================================================
# Data Layer Exceptions
# =============================================================================

class DataError(PeleProcessingError):
    """Base exception for data-related errors."""
    pass


class DataLoadError(DataError):
    """Exception raised when dataset loading fails."""

    def __init__(self, path: str, reason: str):
        super().__init__(
            f"Failed to load dataset from '{path}': {reason}",
            {"path": path, "reason": reason}
        )
        self.path = path
        self.reason = reason


class DataExtractionError(DataError):
    """Exception raised when data extraction fails."""

    def __init__(self, field_name: str, location: Optional[float] = None, reason: str = ""):
        details = {"field": field_name}
        if location is not None:
            details["location"] = location
        if reason:
            details["reason"] = reason

        message = f"Failed to extract field '{field_name}'"
        if location is not None:
            message += f" at location {location}"
        if reason:
            message += f": {reason}"

        super().__init__(message, details)
        self.field_name = field_name
        self.location = location


class ValidationError(DataError):
    """Exception raised when data validation fails."""

    def __init__(self, validation_type: str, reason: str,
                 failed_values: Optional[Dict[str, Any]] = None):
        details = {"validation_type": validation_type, "reason": reason}
        if failed_values:
            details.update(failed_values)

        super().__init__(
            f"Validation failed for {validation_type}: {reason}",
            details
        )
        self.validation_type = validation_type


class FieldNotFoundError(DataError):
    """Exception raised when required field is not found in dataset."""

    def __init__(self, field_name: str, available_fields: Optional[list] = None):
        details = {"requested_field": field_name}
        if available_fields:
            details["available_fields"] = available_fields

        message = f"Field '{field_name}' not found in dataset"
        if available_fields:
            message += f". Available fields: {available_fields}"

        super().__init__(message, details)
        self.field_name = field_name
        self.available_fields = available_fields


# =============================================================================
# Analysis Layer Exceptions
# =============================================================================

class AnalysisError(PeleProcessingError):
    """Base exception for analysis-related errors."""
    pass


class WaveNotFoundError(AnalysisError):
    """Exception raised when wave front cannot be detected."""

    def __init__(self, wave_type: str, criteria: str = ""):
        details = {"wave_type": wave_type}
        if criteria:
            details["criteria"] = criteria

        message = f"Could not detect {wave_type} wave"
        if criteria:
            message += f" using criteria: {criteria}"

        super().__init__(message, details)
        self.wave_type = wave_type


class FlameAnalysisError(AnalysisError):
    """Exception raised during flame analysis."""

    def __init__(self, analysis_type: str, reason: str):
        super().__init__(
            f"Flame {analysis_type} analysis failed: {reason}",
            {"analysis_type": analysis_type, "reason": reason}
        )
        self.analysis_type = analysis_type


class ShockAnalysisError(AnalysisError):
    """Exception raised during shock analysis."""

    def __init__(self, analysis_type: str, reason: str):
        super().__init__(
            f"Shock {analysis_type} analysis failed: {reason}",
            {"analysis_type": analysis_type, "reason": reason}
        )
        self.analysis_type = analysis_type


class BurnedGasAnalysisError(AnalysisError):
    """Exception raised during burned gas analysis."""

    def __init__(self, analysis_type: str, reason: str):
        super().__init__(
            f"Burned gas {analysis_type} analysis failed: {reason}",
            {"analysis_type": analysis_type, "reason": reason}
        )
        self.analysis_type = analysis_type


class ThermodynamicError(AnalysisError):
    """Exception raised during thermodynamic calculations."""

    def __init__(self, calculation_type: str, state_info: Optional[Dict] = None):
        details = {"calculation_type": calculation_type}
        if state_info:
            details.update(state_info)

        super().__init__(
            f"Thermodynamic calculation failed: {calculation_type}",
            details
        )
        self.calculation_type = calculation_type


class ConvergenceError(AnalysisError):
    """Exception raised when iterative analysis fails to converge."""

    def __init__(self, algorithm: str, max_iterations: int, tolerance: float):
        super().__init__(
            f"Algorithm '{algorithm}' failed to converge within {max_iterations} iterations (tolerance: {tolerance})",
            {"algorithm": algorithm, "max_iterations": max_iterations, "tolerance": tolerance}
        )
        self.algorithm = algorithm
        self.max_iterations = max_iterations
        self.tolerance = tolerance


# =============================================================================
# Visualization Layer Exceptions
# =============================================================================

class VisualizationError(PeleProcessingError):
    """Base exception for visualization-related errors."""
    pass


class PlotGenerationError(VisualizationError):
    """Exception raised when plot generation fails."""

    def __init__(self, plot_type: str, reason: str):
        super().__init__(
            f"Failed to generate {plot_type} plot: {reason}",
            {"plot_type": plot_type, "reason": reason}
        )
        self.plot_type = plot_type


class AnimationError(VisualizationError):
    """Exception raised during animation generation."""

    def __init__(self, stage: str, reason: str):
        super().__init__(
            f"Animation generation failed at {stage}: {reason}",
            {"stage": stage, "reason": reason}
        )
        self.stage = stage


class OutputFormatError(VisualizationError):
    """Exception raised when output formatting fails."""

    def __init__(self, format_type: str, reason: str):
        super().__init__(
            f"Output formatting failed for {format_type}: {reason}",
            {"format_type": format_type, "reason": reason}
        )
        self.format_type = format_type


# =============================================================================
# Parallel Processing Exceptions
# =============================================================================

class ParallelProcessingError(PeleProcessingError):
    """Base exception for parallel processing errors."""
    pass


class MPIError(ParallelProcessingError):
    """Exception raised during MPI operations."""

    def __init__(self, operation: str, rank: int, reason: str):
        super().__init__(
            f"MPI {operation} failed on rank {rank}: {reason}",
            {"operation": operation, "rank": rank, "reason": reason}
        )
        self.operation = operation
        self.rank = rank


class WorkDistributionError(ParallelProcessingError):
    """Exception raised during work distribution."""

    def __init__(self, reason: str, work_count: int, worker_count: int):
        super().__init__(
            f"Work distribution failed: {reason}",
            {"reason": reason, "work_count": work_count, "worker_count": worker_count}
        )


class ProcessSynchronizationError(ParallelProcessingError):
    """Exception raised during process synchronization."""

    def __init__(self, operation: str, timeout: Optional[float] = None):
        details = {"operation": operation}
        if timeout:
            details["timeout"] = timeout

        message = f"Process synchronization failed during {operation}"
        if timeout:
            message += f" (timeout: {timeout}s)"

        super().__init__(message, details)


# =============================================================================
# Configuration and System Exceptions
# =============================================================================

class ConfigurationError(PeleProcessingError):
    """Exception raised for configuration-related errors."""

    def __init__(self, config_type: str, reason: str,
                 invalid_values: Optional[Dict[str, Any]] = None):
        details = {"config_type": config_type, "reason": reason}
        if invalid_values:
            details["invalid_values"] = invalid_values

        super().__init__(
            f"Configuration error in {config_type}: {reason}",
            details
        )
        self.config_type = config_type


class DependencyError(PeleProcessingError):
    """Exception raised when required dependencies are missing or invalid."""

    def __init__(self, dependency_name: str, required_version: Optional[str] = None,
                 found_version: Optional[str] = None):
        details = {"dependency": dependency_name}
        if required_version:
            details["required_version"] = required_version
        if found_version:
            details["found_version"] = found_version

        message = f"Dependency '{dependency_name}' is not available"
        if required_version and found_version:
            message += f" (required: {required_version}, found: {found_version})"
        elif required_version:
            message += f" (required: {required_version})"

        super().__init__(message, details)
        self.dependency_name = dependency_name


class ResourceError(PeleProcessingError):
    """Exception raised when system resources are insufficient."""

    def __init__(self, resource_type: str, required: str, available: str):
        super().__init__(
            f"Insufficient {resource_type}: required {required}, available {available}",
            {"resource_type": resource_type, "required": required, "available": available}
        )
        self.resource_type = resource_type


class FileSystemError(PeleProcessingError):
    """Exception raised for file system operations."""

    def __init__(self, operation: str, path: str, reason: str):
        super().__init__(
            f"File system operation '{operation}' failed for '{path}': {reason}",
            {"operation": operation, "path": path, "reason": reason}
        )
        self.operation = operation
        self.path = path


# =============================================================================
# Utility Functions
# =============================================================================

def format_exception_chain(exception: Exception) -> str:
    """Format exception chain for detailed error reporting.

    Args:
        exception: Exception to format

    Returns:
        Formatted string showing exception chain
    """
    lines = []
    current = exception
    level = 0

    while current is not None:
        indent = "  " * level
        if isinstance(current, PeleProcessingError) and current.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in current.details.items())
            lines.append(f"{indent}{type(current).__name__}: {current.message}")
            lines.append(f"{indent}  Details: {detail_str}")
        else:
            lines.append(f"{indent}{type(current).__name__}: {current}")

        current = current.__cause__
        level += 1

    return "\n".join(lines)


def create_error_context(operation: str, **kwargs) -> Dict[str, Any]:
    """Create standardized error context dictionary.

    Args:
        operation: Name of operation being performed
        **kwargs: Additional context information

    Returns:
        Dictionary with error context
    """
    import time
    import os

    context = {
        "operation": operation,
        "timestamp": time.time(),
        "process_id": os.getpid(),
    }
    context.update(kwargs)
    return context