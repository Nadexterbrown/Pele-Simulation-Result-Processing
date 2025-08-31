"""
Filesystem utilities for the Pele processing system.
"""
import os
import re
import shutil
from typing import List, Optional, Union, Pattern
from pathlib import Path

from ..core.exceptions import FileSystemError


def ensure_long_path_prefix(path: Union[str, Path]) -> str:
    """Add Windows long path prefix for paths over 260 characters."""
    path_str = str(path)
    if os.name == 'nt':  # Windows only
        if path_str.startswith(r"\\"):
            return r"\\?\UNC" + path_str[1:]
        return r"\\?\\" + path_str
    return path_str


def safe_mkdir(path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> Path:
    """Safely create directory with error handling."""
    path = Path(path)
    try:
        if os.name == 'nt':
            safe_path = ensure_long_path_prefix(path)
            os.makedirs(safe_path, exist_ok=exist_ok)
        else:
            path.mkdir(parents=parents, exist_ok=exist_ok)
        return path
    except Exception as e:
        raise FileSystemError("mkdir", str(path), str(e))


def safe_copy(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """Safely copy file with error handling."""
    try:
        src_path = ensure_long_path_prefix(src) if os.name == 'nt' else str(src)
        dst_path = ensure_long_path_prefix(dst) if os.name == 'nt' else str(dst)
        shutil.copy2(src_path, dst_path)
    except Exception as e:
        raise FileSystemError("copy", f"{src} -> {dst}", str(e))


def find_files(directory: Union[str, Path],
               pattern: str = "*",
               recursive: bool = False,
               file_only: bool = True) -> List[Path]:
    """Find files matching pattern."""
    directory = Path(directory)

    if not directory.exists():
        raise FileSystemError("find", str(directory), "Directory does not exist")

    try:
        if recursive:
            matches = directory.rglob(pattern)
        else:
            matches = directory.glob(pattern)

        if file_only:
            return [p for p in matches if p.is_file()]
        return list(matches)

    except Exception as e:
        raise FileSystemError("find", str(directory), str(e))


def load_dataset_paths(data_directory: Union[str, Path],
                       pattern: str = "plt*") -> List[Path]:
    """Load Pele dataset paths with numerical sorting."""
    directory = Path(data_directory)

    if not directory.exists():
        raise FileSystemError("read", str(directory), "Directory does not exist")

    # Find all matching directories
    dataset_paths = []
    for item in directory.iterdir():
        if item.is_dir() and item.name.startswith(pattern.replace("*", "")):
            dataset_paths.append(item)

    # Sort numerically by plt number
    return sort_dataset_paths(dataset_paths)


def sort_dataset_paths(paths: List[Path]) -> List[Path]:
    """Sort dataset paths numerically by plt number."""

    def extract_number(path: Path) -> int:
        match = re.search(r'plt(\d+)', path.name)
        return int(match.group(1)) if match else 0

    return sorted(paths, key=extract_number)


def get_file_size(path: Union[str, Path]) -> int:
    """Get file size in bytes."""
    try:
        return Path(path).stat().st_size
    except Exception as e:
        raise FileSystemError("stat", str(path), str(e))


def disk_usage(path: Union[str, Path]) -> tuple[int, int, int]:
    """Get disk usage statistics (total, used, free) in bytes."""
    try:
        if hasattr(shutil, 'disk_usage'):
            return shutil.disk_usage(path)
        else:
            # Fallback for older Python versions
            statvfs = os.statvfs(path)
            free = statvfs.f_bavail * statvfs.f_frsize
            total = statvfs.f_blocks * statvfs.f_frsize
            used = total - free
            return total, used, free
    except Exception as e:
        raise FileSystemError("disk_usage", str(path), str(e))


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    if path.exists() and not path.is_dir():
        raise FileSystemError("mkdir", str(path), "Path exists but is not a directory")

    safe_mkdir(path)
    return path


def clean_filename(filename: str, replacement: str = "_") -> str:
    """Clean filename by removing invalid characters."""
    # Remove or replace characters that are invalid in filenames
    invalid_chars = r'[<>:"/\\|?*]'
    cleaned = re.sub(invalid_chars, replacement, filename)

    # Remove trailing dots and spaces (Windows issue)
    cleaned = cleaned.rstrip('. ')

    return cleaned


class DirectoryManager:
    """Context manager for temporary directory operations."""

    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.created_dirs = []

    def __enter__(self) -> 'DirectoryManager':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup created directories if needed
        pass

    def create_subdir(self, name: str) -> Path:
        """Create subdirectory and track it."""
        subdir = self.base_path / clean_filename(name)
        safe_mkdir(subdir)
        self.created_dirs.append(subdir)
        return subdir

    def get_path(self, *parts: str) -> Path:
        """Get path relative to base."""
        return self.base_path.joinpath(*parts)


class FileFilter:
    """File filtering utility."""

    def __init__(self):
        self.size_filters = []
        self.name_filters = []
        self.extension_filters = []

    def add_size_filter(self, min_size: Optional[int] = None,
                        max_size: Optional[int] = None) -> 'FileFilter':
        """Add file size filter."""
        self.size_filters.append((min_size, max_size))
        return self

    def add_name_filter(self, pattern: str, regex: bool = False) -> 'FileFilter':
        """Add filename pattern filter."""
        if regex:
            compiled_pattern = re.compile(pattern)
        else:
            # Convert glob pattern to regex
            pattern = pattern.replace('*', '.*').replace('?', '.')
            compiled_pattern = re.compile(f'^{pattern}$')

        self.name_filters.append(compiled_pattern)
        return self

    def add_extension_filter(self, *extensions: str) -> 'FileFilter':
        """Add file extension filter."""
        # Normalize extensions (ensure they start with .)
        normalized = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        self.extension_filters.extend(normalized)
        return self

    def filter_files(self, files: List[Path]) -> List[Path]:
        """Apply all filters to file list."""
        filtered = []

        for file_path in files:
            if self._passes_filters(file_path):
                filtered.append(file_path)

        return filtered

    def _passes_filters(self, file_path: Path) -> bool:
        """Check if file passes all filters."""
        # Size filters
        if self.size_filters:
            try:
                file_size = file_path.stat().st_size
                for min_size, max_size in self.size_filters:
                    if min_size is not None and file_size < min_size:
                        return False
                    if max_size is not None and file_size > max_size:
                        return False
            except OSError:
                return False

        # Name filters
        if self.name_filters:
            filename = file_path.name
            if not any(pattern.match(filename) for pattern in self.name_filters):
                return False

        # Extension filters
        if self.extension_filters:
            if file_path.suffix not in self.extension_filters:
                return False

        return True