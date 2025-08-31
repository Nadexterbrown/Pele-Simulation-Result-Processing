"""
Output formatting for the Pele processing system.
"""
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

from ..core.interfaces import OutputFormatter
from ..core.domain import ProcessingBatch, ProcessingResult
from ..core.exceptions import OutputFormatError


class TableFormatter(OutputFormatter):
    """Format results as structured tables."""

    def __init__(self, field_width: int = 20, precision: int = 6):
        self.field_width = field_width
        self.precision = precision

    def format_results_table(self, batch: ProcessingBatch, output_path: Path) -> None:
        """Format processing results as text table."""
        try:
            with open(output_path, 'w') as f:
                self._write_header(f, batch)
                self._write_data_rows(f, batch)
        except Exception as e:
            raise OutputFormatError("table", str(e))

    def format_summary_report(self, batch: ProcessingBatch, output_path: Path) -> None:
        """Generate summary report."""
        try:
            with open(output_path, 'w') as f:
                self._write_summary_header(f, batch)
                self._write_statistics(f, batch)
                self._write_analysis_summary(f, batch)
        except Exception as e:
            raise OutputFormatError("summary", str(e))

    def _write_header(self, f, batch: ProcessingBatch) -> None:
        """Write table header."""
        if not batch.results:
            return

        # Collect all possible headers
        headers = ['Dataset', 'Time']
        sample_result = batch.results[0]

        if sample_result.flame_data:
            headers.extend(['Flame_Pos', 'Flame_Vel', 'Flame_Thick', 'Heat_Release'])
        if sample_result.shock_data:
            headers.extend(['Shock_Pos', 'Shock_Vel'])

        # Write index row
        index_line = "# " + "".join(f"{i + 1:<{self.field_width}}" for i in range(len(headers)))
        f.write(index_line + "\n")

        # Write header row
        header_line = "# " + "".join(f"{h:<{self.field_width}}" for h in headers)
        f.write(header_line + "\n")

    def _write_data_rows(self, f, batch: ProcessingBatch) -> None:
        """Write data rows."""
        for result in batch.results:
            row_data = [
                result.dataset_info.basename,
                f"{result.dataset_info.timestamp:.{self.precision}e}"
            ]

            if result.flame_data:
                row_data.extend([
                    f"{result.flame_data.position:.{self.precision}e}" if result.flame_data.position else "N/A",
                    f"{result.flame_data.velocity:.{self.precision}e}" if result.flame_data.velocity else "N/A",
                    f"{result.flame_data.thickness:.{self.precision}e}" if result.flame_data.thickness else "N/A",
                    f"{result.flame_data.heat_release_rate:.{self.precision}e}" if result.flame_data.heat_release_rate else "N/A",
                ])

            if result.shock_data:
                row_data.extend([
                    f"{result.shock_data.position:.{self.precision}e}" if result.shock_data.position else "N/A",
                    f"{result.shock_data.velocity:.{self.precision}e}" if result.shock_data.velocity else "N/A",
                ])

            # Pad row to match header width
            while len(row_data) < self.field_width:
                row_data.append("N/A")

            row_line = "  " + "".join(f"{str(val):<{self.field_width}}" for val in row_data)
            f.write(row_line + "\n")

    def _write_summary_header(self, f, batch: ProcessingBatch) -> None:
        """Write summary report header."""
        f.write("=" * 80 + "\n")
        f.write("PELE PROCESSING SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total datasets processed: {len(batch.results)}\n")
        f.write(f"Successful: {len(batch.get_successful_results())}\n")
        f.write(f"Failed: {len(batch.results) - len(batch.get_successful_results())}\n\n")

    def _write_statistics(self, f, batch: ProcessingBatch) -> None:
        """Write statistical summary."""
        successful_results = batch.get_successful_results()

        f.write("STATISTICAL SUMMARY\n")
        f.write("-" * 40 + "\n")

        if successful_results:
            # Flame statistics
            flame_positions = [r.flame_data.position for r in successful_results
                               if r.flame_data and r.flame_data.position]
            if flame_positions:
                f.write(f"Flame position range: {np.min(flame_positions):.3e} - {np.max(flame_positions):.3e} m\n")
                f.write(f"Mean flame position: {np.mean(flame_positions):.3e} m\n")

            # Shock statistics
            shock_positions = [r.shock_data.position for r in successful_results
                               if r.shock_data and r.shock_data.position]
            if shock_positions:
                f.write(f"Shock position range: {np.min(shock_positions):.3e} - {np.max(shock_positions):.3e} m\n")
                f.write(f"Mean shock position: {np.mean(shock_positions):.3e} m\n")

        f.write("\n")

    def _write_analysis_summary(self, f, batch: ProcessingBatch) -> None:
        """Write analysis summary."""
        f.write("ANALYSIS SUMMARY\n")
        f.write("-" * 40 + "\n")

        # Calculate wave velocities if possible
        times = batch.get_timestamps()
        flame_positions = batch.get_flame_positions()

        valid_flame = ~np.isnan(flame_positions)
        if np.sum(valid_flame) > 1:
            flame_velocities = np.gradient(flame_positions[valid_flame], times[valid_flame])
            f.write(f"Mean flame velocity: {np.mean(flame_velocities):.3e} m/s\n")

        shock_positions = batch.get_shock_positions()
        valid_shock = ~np.isnan(shock_positions)
        if np.sum(valid_shock) > 1:
            shock_velocities = np.gradient(shock_positions[valid_shock], times[valid_shock])
            f.write(f"Mean shock velocity: {np.mean(shock_velocities):.3e} m/s\n")


class CSVFormatter(OutputFormatter):
    """Format results as CSV."""

    def format_results_table(self, batch: ProcessingBatch, output_path: Path) -> None:
        """Format as CSV."""
        import csv

        if not batch.results:
            return

        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                headers = ['dataset', 'time']
                sample = batch.results[0]
                if sample.flame_data:
                    headers.extend(['flame_position', 'flame_velocity', 'flame_thickness'])
                if sample.shock_data:
                    headers.extend(['shock_position', 'shock_velocity'])

                writer.writerow(headers)

                # Data rows
                for result in batch.results:
                    row = [result.dataset_info.basename, result.dataset_info.timestamp]

                    if result.flame_data:
                        row.extend([
                            result.flame_data.position or '',
                            result.flame_data.velocity or '',
                            result.flame_data.thickness or ''
                        ])

                    if result.shock_data:
                        row.extend([
                            result.shock_data.position or '',
                            result.shock_data.velocity or ''
                        ])

                    writer.writerow(row)

        except Exception as e:
            raise OutputFormatError("csv", str(e))

    def format_summary_report(self, batch: ProcessingBatch, output_path: Path) -> None:
        """CSV summary is same as main table."""
        self.format_results_table(batch, output_path)


class JSONFormatter(OutputFormatter):
    """Format results as JSON."""

    def format_results_table(self, batch: ProcessingBatch, output_path: Path) -> None:
        """Format as JSON."""
        import json

        try:
            data = {
                'metadata': {
                    'total_datasets': len(batch.results),
                    'successful': len(batch.get_successful_results())
                },
                'results': []
            }

            for result in batch.results:
                result_dict = {
                    'dataset': result.dataset_info.basename,
                    'time': result.dataset_info.timestamp,
                    'success': result.success
                }

                if result.flame_data:
                    result_dict['flame'] = {
                        'position': result.flame_data.position,
                        'velocity': result.flame_data.velocity,
                        'thickness': result.flame_data.thickness
                    }

                if result.shock_data:
                    result_dict['shock'] = {
                        'position': result.shock_data.position,
                        'velocity': result.shock_data.velocity
                    }

                data['results'].append(result_dict)

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            raise OutputFormatError("json", str(e))

    def format_summary_report(self, batch: ProcessingBatch, output_path: Path) -> None:
        """JSON summary includes statistics."""
        import json

        try:
            successful = batch.get_successful_results()

            # Calculate statistics
            stats = {}
            if successful:
                flame_pos = [r.flame_data.position for r in successful
                             if r.flame_data and r.flame_data.position]
                if flame_pos:
                    stats['flame'] = {
                        'mean_position': float(np.mean(flame_pos)),
                        'std_position': float(np.std(flame_pos)),
                        'min_position': float(np.min(flame_pos)),
                        'max_position': float(np.max(flame_pos))
                    }

            data = {
                'summary': {
                    'total_datasets': len(batch.results),
                    'successful': len(successful),
                    'success_rate': len(successful) / len(batch.results) if batch.results else 0
                },
                'statistics': stats
            }

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            raise OutputFormatError("json_summary", str(e))


def create_formatter(format_type: str, **kwargs) -> OutputFormatter:
    """Factory for output formatters."""
    formatters = {
        'table': TableFormatter,
        'csv': CSVFormatter,
        'json': JSONFormatter
    }

    if format_type not in formatters:
        raise ValueError(f"Unknown formatter: {format_type}")

    return formatters[format_type](**kwargs)