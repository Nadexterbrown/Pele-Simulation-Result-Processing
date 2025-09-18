"""
Pele-specific constants and variable mappings.
Based on the original 2D-Pele-Processing.py utilities.
"""
from typing import Dict, List, Any

# Pele Variable Map - matches the original PELE_VAR_MAP from utilities/pele.py
PELE_VAR_MAP = {
    'X': {'Name': 'x', 'Units': 'cm'},
    'Y': {'Name': 'y', 'Units': 'cm'},
    'Temperature': {'Name': 'Temp', 'Units': 'K'},
    'Pressure': {'Name': 'pressure', 'Units': 'g / cm / s^2'},
    'Density': {'Name': 'density', 'Units': 'g / cm^3'},
    'Viscosity': {'Name': 'viscosity', 'Units': 'g / cm / s'},
    'Conductivity': {'Name': 'conductivity', 'Units': 'g cm^2 / s^3 / cm / K'},
    'Sound speed': {'Name': 'soundspeed', 'Units': 'cm / s'},
    'Mach Number': {'Name': 'MachNumber', 'Units': ''},
    'X Velocity': {'Name': 'x_velocity', 'Units': 'cm / s'},
    'Y Velocity': {'Name': 'y_velocity', 'Units': 'cm / s'},
    'Heat Release Rate': {'Name': 'heatRelease', 'Units': 'g cm^2 / s^3 / cm^3'},
    'Cp': {'Name': 'cp', 'Units': 'g cm^2 / s^2 / g / K'},
    'Cv': {'Name': 'cv', 'Units': 'g cm^2 / s^2 / g / K'},
}


def add_species_vars(species_list: List[str]) -> None:
    """Add species-specific variables to PELE_VAR_MAP."""
    for species in species_list:
        PELE_VAR_MAP[f'Y({species})'] = {'Name': f'Y({species})', 'Units': ''}
        PELE_VAR_MAP[f'D({species})'] = {'Name': f'D({species})', 'Units': ''}
        PELE_VAR_MAP[f'W({species})'] = {'Name': f'rho_omega_{species}', 'Units': 'g / cm^3 / s'}


def get_missing_fields(field_names_in_data: List[str], exclude_fields: List[str] = None) -> Dict[str, str]:
    """
    Identify fields missing from Pele data that need Cantera calculation.
    Based on the missing_str logic from the original code.
    """
    if exclude_fields is None:
        exclude_fields = {"Grid", "X", "Y"}
    
    missing_fields = {}
    for var_key, var_info in PELE_VAR_MAP.items():
        if var_key in exclude_fields:
            continue
            
        pele_field_name = var_info["Name"]
        if pele_field_name not in field_names_in_data:
            missing_fields[var_key] = pele_field_name
            
    return missing_fields


# Common species for H2 mechanisms
COMMON_H2_SPECIES = ['H2', 'O2', 'H2O', 'H', 'O', 'OH', 'HO2', 'H2O2', 'N2']

# Default flame temperature for contour extraction
DEFAULT_FLAME_TEMP = 2500.0  # K