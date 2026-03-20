"""
AHI (Apnea-Hypopnea Index) computation utilities.

AHI = (Number of apneas + Number of hypopneas) / Hours of sleep
"""

from typing import List, Tuple


def compute_ahi(
    apnea_events: int,
    hypopnea_events: int,
    sleep_duration_hours: float,
) -> float:
    """
    Compute Apnea-Hypopnea Index.
    
    Args:
        apnea_events: Number of apnea events
        hypopnea_events: Number of hypopnea events
        sleep_duration_hours: Total sleep duration in hours
    
    Returns:
        AHI value (events per hour)
    """
    if sleep_duration_hours <= 0:
        return 0.0
    
    return (apnea_events + hypopnea_events) / sleep_duration_hours


def ahi_to_severity(ahi: float) -> int:
    """
    Convert AHI to severity class.
    
    Args:
        ahi: AHI value
    
    Returns:
        Severity class (0=Healthy, 1=Mild, 2=Moderate, 3=Severe)
    """
    if ahi < 5:
        return 0  # Healthy
    elif ahi < 15:
        return 1  # Mild
    elif ahi < 30:
        return 2  # Moderate
    else:
        return 3  # Severe


def ahi_to_severity_label(ahi: float) -> str:
    """
    Convert AHI to severity label string.
    
    Args:
        ahi: AHI value
    
    Returns:
        Severity label ('Healthy', 'Mild', 'Moderate', 'Severe')
    """
    if ahi < 5:
        return 'Healthy'
    elif ahi < 15:
        return 'Mild'
    elif ahi < 30:
        return 'Moderate'
    else:
        return 'Severe'


def severity_to_ahi_range(severity: int) -> Tuple[float, float]:
    """
    Get AHI range for severity class.
    
    Args:
        severity: Severity class (0-3)
    
    Returns:
        (min_ahi, max_ahi) tuple
    """
    ranges = {
        0: (0.0, 5.0),      # Healthy
        1: (5.0, 15.0),     # Mild
        2: (15.0, 30.0),    # Moderate
        3: (30.0, float('inf')),  # Severe
    }
    return ranges.get(severity, (0.0, float('inf')))
