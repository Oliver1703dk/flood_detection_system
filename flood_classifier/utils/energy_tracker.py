"""Energy tracking using codecarbon for CPU-based energy estimation on Raspberry Pi."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Any, Iterator

try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    EmissionsTracker = None
    OfflineEmissionsTracker = None  # type: ignore

import psutil


class _MeasurementContext:
    """Context manager for measuring energy."""
    
    def __init__(self, tracker: EnergyTracker, operation_name: str = "operation") -> None:
        self.tracker = tracker
        self.operation_name = operation_name
        self.metrics: Dict[str, Optional[float]] = {}
        
    def __enter__(self) -> Dict[str, Optional[float]]:
        self.tracker.start_measurement(self.operation_name)
        return self.metrics
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.metrics.update(self.tracker.stop_measurement())


class EnergyTracker:
    """Wrapper around codecarbon's EmissionsTracker for CPU energy estimation."""

    def __init__(self, offline: bool = True) -> None:
        """Initialize the energy tracker.

        Args:
            offline: If True, disable cloud emissions API calls and run offline.
        """
        self.offline = offline
        self._tracker: Optional[Any] = None
        self._lock = threading.Lock()
        self._active = False
        
    def start_measurement(self, operation_name: str = "operation") -> None:
        """Start energy tracking for an operation.
        
        Args:
            operation_name: Label for this measurement interval.
        """
        if not CODECARBON_AVAILABLE:
            return
            
        with self._lock:
            if self._active:
                # Already tracking; stop previous one first
                self._stop_tracker()
            
            try:
                # Configure codecarbon for Raspberry Pi
                kwargs = {
                    "save_to_file": False,  # Don't save CSV files
                    "log_level": "error",   # Suppress verbose logging
                    "measure_power_secs": 0.1,  # Sample every 100ms
                    "tracking_mode": "process",  # Per-process for Pi accuracy
                }
                
                if self.offline:
                    # Use OfflineEmissionsTracker for no-internet carbon intensity
                    kwargs["country_iso_code"] = os.getenv("CODECARBON_COUNTRY_ISO", "USA")  # Adjust default
                    self._tracker = OfflineEmissionsTracker(**kwargs)
                else:
                    # Online mode (no upload)
                    kwargs["save_to_api"] = False
                    self._tracker = EmissionsTracker(**kwargs)
                
                self._tracker.start()
                self._active = True
            except Exception as e:
                # If codecarbon fails to initialize, continue without tracking
                print(f"Warning: Energy tracking failed to start: {e}")
                self._tracker = None
                self._active = False
    
    def stop_measurement(self) -> Dict[str, Optional[float]]:
        """Stop energy tracking and return metrics.
        
        Returns:
            Dict with energy metrics:
                - energy_j: Energy consumption in Joules
                - power_w: Average power consumption in Watts
                - cpu_util_%: CPU utilization percentage
        """
        with self._lock:
            if not self._active or self._tracker is None:
                return {
                    "energy_j": None,
                    "power_w": None,
                    "cpu_util_%": None,
                }
            
            try:
                # Stop the tracker and get emissions data
                emissions_data = self._stop_tracker()
                
                if emissions_data is None:
                    return {
                        "energy_j": None,
                        "power_w": None,
                        "cpu_util_%": None,
                    }
                
                # Extract energy in Joules (emissions_data gives energy in kWh)
                energy_kwh = emissions_data.energy_consumed or 0.0
                energy_j = energy_kwh * 3.6e6 if energy_kwh else None  # kWh to Joules
                
                # Get CPU utilization % (snapshot at end; approx average)
                cpu_util = self.get_cpu_utilization()
                
                # Calculate average power if we have duration
                duration_s = emissions_data.duration or 0.0
                power_w = (energy_j / duration_s) if (energy_j and duration_s > 0) else None
                
                return {
                    "energy_j": energy_j,
                    "power_w": power_w,
                    "cpu_util_%": cpu_util,
                }
            except Exception as e:
                print(f"Warning: Energy tracking failed to stop: {e}")
                return {
                    "energy_j": None,
                    "power_w": None,
                    "cpu_util_%": None,
                }
    
    def _stop_tracker(self) -> Optional[Any]:
        """Stop the tracker and return emissions data.
        
        Returns:
            EmissionsData object or None if tracking failed.
        """
        if not self._tracker:
            return None
            
        try:
            self._tracker.stop()
            emissions_data = self._tracker.final_emissions_data
            self._tracker = None
            self._active = False
            return emissions_data
        except Exception as e:
            print(f"Warning: Failed to stop tracker cleanly: {e}")
            self._tracker = None
            self._active = False
            return None
    
    def get_cpu_utilization(self) -> float:
        """Get current CPU utilization percentage.
        
        Returns:
            CPU utilization as a percentage (0-100).
        """
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def measure(self, operation_name: str = "operation") -> _MeasurementContext:
        """Create a context manager for measuring energy.
        
        Args:
            operation_name: Label for this measurement interval.
            
        Returns:
            Context manager that returns energy metrics dict.
            
        Example:
            >>> tracker = EnergyTracker()
            >>> with tracker.measure("my_operation") as metrics:
            ...     # do work here
            ...     pass
            >>> print(metrics["energy_j"])
        """
        return _MeasurementContext(self, operation_name)


# Create a default singleton instance for convenience
_default_tracker: Optional[EnergyTracker] = None


def get_default_energy_tracker() -> EnergyTracker:
    """Get or create the default energy tracker singleton.
    
    Returns:
        Shared EnergyTracker instance.
    """
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = EnergyTracker(offline=True)
        # Set CODECARBON_COUNTRY_ISO env var for your location (e.g., "GBR" for UK)
    return _default_tracker


# Alias for backwards compatibility
get_energy_tracker = get_default_energy_tracker