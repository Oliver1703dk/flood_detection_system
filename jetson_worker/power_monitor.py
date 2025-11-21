"""Power monitoring stub for Jetson (jtop removed)."""
from __future__ import annotations

from typing import Dict, Optional


class JetsonPowerMonitor:
    """Stub power monitor that returns None for all metrics.
    
    This is a placeholder to keep the API consistent without requiring jtop.
    Use external HMC power analyzers for accurate power measurements.
    """

    def __init__(self, sample_interval: float = 0.1):
        """Initialize the power monitor stub.
        
        Args:
            sample_interval: Unused, kept for API compatibility
        """
        pass

    def start(self) -> bool:
        """Start the monitoring system (no-op).
        
        Returns:
            False (monitoring disabled)
        """
        return False

    def stop(self) -> None:
        """Stop the monitoring system (no-op)."""
        pass

    def start_sampling(self, job_id: str) -> None:
        """Start sampling power for a specific job (no-op).
        
        Args:
            job_id: Unique identifier for the job
        """
        pass

    def stop_sampling(self, job_id: str) -> Dict[str, Optional[float]]:
        """Stop sampling and return None for all metrics.
        
        Args:
            job_id: Unique identifier for the job
            
        Returns:
            Dict with all metrics set to None
        """
        return {
            "energy_j": None,
            "avg_power_w": None,
            "peak_power_w": None,
            "cpu_util_%": None,
            "gpu_util_%": None
        }

    def is_available(self) -> bool:
        """Check if power monitoring is available.
        
        Returns:
            False (always disabled)
        """
        return False


# Global singleton instance
_global_monitor: Optional[JetsonPowerMonitor] = None


def get_power_monitor() -> JetsonPowerMonitor:
    """Get or create the global power monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = JetsonPowerMonitor()
        # Auto-start the monitor
        _global_monitor.start()
    return _global_monitor

