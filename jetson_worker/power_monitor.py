"""Power monitoring for Jetson using jetson-stats (jtop)."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

try:
    import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False
    jtop = None  # type: ignore


class JetsonPowerMonitor:
    """Monitor power consumption using jetson-stats jtop interface.
    
    Samples power at 100ms intervals and provides per-job energy metrics.
    """

    def __init__(self, sample_interval: float = 0.1):
        """Initialize the power monitor.
        
        Args:
            sample_interval: Sampling interval in seconds (default 0.1 = 100ms)
        """
        self.sample_interval = sample_interval
        self.jtop_instance: Optional[Any] = None
        self._sampling_thread: Optional[threading.Thread] = None
        self._stop_sampling = threading.Event()
        self._lock = threading.Lock()
        self._active_jobs: Dict[str, deque] = defaultdict(lambda: deque())
        self._power_samples: deque = deque()

    def start(self) -> bool:
        """Start the monitoring system.
        
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if not JTOP_AVAILABLE:
            return False
        
        with self._lock:
            if self.jtop_instance is None:
                try:
                    self.jtop_instance = jtop()
                    self.jtop_instance.start()
                except Exception as e:
                    print(f"Warning: Failed to initialize jtop: {e}")
                    self.jtop_instance = None
                    return False
            
            if self._sampling_thread is None or not self._sampling_thread.is_alive():
                self._stop_sampling.clear()
                self._sampling_thread = threading.Thread(
                    target=self._sampling_loop,
                    daemon=True,
                    name="jetson-power-monitor"
                )
                self._sampling_thread.start()
        
        return True

    def stop(self) -> None:
        """Stop the monitoring system."""
        self._stop_sampling.set()
        if self._sampling_thread and self._sampling_thread.is_alive():
            self._sampling_thread.join(timeout=1.0)
        
        with self._lock:
            if self.jtop_instance is not None:
                try:
                    self.jtop_instance.close()
                except Exception:
                    pass
                self.jtop_instance = None

    def start_sampling(self, job_id: str) -> None:
        """Start sampling power for a specific job.
        
        Args:
            job_id: Unique identifier for the job
        """
        with self._lock:
            if not self._active_jobs[job_id]:
                self._active_jobs[job_id] = deque()
            # Add a marker to indicate job start
            self._active_jobs[job_id].append({"timestamp": time.perf_counter(), "start": True})

    def stop_sampling(self, job_id: str) -> Dict[str, Optional[float]]:
        """Stop sampling and calculate energy metrics for a job.
        
        Args:
            job_id: Unique identifier for the job
            
        Returns:
            Dict with energy metrics: energy_j, avg_power_w, peak_power_w, cpu_util_%, gpu_util_%
        """
        with self._lock:
            if job_id not in self._active_jobs:
                return {"energy_j": None, "avg_power_w": None, "peak_power_w": None, 
                       "cpu_util_%": None, "gpu_util_%": None}
            
            samples = list(self._active_jobs[job_id])
            
            # Remove the start marker
            if samples and samples[0].get("start"):
                samples.pop(0)
            
            if not samples or self.jtop_instance is None:
                del self._active_jobs[job_id]
                return {"energy_j": None, "avg_power_w": None, "peak_power_w": None,
                       "cpu_util_%": None, "gpu_util_%": None}
            
            # Calculate metrics from samples
            powers = [s.get("power_w", 0.0) for s in samples if "power_w" in s]
            cpu_utils = [s.get("cpu_util_%", 0.0) for s in samples if "cpu_util_%" in s]
            gpu_utils = [s.get("gpu_util_%", 0.0) for s in samples if "gpu_util_%" in s]
            
            if not powers:
                del self._active_jobs[job_id]
                return {"energy_j": None, "avg_power_w": None, "peak_power_w": None,
                       "cpu_util_%": None, "gpu_util_%": None}
            
            # Calculate energy: integrate power over time
            # Use sample_interval for each sample
            total_energy_j = sum(powers) * self.sample_interval
            avg_power_w = sum(powers) / len(powers) if powers else 0.0
            peak_power_w = max(powers) if powers else 0.0
            
            avg_cpu_util = sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0.0
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
            
            del self._active_jobs[job_id]
            
            return {
                "energy_j": total_energy_j,
                "avg_power_w": avg_power_w,
                "peak_power_w": peak_power_w,
                "cpu_util_%": avg_cpu_util,
                "gpu_util_%": avg_gpu_util
            }

    def _sampling_loop(self) -> None:
        """Background thread that samples power data."""
        while not self._stop_sampling.is_set():
            with self._lock:
                if self.jtop_instance is None:
                    break
                
                try:
                    stats = self.jtop_instance.stats
                    
                    # Extract power and utilization data
                    power_w = stats.get("Power", {}).get("tot", 0.0)
                    cpu_util_pct = stats.get("CPU", {}).get("tot", 0.0)
                    gpu_util_pct = stats.get("GPU", {}).get("val", 0.0)
                    
                    sample = {
                        "timestamp": time.perf_counter(),
                        "power_w": power_w,
                        "cpu_util_%": cpu_util_pct,
                        "gpu_util_%": gpu_util_pct
                    }
                    
                    # Add sample to all active jobs
                    for job_id in list(self._active_jobs.keys()):
                        self._active_jobs[job_id].append(sample.copy())
                        
                except Exception as e:
                    # Continue sampling even if one sample fails
                    print(f"Warning: Failed to read jtop stats: {e}")
            
            # Sleep for sample_interval
            self._stop_sampling.wait(timeout=self.sample_interval)

    def is_available(self) -> bool:
        """Check if power monitoring is available."""
        return JTOP_AVAILABLE


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

