"""Queue depth monitoring utility for tracking processor queue metrics."""
from __future__ import annotations

from collections import deque
from threading import Lock
from time import perf_counter
from typing import Dict, Optional


class QueueMonitor:
    """Thread-safe queue depth monitor for tracking processor queue metrics."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize queue monitor.
        
        Args:
            max_history: Maximum number of queue depth samples to keep in history
        """
        self._lock = Lock()
        self._current_depth = 0
        self._max_depth = 0
        self._total_entries = 0
        self._total_exits = 0
        self._depth_history: deque = deque(maxlen=max_history)
        self._wait_time_history: deque = deque(maxlen=max_history)
    
    def enter(self) -> float:
        """Record a queue entry and return entry timestamp.
        
        Returns:
            Entry timestamp (perf_counter)
        """
        with self._lock:
            entry_ts = perf_counter()
            self._current_depth += 1
            self._total_entries += 1
            if self._current_depth > self._max_depth:
                self._max_depth = self._current_depth
            self._depth_history.append((entry_ts, self._current_depth))
            return entry_ts
    
    def exit(self, entry_ts: Optional[float] = None) -> float:
        """Record a queue exit and return wait time.
        
        Args:
            entry_ts: Entry timestamp (if None, uses current time)
            
        Returns:
            Wait time in seconds
        """
        with self._lock:
            exit_ts = perf_counter()
            if entry_ts is not None:
                wait_time = exit_ts - entry_ts
            else:
                wait_time = 0.0
            
            if self._current_depth > 0:
                self._current_depth -= 1
            self._total_exits += 1
            self._wait_time_history.append(wait_time)
            return wait_time
    
    @property
    def current_depth(self) -> int:
        """Current queue depth."""
        with self._lock:
            return self._current_depth
    
    @property
    def max_depth(self) -> int:
        """Maximum queue depth observed."""
        with self._lock:
            return self._max_depth
    
    def get_stats(self) -> Dict[str, float]:
        """Get queue statistics.
        
        Returns:
            Dictionary with queue statistics
        """
        with self._lock:
            avg_wait = (
                sum(self._wait_time_history) / len(self._wait_time_history)
                if self._wait_time_history else 0.0
            )
            max_wait = max(self._wait_time_history) if self._wait_time_history else 0.0
            
            avg_depth = (
                sum(depth for _, depth in self._depth_history) / len(self._depth_history)
                if self._depth_history else 0.0
            )
            
            return {
                "current_depth": float(self._current_depth),
                "max_depth": float(self._max_depth),
                "avg_depth": avg_depth,
                "total_entries": float(self._total_entries),
                "total_exits": float(self._total_exits),
                "avg_wait_s": avg_wait,
                "max_wait_s": max_wait,
            }
    
    def check_alert_threshold(self, threshold_depth: int = 5, threshold_wait_s: float = 0.5) -> Optional[str]:
        """Check if queue metrics exceed alert thresholds.
        
        Args:
            threshold_depth: Alert if current depth exceeds this
            threshold_wait_s: Alert if average wait time exceeds this
            
        Returns:
            Alert message if threshold exceeded, None otherwise
        """
        stats = self.get_stats()
        
        if stats["current_depth"] >= threshold_depth:
            return (
                f"⚠️ Queue depth alert: {stats['current_depth']:.0f} items "
                f"(threshold: {threshold_depth})"
            )
        
        if stats["avg_wait_s"] >= threshold_wait_s:
            return (
                f"⚠️ Queue wait time alert: {stats['avg_wait_s']:.3f}s average "
                f"(threshold: {threshold_wait_s}s)"
            )
        
        return None

