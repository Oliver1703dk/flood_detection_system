# platform_utils.py
"""
Platform detection utilities for flood detection system.
Automatically detects if running on NVIDIA Jetson or Raspberry Pi.
"""
import os
import platform


def is_jetson():
    """
    Check if running on NVIDIA Jetson.
    
    Returns:
        bool: True if running on Jetson, False otherwise
    """
    hostname = platform.node().lower()
    return (
        "jetson" in hostname or 
        os.path.exists("/etc/nv_tegra_release")
    )


def get_platform_yolo_tier():
    """
    Return appropriate YOLO tier for current platform.
    
    Platform defaults:
        - Jetson: 'small' (more powerful GPU)
        - Pi: 'nano' (lightweight for edge)
    
    Can be overridden with LOCAL_YOLO_TIER environment variable.
    
    Returns:
        str: YOLO tier name ('nano', 'small', 'medium', etc.)
    """
    # Allow manual override via environment variable
    override = os.getenv("LOCAL_YOLO_TIER")
    if override:
        return override
    
    return "small" if is_jetson() else "nano"


# Convenience constants for importing
PLATFORM_YOLO_TIER = get_platform_yolo_tier()
PLATFORM_NAME = "jetson" if is_jetson() else "pi"


if __name__ == "__main__":
    # Simple test when run directly
    print(f"Platform: {PLATFORM_NAME}")
    print(f"Default YOLO Tier: {PLATFORM_YOLO_TIER}")
    print(f"Is Jetson: {is_jetson()}")

