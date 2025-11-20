# Platform Configuration Guide

## Overview

The flood detection system now automatically detects whether it's running on a Raspberry Pi or NVIDIA Jetson and configures the appropriate YOLO model tier.

## Implementation

### Platform Detection
- **File**: `platform_utils.py`
- **Detection Method**: Checks for Jetson-specific files (`/etc/nv_tegra_release`) and hostname
- **Default Tiers**:
  - **Raspberry Pi**: `nano` (lightweight for edge)
  - **Jetson**: `small` (more powerful GPU)

### Updated Files

All configuration files now import platform detection:
- `config.py` - Main configuration
- `config_ablation2.py`, `config_ablation2b.py` - Vision-only FSM configs
- `config_ablation3.py`, `config_ablation3b.py` - Full system local configs
- `config_ablation4.py`, `config_ablation4b.py` - Production configs with remote offload

**Note**: `config_ablation1.py` and `config_ablation1b.py` explicitly use `"medium"` tier as part of the baseline ablation design (not platform-dependent).

## Usage

### Automatic Detection (Recommended)

Simply run your scripts as normal. The system will automatically detect the platform:

```bash
# On Raspberry Pi - automatically uses nano tier
python main_ablation4b.py

# On Jetson - automatically uses small tier
python jetson_worker/worker.py
```

### Manual Override

You can override the automatic detection using an environment variable:

```bash
# Force a specific tier regardless of platform
LOCAL_YOLO_TIER=medium python main_ablation4b.py

# Force large tier
LOCAL_YOLO_TIER=large python main_ablation4b.py
```

### Testing Platform Detection

Check what the system detects on your current hardware:

```bash
python platform_utils.py
```

Output example:
```
Platform: pi
Default YOLO Tier: nano
Is Jetson: False
```

## Configuration Matrix

| Config | Model Tier | Platform-Specific? | Notes |
|--------|------------|-------------------|-------|
| ablation1, ablation1b | `medium` | ❌ No | Baseline - always medium |
| ablation2, ablation2b | Platform | ✅ Yes | nano (Pi) / small (Jetson) |
| ablation3, ablation3b | Platform | ✅ Yes | nano (Pi) / small (Jetson) |
| ablation4, ablation4b | Platform | ✅ Yes | nano (Pi) / small (Jetson) |

## Technical Details

### How It Works

1. **Import**: Config files import `PLATFORM_YOLO_TIER` from `platform_utils.py`
2. **Detection**: On import, `platform_utils.py` checks system characteristics
3. **Assignment**: `LOCAL_YOLO_TIER = PLATFORM_YOLO_TIER` sets the appropriate tier
4. **Override**: Environment variable `LOCAL_YOLO_TIER` can override if set

### Detection Logic

```python
def is_jetson():
    hostname = platform.node().lower()
    return (
        "jetson" in hostname or 
        os.path.exists("/etc/nv_tegra_release")
    )
```

### Benefits

✅ **Automatic**: No manual config changes when switching platforms  
✅ **Simple**: One tiny utility file handles all detection  
✅ **Flexible**: Can override with environment variables  
✅ **Consistent**: All configs use the same detection logic  
✅ **Minimal Changes**: Only 2 lines changed per config file  

## Deployment

### On Raspberry Pi

No special configuration needed. The system automatically detects it's a Pi and uses `nano` tier:

```bash
python main_ablation4b.py
```

### On Jetson

No special configuration needed. The system automatically detects it's a Jetson and uses `small` tier:

```bash
# Start the worker
python jetson_worker/worker.py

# Or run ablations directly on Jetson for testing
python main_ablation4b.py
```

### Cross-Platform Testing

When testing on a development machine (Mac/Linux), the system defaults to Pi behavior (`nano` tier). To simulate Jetson:

```bash
LOCAL_YOLO_TIER=small python main_ablation4b.py
```

## Troubleshooting

### Check Current Configuration

```bash
python -c "import config; print(f'Tier: {config.LOCAL_YOLO_TIER}')"
```

### Force Specific Tier

If automatic detection isn't working correctly:

```bash
# Temporary override
LOCAL_YOLO_TIER=nano python main_ablation4b.py

# Or set in your shell environment
export LOCAL_YOLO_TIER=small
python main_ablation4b.py
```

### Verify Platform Detection

```bash
python platform_utils.py
```

## Summary

The simplest possible solution with minimal code changes:
- ✅ 1 new file: `platform_utils.py` (50 lines)
- ✅ 2 lines changed per config file (1 import + 1 assignment)
- ✅ Automatic platform detection
- ✅ Environment variable override support
- ✅ No code duplication

