<!-- 666f87e1-fe6b-4eec-9a0e-8ad421a6c4aa 1fe95989-7698-4436-b728-e1e398442c77 -->
# Remove IMPORTANCE Dependency

## Overview

Remove the IMPORTANCE configuration system since FSM tier selection already determines model usage:

- S0 always uses NANO locally (determined by FSM)
- S1/S2/S3 use FSM-selected tiers remotely (determined by FSM)
- Local inference only needs models specified by model_size and model_number

## Changes Required

### 1. Simplify MultiModelInference model selection

**File:** `yolov8_processor/inference/multi_model_inference.py`

- **Simplify `_determine_model_params()` method (lines 39-74):**
- Remove all IMPORTANCE-based logic (lines 41-44, 53-72)
- Return `model_size` and `model_number` directly from config
- Update docstring to remove IMPORTANCE reference

- **Update `reload_from_config()` docstring (line 77):**
- Change from "according to IMPORTANCE settings" to "according to model_size and model_number settings"

- **Remove or deprecate `update_importance()` method (lines 85-91):**
- Option: Remove entirely (not used anywhere)
- Option: Keep but mark as deprecated/no-op

### 2. Update FSM weight defaults

**File:** `flood_classifier/fsm/flood_fsm.py`

- **Update FSMParams weight fields (lines 140-147):**
- Change `accuracy_weight`, `timeliness_weight`, `energy_weight` default_factory
- Remove `getattr(config, "IMPORTANCE", {})` dependency
- Use hardcoded defaults: `lambda: 0.34`, `lambda: 0.33`, `lambda: 0.33`

### 3. Remove IMPORTANCE from all config files

Remove the IMPORTANCE dictionary from:

- `config.py` (lines 34-38)
- `config_ablation1.py`
- `config_ablation1b.py`
- `config_ablation2.py`
- `config_ablation2b.py`
- `config_ablation3.py`
- `config_ablation3b.py`
- `config_ablation4.py`
- `config_ablation4b.py`

### 4. Update documentation

**File:** `system_overview.md`

- **Line 146:** Remove or update the IMPORTANCE reference to explain that model selection is now handled by FSM tier selection and model_size/model_number config

## Implementation Notes

- The FSM already selects NANO for S0 (local) and appropriate tiers for other states (remote)
- Local inference only needs the models specified by `model_size` and `model_number` (typically nano, 3 models)
- FSM weights are used for tier selection logic but don't need to come from IMPORTANCE config
- No runtime behavior changes - this is a simplification/cleanup

### To-dos

- [ ] Simplify _determine_model_params() in multi_model_inference.py to use model_size and model_number directly, removing all IMPORTANCE logic
- [ ] Remove accuracy_weight, timeliness_weight, energy_weight fields from FSMParams in flood_fsm.py
- [ ] Remove accuracy_dominant() function from flood_fsm.py
- [ ] Simplify S1 tier selection to always use SMALL (remove accuracy_dominant check)
- [ ] Simplify S2 tier selection to always stay at MEDIUM (remove LARGE upgrade logic)
- [ ] Remove optional LLM confirmation logic that depends on accuracy_weight comparison
- [ ] Remove IMPORTANCE dictionary from all 9 config files (config.py and all config_ablation*.py files)
- [ ] Update system_overview.md to remove/update IMPORTANCE reference
- [ ] Update docstrings and comments in multi_model_inference.py to remove IMPORTANCE references