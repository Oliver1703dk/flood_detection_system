"""Factory for creating LLM classifiers (local or API-based)."""

from typing import Optional
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def create_llm_classifier(
    model: Optional[str] = None,
    use_local: Optional[bool] = None,
    **kwargs
):
    """Create an LLM classifier based on configuration.
    
    Parameters
    ----------
    model:
        Model name/identifier. If None, uses config defaults.
    use_local:
        If True, use local VLM. If False, use OpenAI API.
        If None, uses config.USE_LOCAL_LLM.
    **kwargs:
        Additional arguments passed to the classifier constructor.
        
    Returns
    -------
    LLMImageClassifier or LocalVLMClassifier
        The appropriate classifier instance.
    """
    
    # Determine whether to use local or API-based model
    if use_local is None:
        use_local = getattr(config, "USE_LOCAL_LLM", False)
    
    if use_local:
        print("Creating local VLM classifier...")
        # FIXED: Import from correct location
        from jetson_worker.llm.local_llm_classifier import LocalVLMClassifier
        
        model_name = model or getattr(config, "LOCAL_LLM_MODEL", "Efficient-Large-Model/VILA1.5-3b")
        device = kwargs.pop("device", getattr(config, "LOCAL_LLM_DEVICE", None))
        use_4bit = kwargs.pop("use_4bit", getattr(config, "LOCAL_LLM_USE_4BIT", True))
        
        return LocalVLMClassifier(
            model_name=model_name,
            device=device,
            use_4bit=use_4bit,
            **kwargs
        )
    else:
        print("Creating OpenAI API classifier...")
        from flood_classifier.inference.llm_image_classifier import LLMImageClassifier
        
        model_name = model or getattr(config, "FSM_DEFAULTS", {}).get("llm_model", "gpt-4.1-mini")
        
        return LLMImageClassifier(
            model=model_name,
            **kwargs
        )