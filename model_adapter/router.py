import os
import time
import random
import requests
from typing import Dict, Any, List, Optional
from .base import get_adapter

def infer(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Provider-agnostic router function implementing the infer interface:
    infer(messages, tools, config) -> {text, tool_calls, usage, stop_reason}
    Includes retry logic with jittered exponential backoff (up to 3 attempts) for transport-level failures.
    Supports model configuration dictionaries (e.g., config={"model": {"planner": "...", "executor": "..."}}).
    """
    config = dict(config or {})
    
    # Check if config has a nested 'model' dictionary containing tiers (e.g. planner, executor)
    model_cfg = config.get("model")
    tier_model_id = None
    if isinstance(model_cfg, dict):
        tier = config.get("tier") or config.get("role") or "planner"
        tier_model_id = model_cfg.get(tier) or model_cfg.get("default")
    
    model_id = tier_model_id or config.get("model_id") or os.getenv("MODEL_ID", "openrouter/free")
    
    # Determine provider and clean model_id if prefixed
    provider = config.get("provider")
    if not provider:
        if "/" in model_id:
            parts = model_id.split("/", 1)
            provider = parts[0]
            model_id = parts[1]
        else:
            provider = "openrouter"
    elif isinstance(model_id, str):
        if "/" in model_id:
            parts = model_id.split("/", 1)
            provider = parts[0]
            model_id = parts[1]

    api_key = config.get("api_key")
    
    # Pop provider and model_id from config copy if passed to avoid duplicate keyword argument error
    config_clean = dict(config)
    config_clean.pop("provider", None)
    config_clean.pop("model_id", None)
    config_clean.pop("api_key", None)
    config_clean.pop("model", None)

    adapter = get_adapter(provider=provider, model_id=model_id, api_key=api_key, **config_clean)

    max_attempts = 3
    base_delay = 1.0
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return adapter.infer(messages, tools=tools, config=config)
        except (requests.exceptions.RequestException, TimeoutError, RuntimeError) as e:
            err_str = str(e)
            if "error (4" in err_str and not "429" in err_str:
                raise
            
            last_exception = e
            if attempt == max_attempts:
                break
            
            sleep_time = (base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            time.sleep(sleep_time)

    raise RuntimeError(f"Model inference failed after {max_attempts} attempts. Last error: {last_exception}")
