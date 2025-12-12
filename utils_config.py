"""
Configuration utilities for loading and validating JSON config.
"""
import json
import os
from typing import Dict, Any


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that required configuration keys exist.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_keys = [
        'cameras', 'detection', 'scoring', 'alerts', 
        'road_detection', 'logging', 'display'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    return True


def get_config_value(config: Dict[str, Any], *keys) -> Any:
    """
    Safely get nested config value.
    
    Args:
        config: Configuration dictionary
        *keys: Nested keys to traverse
        
    Returns:
        Config value or None if not found
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value

