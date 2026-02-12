"""Configuration loading and validation utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def load_config(yaml_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration from a YAML file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing validated configuration
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ConfigValidationError: If required fields are missing or invalid
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(yaml_path)
    
    # Check if file exists
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    # Load YAML
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML file {yaml_path}: {e}")
    
    # Validate configuration
    _validate_config(config, yaml_path)
    
    return config


def _validate_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Validate that configuration contains all required fields.
    
    Args:
        config: Configuration dictionary to validate
        config_path: Path to config file (for error messages)
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if config is None:
        raise ConfigValidationError(f"Configuration file {config_path} is empty")
    
    # Required top-level sections
    required_sections = ['model', 'data', 'training']
    missing_sections = [section for section in required_sections if section not in config]
    
    if missing_sections:
        raise ConfigValidationError(
            f"Missing required configuration sections in {config_path}: {', '.join(missing_sections)}"
        )
    
    # Validate model section
    _validate_model_config(config['model'], config_path)
    
    # Validate data section
    _validate_data_config(config['data'], config_path)
    
    # Validate training section
    _validate_training_config(config['training'], config_path)


def _validate_model_config(model_config: Dict[str, Any], config_path: str) -> None:
    """Validate model configuration section."""
    required_fields = ['name']
    _check_required_fields(model_config, required_fields, 'model', config_path)
    
    # Validate model name is a known type
    valid_models = ['whisper', 'wav2vec2']
    if model_config['name'] not in valid_models:
        raise ConfigValidationError(
            f"Invalid model name '{model_config['name']}' in {config_path}. "
            f"Must be one of: {', '.join(valid_models)}"
        )
    
    # For Whisper models, variant and pretrained are recommended but not strictly required
    # (they can have defaults)


def _validate_data_config(data_config: Dict[str, Any], config_path: str) -> None:
    """Validate data configuration section."""
    required_fields = ['train_manifest', 'audio_dirs']
    _check_required_fields(data_config, required_fields, 'data', config_path)
    
    # Validate audio_dirs is a list
    if not isinstance(data_config['audio_dirs'], list):
        raise ConfigValidationError(
            f"'data.audio_dirs' must be a list in {config_path}"
        )
    
    if len(data_config['audio_dirs']) == 0:
        raise ConfigValidationError(
            f"'data.audio_dirs' cannot be empty in {config_path}"
        )


def _validate_training_config(training_config: Dict[str, Any], config_path: str) -> None:
    """Validate training configuration section."""
    required_fields = ['output_dir', 'batch_size', 'num_epochs']
    _check_required_fields(training_config, required_fields, 'training', config_path)
    
    # Validate numeric fields are positive
    if training_config['batch_size'] <= 0:
        raise ConfigValidationError(
            f"'training.batch_size' must be positive in {config_path}"
        )
    
    if training_config['num_epochs'] <= 0:
        raise ConfigValidationError(
            f"'training.num_epochs' must be positive in {config_path}"
        )


def _check_required_fields(
    config_section: Dict[str, Any], 
    required_fields: List[str], 
    section_name: str, 
    config_path: str
) -> None:
    """
    Check that all required fields are present in a configuration section.
    
    Args:
        config_section: Configuration section to check
        required_fields: List of required field names
        section_name: Name of the section (for error messages)
        config_path: Path to config file (for error messages)
        
    Raises:
        ConfigValidationError: If any required fields are missing
    """
    if config_section is None:
        raise ConfigValidationError(
            f"Configuration section '{section_name}' is empty in {config_path}"
        )
    
    missing_fields = [field for field in required_fields if field not in config_section]
    
    if missing_fields:
        raise ConfigValidationError(
            f"Missing required fields in '{section_name}' section of {config_path}: "
            f"{', '.join(missing_fields)}"
        )


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from nested configuration using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., 'model.variant')
        default: Default value if key path doesn't exist
        
    Returns:
        Value at the specified path, or default if not found
        
    Example:
        >>> config = {'model': {'name': 'whisper', 'variant': 'medium'}}
        >>> get_nested_value(config, 'model.variant')
        'medium'
        >>> get_nested_value(config, 'model.unknown', 'default')
        'default'
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value
