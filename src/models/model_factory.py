"""
Model factory for creating ASR models from configuration.

This module provides a factory pattern for instantiating ASR models based on
configuration files, enabling flexible model selection without hard-coding
model creation logic throughout the codebase.
"""

import logging
from typing import Any, Dict, Type

from src.models.base_model import BaseASRModel
from src.models.whisper_model import WhisperModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating ASR model instances from configuration.

    This factory pattern allows for easy model selection and instantiation
    based on configuration files. Models are selected via config['model']['name']
    and instantiated with appropriate parameters from the config.

    The factory maintains a registry of available models and supports dynamic
    registration of custom models for extensibility.

    Attributes:
        _MODEL_REGISTRY: Class-level dictionary mapping model names to model classes

    Example:
        >>> from src.utils.config import load_config
        >>> config = load_config('configs/baseline_whisper_small.yaml')
        >>> model = ModelFactory.create_model(config)
        >>> model.load()
        >>> transcription = model.transcribe('audio.flac')
    """

    # Registry mapping model names to model classes
    _MODEL_REGISTRY: Dict[str, Type[BaseASRModel]] = {
        "whisper": WhisperModel,
        # Future models can be added here:
        # 'wav2vec': Wav2VecModel,
        # 'hubert': HubertModel,
    }

    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> BaseASRModel:
        """
        Create and return an ASR model instance based on configuration.

        Reads the model specification from config['model'] and instantiates
        the appropriate model class with configuration parameters. The model
        is NOT loaded at this stage - call model.load() separately.

        Args:
            config: Configuration dictionary containing model specification.
                   Required fields:
                   - config['model']['name']: Model type (e.g., 'whisper')
                   - config['model']['variant']: Model variant (e.g., 'small')
                   Optional fields:
                   - config['model']['device']: Device to use ('cuda', 'cpu', 'mps')

        Returns:
            Instantiated model object implementing BaseASRModel interface.
            The model is initialized but NOT loaded - call model.load() to
            load pretrained weights or checkpoints.

        Raises:
            ValueError: If model name not recognized in registry
            KeyError: If config doesn't contain required model fields
            NotImplementedError: If model type is registered but creation
                               logic not yet implemented

        Example:
            >>> config = {
            ...     'model': {
            ...         'name': 'whisper',
            ...         'variant': 'small',
            ...         'pretrained': 'openai/whisper-small'
            ...     }
            ... }
            >>> model = ModelFactory.create_model(config)
            >>> isinstance(model, WhisperModel)
            True
            >>> model.load(config['model']['pretrained'])
        """
        try:
            model_config = config["model"]
            model_name = model_config["name"].lower()

            logger.info(f"Creating model: {model_name}")

            # Look up model class in registry
            if model_name not in cls._MODEL_REGISTRY:
                available_models = ", ".join(cls._MODEL_REGISTRY.keys())
                raise ValueError(f"Unknown model type: '{model_name}'. " f"Available models: {available_models}")

            cls._MODEL_REGISTRY[model_name]

            # Create model instance with model-specific parameters
            if model_name == "whisper":
                # Safe to cast since we know it's WhisperModel from registry
                from src.models.whisper_model import WhisperModel as WM

                variant = model_config.get("variant", "small")
                device = model_config.get("device", None)
                model = WM(variant=variant, device=device)
                logger.info(f"Created WhisperModel with variant='{variant}'")
            else:
                # For future models, add specific initialization logic here
                raise NotImplementedError(
                    f"Model creation logic for '{model_name}' not yet implemented. "
                    f"The model class is registered but needs initialization code."
                )

            return model

        except KeyError as e:
            logger.error(f"Missing required configuration field: {e}")
            raise KeyError(f"Configuration must contain 'model' section with 'name' field. " f"Missing: {e}") from e
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseASRModel]) -> None:
        """
        Register a new model class in the factory.

        This allows for dynamic registration of custom models without
        modifying the factory code directly. Useful for plugins or
        project-specific model variants.

        Args:
            name: Model name identifier (e.g., 'custom_whisper', 'my_model')
                 Will be converted to lowercase for case-insensitive lookup
            model_class: Model class that implements BaseASRModel interface

        Raises:
            ValueError: If model class doesn't inherit from BaseASRModel

        Example:
            >>> class CustomModel(BaseASRModel):
            ...     def load(self, checkpoint_path=None):
            ...         pass
            ...     def transcribe(self, audio_paths, **kwargs):
            ...         pass
            ...     def save(self, path):
            ...         pass
            ...     def get_model_info(self):
            ...         return {'name': 'custom'}
            >>> ModelFactory.register_model('custom', CustomModel)
            >>> # Now can create via config: {'model': {'name': 'custom', ...}}
        """
        if not issubclass(model_class, BaseASRModel):
            raise ValueError(f"Model class must inherit from BaseASRModel, " f"got {model_class.__name__}")

        name_lower = name.lower()
        cls._MODEL_REGISTRY[name_lower] = model_class
        logger.info(f"Registered model '{name_lower}' -> {model_class.__name__}")

    @classmethod
    def list_available_models(cls) -> list:
        """
        Get list of available model names in the registry.

        Returns:
            List of registered model names (strings)

        Example:
            >>> ModelFactory.list_available_models()
            ['whisper']
            >>> # After registering custom model:
            >>> ModelFactory.register_model('custom', CustomModel)
            >>> ModelFactory.list_available_models()
            ['whisper', 'custom']
        """
        return sorted(list(cls._MODEL_REGISTRY.keys()))
