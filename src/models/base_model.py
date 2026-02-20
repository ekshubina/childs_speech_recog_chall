"""
Base model interface for ASR models.

This module defines the abstract base class that all ASR model implementations
must follow, ensuring a consistent interface across different model architectures.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BaseASRModel(ABC):
    """
    Abstract base class defining the interface for ASR models.

    This interface ensures that all ASR model implementations (Whisper, Wav2Vec, etc.)
    provide consistent methods for loading, inference, saving, and metadata access.
    Subclasses must implement all abstract methods.

    Example:
        >>> class MyASRModel(BaseASRModel):
        ...     def load(self, checkpoint_path):
        ...         # Implementation
        ...         pass
        ...
        ...     def transcribe(self, audio_paths):
        ...         # Implementation
        ...         pass
        ...
        ...     def save(self, path):
        ...         # Implementation
        ...         pass
        ...
        ...     def get_model_info(self):
        ...         # Implementation
        ...         return {"name": "my_model"}
    """

    @abstractmethod
    def load(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load model from checkpoint or pretrained weights.

        This method initializes the model either from a pretrained checkpoint
        (e.g., "openai/whisper-medium") or from a saved fine-tuned checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file or pretrained model identifier.
                           If None, should load default pretrained model.

        Raises:
            FileNotFoundError: If checkpoint_path is specified but doesn't exist
            RuntimeError: If model loading fails

        Example:
            >>> model = MyASRModel()
            >>> model.load("openai/whisper-medium")  # Load pretrained
            >>> model.load("checkpoints/best_model")  # Load fine-tuned
        """
        raise NotImplementedError("Subclasses must implement load()")

    @abstractmethod
    def transcribe(self, audio_paths: Union[str, Path, List[Union[str, Path]]], **kwargs) -> Union[str, List[str]]:
        """
        Transcribe audio file(s) to text.

        Performs inference on one or more audio files and returns transcriptions.
        Should handle both single file and batch processing.

        Args:
            audio_paths: Path to audio file or list of paths
            **kwargs: Additional inference parameters (e.g., language, task,
                     beam_size, temperature)

        Returns:
            Transcription text (if single audio) or list of transcriptions (if batch)

        Raises:
            FileNotFoundError: If audio file(s) don't exist
            RuntimeError: If transcription fails

        Example:
            >>> model = MyASRModel()
            >>> model.load("openai/whisper-medium")
            >>> text = model.transcribe("audio.flac")
            >>> print(text)
            "hello world"
            >>>
            >>> texts = model.transcribe(["audio1.flac", "audio2.flac"])
            >>> print(texts)
            ["hello world", "goodbye world"]
        """
        raise NotImplementedError("Subclasses must implement transcribe()")

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint to disk.

        Saves the model weights and any associated artifacts (tokenizer, processor,
        config) to the specified path for later loading.

        Args:
            path: Directory path where model should be saved

        Raises:
            OSError: If saving fails due to permissions or disk space

        Example:
            >>> model = MyASRModel()
            >>> model.load("openai/whisper-medium")
            >>> # ... fine-tune model ...
            >>> model.save("checkpoints/my_finetuned_model")
        """
        raise NotImplementedError("Subclasses must implement save()")

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and configuration information.

        Returns a dictionary with model details useful for tracking experiments,
        debugging, and reproducibility.

        Returns:
            Dictionary containing model information. Should include at minimum:
                - name: Model architecture name (e.g., "whisper", "wav2vec")
                - variant: Model size/variant (e.g., "medium", "large-v2")
                - parameters: Number of model parameters (optional)
                - sample_rate: Expected audio sample rate
                - language: Language(s) supported (optional)

        Example:
            >>> model = MyASRModel()
            >>> model.load("openai/whisper-medium")
            >>> info = model.get_model_info()
            >>> print(info)
            {
                'name': 'whisper',
                'variant': 'medium',
                'parameters': 769000000,
                'sample_rate': 16000,
                'language': 'en'
            }
        """
        raise NotImplementedError("Subclasses must implement get_model_info()")
