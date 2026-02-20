"""
Whisper model implementation for ASR.

This module provides a concrete implementation of the BaseASRModel interface
for OpenAI's Whisper model using Hugging Face transformers.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    GenerationConfig
)

from src.models.base_model import BaseASRModel

logger = logging.getLogger(__name__)


class WhisperModel(BaseASRModel):
    """
    Whisper ASR model implementation.
    
    Wraps Hugging Face's WhisperForConditionalGeneration and WhisperProcessor
    to provide a consistent interface for loading, fine-tuning, and inference.
    
    Attributes:
        variant (str): Whisper model variant (e.g., 'medium', 'large-v2')
        device (str): Device to run model on ('cuda', 'cpu', 'mps')
        model (WhisperForConditionalGeneration): The Whisper model
        processor (WhisperProcessor): The Whisper processor (feature extractor + tokenizer)
    
    Example:
        >>> model = WhisperModel(variant='small', device='cuda')
        >>> model.load('openai/whisper-small')
        >>> transcription = model.transcribe('audio.flac')
        >>> model.save('checkpoints/finetuned_model')
    """

    def __init__(self, variant: str = 'small', device: Optional[str] = None):
        """
        Initialize WhisperModel.
        
        Args:
            variant: Whisper model variant ('tiny', 'base', 'small', 'medium', 
                    'large', 'large-v2', 'large-v3')
            device: Device to run model on. If None, automatically selects
                   'cuda' if available, 'mps' if on Mac with Apple Silicon, 
                   otherwise 'cpu'
        """
        self.variant = variant

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS (Apple Silicon GPU) for training")
            else:
                device = 'cpu'

        self.device = device
        self.model: Optional[WhisperForConditionalGeneration] = None
        self.processor: Optional[WhisperProcessor] = None

        logger.info(f"Initialized WhisperModel with variant='{variant}', device='{device}'")

    def load(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load Whisper model from checkpoint or pretrained weights.
        
        If checkpoint_path is a Hugging Face model ID (e.g., 'openai/whisper-medium'),
        loads pretrained weights. If it's a local path, loads fine-tuned checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint or HuggingFace model ID.
                           If None, uses f'openai/whisper-{self.variant}'
        
        Raises:
            FileNotFoundError: If local checkpoint path doesn't exist
            RuntimeError: If model loading fails
        """
        if checkpoint_path is None:
            checkpoint_path = f'openai/whisper-{self.variant}'

        checkpoint_path = str(checkpoint_path)
        logger.info(f"Loading Whisper model from: {checkpoint_path}")

        try:
            # Load processor (feature extractor + tokenizer)
            self.processor = WhisperProcessor.from_pretrained(
                checkpoint_path,
                language='english',
                task='transcribe'
            )

            # Load model
            self.model = WhisperForConditionalGeneration.from_pretrained(
                checkpoint_path
            )

            # Configure for English transcription
            # Setting forced_decoder_ids=None allows the model to predict language
            # but we'll set language in generation config instead
            self.model.config.forced_decoder_ids = None

            # Ensure suppress_tokens is not in config (transformers requirement)
            if hasattr(self.model.config, 'suppress_tokens'):
                delattr(self.model.config, 'suppress_tokens')

            # Set suppress_tokens in generation_config where it belongs
            self.model.generation_config.suppress_tokens = []

            # Set language to English in generation config
            self.model.generation_config.language = 'english'
            self.model.generation_config.task = 'transcribe'

            # Move model to device
            self.model.to(self.device)

            logger.info(f"Successfully loaded Whisper model on {self.device}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        except Exception as e:
            logger.error(f"Failed to load model from {checkpoint_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def transcribe(
        self, 
        audio_paths: Union[str, Path, List[Union[str, Path]]],
        batch_size: int = 8,
        language: str = 'english',
        task: str = 'transcribe',
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Transcribe audio file(s) to text.
        
        Args:
            audio_paths: Path to audio file or list of paths
            batch_size: Batch size for processing multiple files
            language: Language for transcription (default: 'english')
            task: Task type ('transcribe' or 'translate')
            **kwargs: Additional generation parameters (e.g., max_length, num_beams)
        
        Returns:
            Transcription text (if single audio) or list of transcriptions (if batch)
        
        Raises:
            RuntimeError: If model not loaded or transcription fails
            FileNotFoundError: If audio file doesn't exist
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Handle single file vs batch
        single_input = isinstance(audio_paths, (str, Path))
        if single_input:
            audio_paths = [audio_paths]

        # Convert to list of Path objects
        audio_paths = [Path(p) for p in audio_paths]

        # Verify all files exist
        for audio_path in audio_paths:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing {len(audio_paths)} audio file(s)")

        transcriptions = []

        # Process in batches
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]

            try:
                # Load and process audio
                import librosa
                batch_audio = []
                for path in batch_paths:
                    audio, sr = librosa.load(str(path), sr=16000, mono=True)
                    batch_audio.append(audio)

                # Process with WhisperProcessor
                inputs = self.processor(
                    batch_audio,
                    sampling_rate=16000,
                    return_tensors='pt',
                    padding=True
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate transcriptions
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        inputs['input_features'],
                        language=language,
                        task=task,
                        **kwargs
                    )

                # Decode transcriptions
                batch_transcriptions = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                transcriptions.extend(batch_transcriptions)

                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(audio_paths) + batch_size - 1)//batch_size}")

            except Exception as e:
                logger.error(f"Transcription failed for batch starting at index {i}: {e}")
                raise RuntimeError(f"Transcription failed: {e}") from e

        # Return single string if single input, else list
        if single_input:
            return transcriptions[0]
        return transcriptions

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint to disk.
        
        Saves both the model and processor to enable complete checkpoint restoration.
        
        Args:
            path: Directory path where model should be saved
        
        Raises:
            RuntimeError: If model not loaded
            OSError: If saving fails
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Cannot save.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to: {path}")

        try:
            # Save model and processor
            self.model.save_pretrained(str(path))
            self.processor.save_pretrained(str(path))

            logger.info(f"Successfully saved model to {path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise OSError(f"Model saving failed: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and configuration.
        
        Returns:
            Dictionary with model information including name, variant, 
            parameters count, device, and configuration details.
        """
        info = {
            'name': 'whisper',
            'variant': self.variant,
            'device': self.device,
            'sample_rate': 16000,
            'language': 'english',
            'task': 'transcribe'
        }

        if self.model is not None:
            info['parameters'] = sum(p.numel() for p in self.model.parameters())
            info['trainable_parameters'] = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            info['model_loaded'] = True
        else:
            info['model_loaded'] = False

        return info


def prepare_model_for_finetuning(
    model: WhisperForConditionalGeneration,
    freeze_encoder: bool = False,
    dropout: float = 0.1,
    language: str = 'english',
    task: str = 'transcribe'
) -> WhisperForConditionalGeneration:
    """
    Prepare Whisper model for fine-tuning on children's speech.
    
    Configures model settings including language, task, dropout, and optionally
    freezes encoder layers for faster training with less memory.
    
    Args:
        model: WhisperForConditionalGeneration model to prepare
        freeze_encoder: If True, freeze encoder weights (only train decoder)
        dropout: Dropout probability for regularization
        language: Target language for transcription
        task: Task type ('transcribe' or 'translate')
    
    Returns:
        Configured model ready for fine-tuning
    
    Example:
        >>> model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')
        >>> model = prepare_model_for_finetuning(model, freeze_encoder=False, dropout=0.1)
        >>> # Now ready for training
    """
    logger.info("Preparing model for fine-tuning")

    # Set language and task
    model.config.forced_decoder_ids = None

    # Ensure suppress_tokens is not in config (transformers requirement)
    if hasattr(model.config, 'suppress_tokens'):
        delattr(model.config, 'suppress_tokens')

    # Set suppress_tokens in generation_config where it belongs
    model.generation_config.suppress_tokens = []
    model.generation_config.language = language
    model.generation_config.task = task

    # Disable use_cache - incompatible with gradient checkpointing and not needed during training
    model.config.use_cache = False

    # Set dropout
    model.config.dropout = dropout
    model.config.attention_dropout = dropout
    model.config.activation_dropout = dropout

    # Optionally freeze encoder
    if freeze_encoder:
        logger.info("Freezing encoder weights")
        for param in model.model.encoder.parameters():
            param.requires_grad = False

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
    else:
        logger.info("Training full model (encoder + decoder)")

    return model
