"""
Audio processing utilities for children's speech recognition.

This module provides functions for loading, resampling, converting to mono,
and normalizing audio files for ASR model input.
"""

import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Union

logger = logging.getLogger(__name__)


def load_audio(path: Union[str, Path], sr: int = 16000) -> Tuple[np.ndarray, float]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        path: Path to audio file (supports FLAC, WAV, MP3, etc.)
        sr: Target sample rate in Hz (default: 16000)
        
    Returns:
        Tuple of (audio_array, sample_rate) where:
            - audio_array: numpy array of audio samples (mono)
            - sample_rate: sample rate of the returned audio
            
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio file cannot be loaded
        
    Example:
        >>> audio, sr = load_audio('path/to/audio.flac', sr=16000)
        >>> print(audio.shape, sr)
        (48000,) 16000
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    try:
        # Load audio with librosa (automatically converts to mono and resamples)
        audio, sample_rate = librosa.load(path, sr=sr, mono=True)
        
        logger.debug(f"Loaded audio from {path.name}: {len(audio)} samples at {sample_rate}Hz")
        
        return audio, sample_rate
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {path}: {str(e)}")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to a different sample rate.
    
    Args:
        audio: Audio array to resample
        orig_sr: Original sample rate in Hz
        target_sr: Target sample rate in Hz
        
    Returns:
        Resampled audio array
        
    Example:
        >>> audio_48k = np.random.randn(48000)
        >>> audio_16k = resample_audio(audio_48k, orig_sr=48000, target_sr=16000)
        >>> len(audio_16k)
        16000
    """
    if orig_sr == target_sr:
        logger.debug(f"Sample rates match ({orig_sr}Hz), skipping resampling")
        return audio
    
    logger.debug(f"Resampling from {orig_sr}Hz to {target_sr}Hz")
    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    return resampled


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.
    
    Args:
        audio: Audio array, shape (samples,) for mono or (channels, samples) for stereo
        
    Returns:
        Mono audio array of shape (samples,)
        
    Example:
        >>> stereo_audio = np.random.randn(2, 16000)  # 2 channels
        >>> mono_audio = convert_to_mono(stereo_audio)
        >>> mono_audio.shape
        (16000,)
    """
    if audio.ndim == 1:
        # Already mono
        logger.debug("Audio is already mono")
        return audio
    
    elif audio.ndim == 2:
        # Convert stereo to mono by averaging channels
        if audio.shape[0] == 2:
            # Shape is (channels, samples)
            logger.debug("Converting stereo to mono (averaging channels)")
            return np.mean(audio, axis=0)
        elif audio.shape[1] == 2:
            # Shape is (samples, channels)
            logger.debug("Converting stereo to mono (averaging channels)")
            return np.mean(audio, axis=1)
        else:
            logger.warning(f"Unexpected audio shape {audio.shape}, taking mean across axis 0")
            return np.mean(audio, axis=0)
    else:
        raise ValueError(f"Unexpected audio array dimensions: {audio.ndim}. Expected 1D (mono) or 2D (stereo)")


def normalize_audio(audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
    """
    Normalize audio amplitude to target peak level.
    
    Scales audio so that the maximum absolute value equals target_level.
    This prevents clipping while maximizing signal strength.
    
    Args:
        audio: Audio array to normalize
        target_level: Target peak amplitude level (default: 0.95, leaving headroom)
        
    Returns:
        Normalized audio array
        
    Example:
        >>> audio = np.array([0.1, -0.5, 0.3, -0.2])
        >>> normalized = normalize_audio(audio, target_level=0.95)
        >>> np.max(np.abs(normalized))
        0.95
    """
    if len(audio) == 0:
        logger.warning("Empty audio array, returning as-is")
        return audio
    
    max_val = np.max(np.abs(audio))
    
    if max_val == 0:
        logger.warning("Audio contains only silence (all zeros), cannot normalize")
        return audio
    
    # Scale to target level
    scaling_factor = target_level / max_val
    normalized = audio * scaling_factor
    
    logger.debug(f"Normalized audio: max_val={max_val:.4f}, scaling_factor={scaling_factor:.4f}")
    
    return normalized
