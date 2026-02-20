"""
PyTorch Dataset for children's speech recognition.

This module provides the ChildSpeechDataset class for loading and processing
children's speech data from JSONL manifests with multi-directory audio support.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from src.data.audio_processor import load_audio, normalize_audio

logger = logging.getLogger(__name__)


class ChildSpeechDataset(Dataset):
    """
    PyTorch Dataset for children's speech with multi-directory audio support.

    This dataset loads audio utterances from JSONL manifests and processes them
    for ASR model training/inference. Audio files are searched across multiple
    directories to handle the split data structure (audio_0, audio_1, audio_2).

    Args:
        manifest_path: Path to JSONL file with utterance metadata
        audio_dirs: List of directories to search for audio files
        processor: WhisperProcessor for audio feature extraction
        text_normalizer: Optional text normalizer for transcriptions
        sample_rate: Target audio sample rate (default: 16000)
        normalize_audio_amplitude: Whether to normalize audio amplitude (default: True)
        max_audio_length: Maximum audio length in seconds (default: 30)

    Example:
        >>> from transformers import WhisperProcessor
        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        >>> dataset = ChildSpeechDataset(
        ...     manifest_path="data/train_word_transcripts.jsonl",
        ...     audio_dirs=["data/audio_0", "data/audio_1", "data/audio_2"],
        ...     processor=processor
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
    """

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        audio_dirs: Optional[List[Union[str, Path]]] = None,
        processor: Optional[WhisperProcessor] = None,
        text_normalizer: Optional[object] = None,
        sample_rate: int = 16000,
        normalize_audio_amplitude: bool = True,
        max_audio_length: float = 30.0,
        samples: Optional[List[Dict]] = None,
    ):
        self.audio_dirs = [Path(d) for d in audio_dirs] if audio_dirs else []
        self.processor = processor
        self.text_normalizer = text_normalizer
        self.sample_rate = sample_rate
        self.normalize_audio_amplitude = normalize_audio_amplitude
        self.max_audio_length = max_audio_length

        # Support both manifest_path and pre-loaded samples
        if samples is not None:
            # Use pre-loaded samples (e.g., from train/val split)
            self.samples = samples
            self.manifest_path = None
            logger.info(f"Initialized with {len(self.samples)} pre-loaded samples")
        elif manifest_path is not None:
            # Load from manifest file
            self.manifest_path = Path(manifest_path)

            # Validate inputs
            if not self.manifest_path.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

            # Load manifest
            self.samples = self._load_manifest()
            logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")
        else:
            raise ValueError("Must provide either manifest_path or samples")

        # Validate audio directories exist
        for audio_dir in self.audio_dirs:
            if not audio_dir.exists():
                logger.warning(f"Audio directory not found: {audio_dir}")

    def _load_manifest(self) -> List[Dict]:
        """
        Load samples from JSONL manifest file.

        Returns:
            List of sample dictionaries with utterance metadata
        """
        samples = []

        with open(self.manifest_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        return samples

    def _find_audio_file(self, audio_path: str) -> Optional[Path]:
        """
        Search for audio file across multiple directories.

        The audio_path in JSONL uses format like "audio/U_xxxxx.flac".
        We need to search across audio_0, audio_1, audio_2 directories.

        Args:
            audio_path: Relative audio path from manifest (e.g., "audio/U_00003c3ae1c35c6f.flac")

        Returns:
            Full path to audio file if found, None otherwise

        Example:
            >>> dataset._find_audio_file("audio/U_00003c3ae1c35c6f.flac")
            PosixPath('data/audio_0/U_00003c3ae1c35c6f.flac')
        """
        # Extract just the filename from the path
        filename = Path(audio_path).name

        # Search in each audio directory
        for audio_dir in self.audio_dirs:
            full_path = audio_dir / filename
            if full_path.exists():
                return full_path

        # If not found, return None
        logger.warning(f"Audio file not found in any directory: {filename}")
        return None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process a single sample.

        Args:
            idx: Index of sample to load

        Returns:
            Dictionary containing:
                - input_features: Processed audio features for model input
                - labels: Tokenized transcription for training
                - utterance_id: Unique identifier for the utterance
                - (optional) other metadata fields

        Raises:
            FileNotFoundError: If audio file cannot be found
            RuntimeError: If audio processing fails
        """
        sample = self.samples[idx]

        # Find audio file
        audio_path = self._find_audio_file(sample['audio_path'])
        if audio_path is None:
            raise FileNotFoundError(
                f"Audio file not found for utterance {sample['utterance_id']}: {sample['audio_path']}"
            )

        # Load audio
        try:
            audio, sr = load_audio(audio_path, sr=self.sample_rate)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio for utterance {sample['utterance_id']}: {e}"
            )

        # Normalize audio amplitude if requested
        if self.normalize_audio_amplitude:
            audio = normalize_audio(audio)

        # Truncate if longer than max length
        max_samples = int(self.max_audio_length * self.sample_rate)
        if len(audio) > max_samples:
            logger.debug(
                f"Truncating audio from {len(audio)} to {max_samples} samples "
                f"for utterance {sample['utterance_id']}"
            )
            audio = audio[:max_samples]

        # Process audio with WhisperProcessor to get input features
        input_features = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features[0]  # Remove batch dimension

        # Process transcription text
        transcription = sample['orthographic_text']

        # Apply text normalization if available
        if self.text_normalizer is not None:
            transcription = self.text_normalizer(transcription)

        # Tokenize transcription for labels
        # Whisper's decoder has a hard limit of 448 tokens; truncate to avoid runtime errors
        labels = self.processor.tokenizer(
            transcription,
            return_tensors="pt",
            max_length=448,
            truncation=True,
        ).input_ids[
            0
        ]  # Remove batch dimension

        # Return processed sample
        return {
            'input_features': input_features,
            'labels': labels,
            'utterance_id': sample['utterance_id'],
            'age_bucket': sample.get('age_bucket', 'unknown'),
            'audio_duration_sec': sample.get('audio_duration_sec', 0.0),
        }


def create_train_val_split(
    manifest_path: Union[str, Path],
    val_ratio: float = 0.1,
    stratify_by: str = 'age_bucket',
    random_seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create stratified train/validation split from manifest file.

    Splits the dataset while maintaining proportional representation of
    stratification groups (e.g., age buckets) in both train and validation sets.

    Args:
        manifest_path: Path to JSONL manifest file
        val_ratio: Fraction of data to use for validation (default: 0.1)
        stratify_by: Field name to stratify by (default: 'age_bucket')
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_samples, val_samples) where each is a list of sample dicts

    Example:
        >>> train, val = create_train_val_split(
        ...     "data/train_word_transcripts.jsonl",
        ...     val_ratio=0.1,
        ...     stratify_by='age_bucket'
        ... )
        >>> print(f"Train: {len(train)}, Val: {len(val)}")
        Train: 86015, Val: 9558
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # Load all samples
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
                continue

    if len(samples) == 0:
        raise ValueError(f"No valid samples found in {manifest_path}")

    # Extract stratification labels
    stratify_labels = None
    if stratify_by:
        stratify_labels = [sample.get(stratify_by, 'unknown') for sample in samples]

        # Log distribution
        from collections import Counter
        distribution = Counter(stratify_labels)
        logger.info(f"Dataset distribution by {stratify_by}: {dict(distribution)}")

    # Perform stratified split
    try:
        train_samples, val_samples = train_test_split(
            samples,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=stratify_labels
        )

        logger.info(
            f"Split dataset: {len(train_samples)} train, {len(val_samples)} val "
            f"({val_ratio*100:.1f}% validation)"
        )

        return train_samples, val_samples

    except Exception as e:
        logger.error(f"Failed to create stratified split: {e}")
        raise


def save_manifest(samples: List[Dict], output_path: Union[str, Path]) -> None:
    """
    Save samples to JSONL manifest file.

    Args:
        samples: List of sample dictionaries
        output_path: Path to output JSONL file

    Example:
        >>> train, val = create_train_val_split("data/train.jsonl")
        >>> save_manifest(train, "data/train_split.jsonl")
        >>> save_manifest(val, "data/val_split.jsonl")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    logger.info(f"Saved {len(samples)} samples to {output_path}")


class WhisperDataCollator:
    """
    Data collator for Whisper models with dynamic padding.

    This collator handles batching of variable-length audio features and
    transcription labels, applying appropriate padding for efficient training.

    Args:
        processor: WhisperProcessor for tokenization
        padding: Padding strategy ('longest', 'max_length', or False)
        max_length: Maximum sequence length for padding (optional)

    Example:
        >>> from transformers import WhisperProcessor
        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        >>> collator = WhisperDataCollator(processor=processor)
        >>> batch = collator([dataset[0], dataset[1], dataset[2]])
    """

    def __init__(
        self,
        processor: WhisperProcessor,
        padding: Union[bool, str] = "longest",
        max_length: Optional[int] = None,
    ):
        self.processor = processor
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features with dynamic padding.

        Args:
            features: List of feature dictionaries from dataset

        Returns:
            Batched and padded features ready for model input
        """
        # Extract input features (already processed by WhisperProcessor)
        input_features = [f['input_features'] for f in features]
        input_features = torch.stack(input_features)

        # Extract labels and pad them
        labels = [f['labels'] for f in features]

        # Pad labels to same length in batch
        max_label_length = max(len(l) for l in labels)

        # Pad with -100 (ignored in loss computation)
        padded_labels = []
        for label in labels:
            padding_length = max_label_length - len(label)
            padded_label = torch.cat([
                label,
                torch.full((padding_length,), -100, dtype=label.dtype)
            ])
            padded_labels.append(padded_label)

        padded_labels = torch.stack(padded_labels)

        # Prepare batch (only include model inputs, not metadata)
        # Metadata fields like utterance_id, age_bucket should not be passed to model.generate()
        batch = {
            'input_features': input_features,
            'labels': padded_labels,
        }

        return batch
