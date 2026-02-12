"""
Inference pipeline for batch predictions.

This module provides a Predictor class for performing batch inference
on audio files using trained Whisper models.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.data.audio_processor import load_audio, resample_audio, convert_to_mono

logger = logging.getLogger(__name__)


class Predictor:
    """
    Predictor for batch inference on audio files.
    
    Loads a trained Whisper model and performs transcription on batches
    of audio files. Supports loading from manifests and writing predictions
    to JSONL format.
    
    Attributes:
        model: WhisperForConditionalGeneration model
        processor: WhisperProcessor for audio preprocessing
        device: Device to run inference on ('cuda', 'cpu', 'mps')
        batch_size: Batch size for inference
    
    Example:
        >>> predictor = Predictor(
        ...     model_path='checkpoints/finetuned_model',
        ...     device='cuda',
        ...     batch_size=16
        ... )
        >>> 
        >>> # Single file prediction
        >>> text = predictor.predict_single('audio.flac')
        >>> 
        >>> # Batch prediction
        >>> audio_paths = ['audio1.flac', 'audio2.flac', 'audio3.flac']
        >>> predictions = predictor.predict_batch(audio_paths)
        >>> 
        >>> # Predict from manifest
        >>> results = predictor.predict_from_manifest(
        ...     'data/test.jsonl',
        ...     audio_dirs=['data/audio_0', 'data/audio_1']
        ... )
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        batch_size: int = 16,
        language: str = 'english',
        task: str = 'transcribe'
    ):
        """
        Initialize Predictor with trained model.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run on. If None, auto-detects.
            batch_size: Batch size for inference
            language: Language for transcription
            task: Task type ('transcribe' or 'translate')
        
        Raises:
            FileNotFoundError: If model_path doesn't exist
            RuntimeError: If model loading fails
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        self.batch_size = batch_size
        self.language = language
        self.task = task
        
        logger.info(f"Initializing Predictor")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Batch size: {batch_size}")
        
        # Load model and processor
        self._load_model()
    
    def _load_model(self):
        """Load model and processor from checkpoint."""
        logger.info("Loading model and processor...")
        
        try:
            # Load processor
            self.processor = WhisperProcessor.from_pretrained(
                str(self.model_path),
                language=self.language,
                task=self.task
            )
            
            # Load model
            self.model = WhisperForConditionalGeneration.from_pretrained(
                str(self.model_path)
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def predict_single(self, audio_path: Union[str, Path]) -> str:
        """
        Predict transcription for a single audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Transcribed text
        
        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        predictions = self.predict_batch([audio_path])
        return predictions[0]
    
    def predict_batch(
        self,
        audio_paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> List[str]:
        """
        Predict transcriptions for a batch of audio files.
        
        Args:
            audio_paths: List of paths to audio files
            show_progress: Whether to show progress bar
        
        Returns:
            List of transcribed texts
        
        Raises:
            FileNotFoundError: If any audio file doesn't exist
            RuntimeError: If prediction fails
        """
        audio_paths = [Path(p) for p in audio_paths]
        
        # Verify all files exist
        for path in audio_paths:
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
        
        logger.info(f"Predicting {len(audio_paths)} audio files...")
        
        all_predictions = []
        
        # Process in batches
        num_batches = (len(audio_paths) + self.batch_size - 1) // self.batch_size
        
        iterator = range(0, len(audio_paths), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Predicting")
        
        for i in iterator:
            batch_paths = audio_paths[i:i + self.batch_size]
            
            try:
                # Load and process audio
                batch_audio = []
                for path in batch_paths:
                    audio, sr = load_audio(str(path), sr=16000)
                    audio = convert_to_mono(audio)
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
                        language=self.language,
                        task=self.task,
                        max_length=225
                    )
                
                # Decode transcriptions
                batch_predictions = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                
                all_predictions.extend(batch_predictions)
                
            except Exception as e:
                logger.error(f"Prediction failed for batch starting at index {i}: {e}")
                raise RuntimeError(f"Prediction failed: {e}") from e
        
        logger.info(f"Successfully predicted {len(all_predictions)} transcriptions")
        
        return all_predictions
    
    def predict_from_manifest(
        self,
        manifest_path: Union[str, Path],
        audio_dirs: List[Union[str, Path]],
        output_path: Optional[Union[str, Path]] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict transcriptions from a JSONL manifest file.
        
        Loads audio file information from manifest, performs predictions,
        and optionally writes results to JSONL output file.
        
        Args:
            manifest_path: Path to input JSONL manifest
            audio_dirs: List of directories to search for audio files
            output_path: Optional path to write predictions JSONL
            show_progress: Whether to show progress bar
        
        Returns:
            List of dictionaries with 'utterance_id' and 'orthographic_text'
        
        Raises:
            FileNotFoundError: If manifest doesn't exist or audio files not found
        """
        manifest_path = Path(manifest_path)
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        logger.info(f"Loading manifest: {manifest_path}")
        
        # Load manifest
        manifest_data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                manifest_data.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(manifest_data)} entries from manifest")
        
        # Find audio files
        audio_dirs = [Path(d) for d in audio_dirs]
        audio_paths = []
        utterance_ids = []
        
        for entry in manifest_data:
            utterance_id = entry['utterance_id']
            audio_filename = entry['audio_file']
            
            # Search for audio file in all directories
            audio_path = None
            for audio_dir in audio_dirs:
                candidate_path = audio_dir / audio_filename
                if candidate_path.exists():
                    audio_path = candidate_path
                    break
            
            if audio_path is None:
                raise FileNotFoundError(
                    f"Audio file not found: {audio_filename} "
                    f"(searched in {len(audio_dirs)} directories)"
                )
            
            audio_paths.append(audio_path)
            utterance_ids.append(utterance_id)
        
        logger.info(f"Found all {len(audio_paths)} audio files")
        
        # Perform predictions
        predictions = self.predict_batch(audio_paths, show_progress=show_progress)
        
        # Create results
        results = []
        for utterance_id, prediction in zip(utterance_ids, predictions):
            results.append({
                'utterance_id': utterance_id,
                'orthographic_text': prediction
            })
        
        # Write to output file if specified
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Writing predictions to: {output_path}")
            
            with open(output_path, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            logger.info(f"Wrote {len(results)} predictions to {output_path}")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_path': str(self.model_path),
            'device': self.device,
            'batch_size': self.batch_size,
            'language': self.language,
            'task': self.task,
            'parameters': sum(p.numel() for p in self.model.parameters())
        }
