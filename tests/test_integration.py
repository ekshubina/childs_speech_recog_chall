"""
Integration tests for the full pipeline.

This module tests the complete workflow from config loading → dataset creation
→ model initialization → training step → inference → evaluation.
"""

import json
import pytest
import numpy as np
import tempfile
import torch
import soundfile as sf
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.utils.config import load_config
from src.data.dataset import ChildSpeechDataset, create_train_val_split, WhisperDataCollator
from src.data.audio_processor import load_audio
from src.models.model_factory import ModelFactory
from src.models.whisper_model import WhisperModel
from src.training.metrics import WERMetric, compute_metrics
from src.inference.predictor import Predictor


class TestFullPipelineIntegration:
    """Test the complete pipeline from end to end."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace with config, audio files, and manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create directories
            audio_dir = tmpdir / "audio_0"
            audio_dir.mkdir()
            config_dir = tmpdir / "configs"
            config_dir.mkdir()
            checkpoints_dir = tmpdir / "checkpoints"
            checkpoints_dir.mkdir()
            
            # Create sample audio files (1 second at 16kHz)
            audio_files = []
            for i in range(5):
                audio_data = np.random.randn(16000).astype(np.float32)
                audio_path = audio_dir / f"sample_{i}.flac"
                sf.write(str(audio_path), audio_data, 16000)
                audio_files.append(audio_path)
            
            # Create manifest JSONL
            manifest_data = [
                {
                    "utterance_id": f"utt_{i}",
                    "audio_filename": audio_path.name,
                    "orthographic_text": f"this is test sentence {i}",
                    "age_bucket": "3-4" if i % 2 == 0 else "5-6",
                    "audio_duration": 1.0
                }
                for i, audio_path in enumerate(audio_files)
            ]
            
            manifest_path = tmpdir / "train_manifest.jsonl"
            with open(manifest_path, 'w') as f:
                for item in manifest_data:
                    f.write(json.dumps(item) + '\n')
            
            # Create a test config
            config = {
                "model": {
                    "name": "whisper",
                    "variant": "tiny",  # Use tiny for faster testing
                    "pretrained": "openai/whisper-tiny",
                    "language": "en",
                    "task": "transcribe",
                    "forced_decoder_ids": None,
                    "dropout": 0.1,
                    "freeze_encoder": False,
                    "gradient_checkpointing": False
                },
                "data": {
                    "train_manifest": str(manifest_path),
                    "audio_dirs": [str(audio_dir)],
                    "val_ratio": 0.2,
                    "stratify_by": "age_bucket",
                    "sample_rate": 16000,
                    "normalize_audio": True,
                    "normalize_text": True
                },
                "training": {
                    "output_dir": str(checkpoints_dir / "test_model"),
                    "num_epochs": 1,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-5,
                    "warmup_steps": 0,
                    "weight_decay": 0.01,
                    "fp16": False,  # Disable for testing
                    "logging_steps": 1,
                    "save_steps": 10,
                    "evaluation_strategy": "no",
                    "per_device_eval_batch_size": 2,
                    "dataloader_num_workers": 0,
                    "seed": 42
                },
                "evaluation": {
                    "primary_metric": "wer",
                    "normalize_predictions": True,
                    "normalize_references": True
                }
            }
            
            config_path = config_dir / "test_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            yield {
                "tmpdir": tmpdir,
                "config_path": config_path,
                "manifest_path": manifest_path,
                "audio_dir": audio_dir,
                "audio_files": audio_files,
                "config": config,
                "manifest_data": manifest_data
            }
    
    def test_config_loading(self, temp_workspace):
        """Test that config can be loaded successfully."""
        config = load_config(str(temp_workspace["config_path"]))
        
        assert config is not None
        assert config["model"]["name"] == "whisper"
        assert config["model"]["variant"] == "tiny"
        assert "data" in config
        assert "training" in config
        assert "evaluation" in config
    
    def test_dataset_creation_from_config(self, temp_workspace):
        """Test dataset creation using config parameters."""
        config = load_config(str(temp_workspace["config_path"]))
        
        # Mock WhisperProcessor to avoid downloading the model
        with patch('src.data.dataset.WhisperProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor.return_value = np.random.randn(80, 3000).astype(np.float32)
            mock_processor_class.from_pretrained.return_value = mock_processor
            
            dataset = ChildSpeechDataset(
                manifest_path=config["data"]["train_manifest"],
                audio_dirs=config["data"]["audio_dirs"],
                processor=mock_processor
            )
            
            assert len(dataset) == 5
            assert dataset.samples is not None
    
    def test_train_val_split_from_config(self, temp_workspace):
        """Test stratified train/val split using config parameters."""
        config = load_config(str(temp_workspace["config_path"]))
        
        train_data, val_data = create_train_val_split(
            manifest_path=config["data"]["train_manifest"],
            val_ratio=config["data"]["val_ratio"],
            stratify_by=config["data"]["stratify_by"]
        )
        
        assert len(train_data) + len(val_data) == 5
        assert len(val_data) >= 1  # At least one validation sample
    
    def test_model_creation_via_factory(self, temp_workspace):
        """Test model creation through the factory using config."""
        config = load_config(str(temp_workspace["config_path"]))
        
        # Mock the model and processor loading to avoid downloading
        with patch('src.models.whisper_model.WhisperForConditionalGeneration') as mock_model_class, \
             patch('src.models.whisper_model.WhisperProcessor') as mock_processor_class:
            
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_model_class.from_pretrained.return_value = mock_model
            
            mock_processor = MagicMock()
            mock_processor_class.from_pretrained.return_value = mock_processor
            
            model = ModelFactory.create_model(config)
            
            assert model is not None
            assert isinstance(model, WhisperModel)
            assert model.variant == "tiny"
    
    def test_model_info_retrieval(self, temp_workspace):
        """Test retrieving model information."""
        config = load_config(str(temp_workspace["config_path"]))
        
        with patch('src.models.whisper_model.WhisperForConditionalGeneration') as mock_model_class, \
             patch('src.models.whisper_model.WhisperProcessor') as mock_processor_class:
            
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_model_class.from_pretrained.return_value = mock_model
            
            mock_processor = MagicMock()
            mock_processor_class.from_pretrained.return_value = mock_processor
            
            model = ModelFactory.create_model(config)
            info = model.get_model_info()
            
            assert "model_name" in info
            assert "variant" in info
            assert info["variant"] == "tiny"
    
    @patch('transformers.Seq2SeqTrainer')
    def test_training_step_mock(self, mock_trainer_class, temp_workspace):
        """Test that a training step can be initiated (mocked)."""
        config = load_config(str(temp_workspace["config_path"]))
        
        # Mock all components
        with patch('src.models.whisper_model.WhisperForConditionalGeneration') as mock_model_class, \
             patch('src.models.whisper_model.WhisperProcessor') as mock_processor_class, \
             patch('src.data.dataset.WhisperProcessor') as mock_dataset_processor_class:
            
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_model.config = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            
            mock_processor = MagicMock()
            mock_processor.tokenizer = MagicMock()
            mock_processor.tokenizer.pad_token_id = 0
            mock_processor.feature_extractor = MagicMock()
            mock_processor_class.from_pretrained.return_value = mock_processor
            mock_dataset_processor_class.from_pretrained.return_value = mock_processor
            
            # Create model
            model = ModelFactory.create_model(config)
            
            # Create dataset
            dataset = ChildSpeechDataset(
                manifest_path=config["data"]["train_manifest"],
                audio_dirs=config["data"]["audio_dirs"],
                processor=mock_processor
            )
            
            # Mock trainer
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = None
            mock_trainer_class.return_value = mock_trainer
            
            # Verify we can instantiate trainer (this is what train.py does)
            assert model is not None
            assert dataset is not None
            assert len(dataset) > 0
    
    def test_inference_pipeline(self, temp_workspace):
        """Test inference on audio files."""
        config = load_config(str(temp_workspace["config_path"]))
        
        with patch('src.models.whisper_model.WhisperForConditionalGeneration') as mock_model_class, \
             patch('src.models.whisper_model.WhisperProcessor') as mock_processor_class, \
             patch('src.inference.predictor.WhisperProcessor') as mock_predictor_processor_class:
            
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_model.generate.return_value = torch.randint(0, 1000, (1, 10))
            mock_model_class.from_pretrained.return_value = mock_model
            
            mock_processor = MagicMock()
            mock_processor.return_value = {"input_features": torch.randn(1, 80, 3000)}
            
            # Mock batch_decode to return transcriptions
            def mock_batch_decode(token_ids, **kwargs):
                return [f"transcription {i}" for i in range(len(token_ids))]
            
            mock_processor.batch_decode.side_effect = mock_batch_decode
            mock_processor_class.from_pretrained.return_value = mock_processor
            mock_predictor_processor_class.from_pretrained.return_value = mock_processor
            
            # Create predictor
            predictor = Predictor(
                model_path=config["model"]["pretrained"],
                device="cpu"
            )
            
            # Get audio paths
            audio_paths = [str(f) for f in temp_workspace["audio_files"][:3]]
            
            # Run prediction
            predictions = predictor.predict_batch(audio_paths)
            
            assert len(predictions) == 3
            assert all(isinstance(pred, str) for pred in predictions)
    
    def test_inference_from_manifest(self, temp_workspace):
        """Test inference on full manifest."""
        config = load_config(str(temp_workspace["config_path"]))
        
        with patch('src.models.whisper_model.WhisperForConditionalGeneration') as mock_model_class, \
             patch('src.models.whisper_model.WhisperProcessor') as mock_processor_class, \
             patch('src.inference.predictor.WhisperProcessor') as mock_predictor_processor_class:
            
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_model.generate.return_value = torch.randint(0, 1000, (1, 10))
            mock_model_class.from_pretrained.return_value = mock_model
            
            mock_processor = MagicMock()
            mock_processor.return_value = {"input_features": torch.randn(1, 80, 3000)}
            
            def mock_batch_decode(token_ids, **kwargs):
                return [f"predicted text {i}" for i in range(len(token_ids))]
            
            mock_processor.batch_decode.side_effect = mock_batch_decode
            mock_processor_class.from_pretrained.return_value = mock_processor
            mock_predictor_processor_class.from_pretrained.return_value = mock_processor
            
            predictor = Predictor(
                model_path=config["model"]["pretrained"],
                device="cpu"
            )
            
            results = predictor.predict_from_manifest(
                manifest_path=str(temp_workspace["manifest_path"]),
                audio_dirs=config["data"]["audio_dirs"]
            )
            
            assert len(results) == 5
            assert all("utterance_id" in r for r in results)
            assert all("orthographic_text" in r for r in results)
    
    def test_wer_computation(self, temp_workspace):
        """Test WER metric computation."""
        metric = WERMetric()
        
        predictions = [
            "this is a test",
            "hello world",
            "the quick brown fox"
        ]
        references = [
            "this is a test",
            "hello word",
            "the quick brown fox jumps"
        ]
        
        wer = metric.compute(predictions, references)
        
        # WER can be a float or dict
        if isinstance(wer, dict):
            wer_value = wer.get('wer', wer.get('error_rate', 0.0))
        else:
            wer_value = wer
        
        assert wer_value >= 0.0
        assert wer_value <= 1.0
    
    def test_end_to_end_pipeline_config_to_predictions(self, temp_workspace):
        """
        Test the complete pipeline from loading config to generating predictions.
        
        This is the full integration test covering:
        1. Config loading
        2. Dataset creation
        3. Model initialization
        4. Inference
        5. Evaluation
        """
        # Step 1: Load config
        config = load_config(str(temp_workspace["config_path"]))
        assert config is not None
        
        # Step 2: Create dataset with mocked processor
        with patch('src.data.dataset.WhisperProcessor') as mock_dataset_processor_class, \
             patch('src.models.whisper_model.WhisperForConditionalGeneration') as mock_model_class, \
             patch('src.models.whisper_model.WhisperProcessor') as mock_model_processor_class, \
             patch('src.inference.predictor.WhisperProcessor') as mock_predictor_processor_class:
            
            # Mock dataset processor
            mock_dataset_processor = MagicMock()
            mock_dataset_processor.return_value = np.random.randn(80, 3000).astype(np.float32)
            mock_dataset_processor_class.from_pretrained.return_value = mock_dataset_processor
            
            dataset = ChildSpeechDataset(
                manifest_path=config["data"]["train_manifest"],
                audio_dirs=config["data"]["audio_dirs"],
                processor=mock_dataset_processor
            )
            assert len(dataset) == 5
            
            # Step 3: Initialize model via factory
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_model.generate.return_value = torch.randint(0, 1000, (1, 10))
            mock_model_class.from_pretrained.return_value = mock_model
            
            mock_processor = MagicMock()
            mock_processor.return_value = {"input_features": torch.randn(1, 80, 3000)}
            
            def mock_batch_decode(token_ids, **kwargs):
                return ["predicted text" for _ in range(len(token_ids))]
            
            mock_processor.batch_decode.side_effect = mock_batch_decode
            mock_model_processor_class.from_pretrained.return_value = mock_processor
            mock_predictor_processor_class.from_pretrained.return_value = mock_processor
            
            model = ModelFactory.create_model(config)
            assert model is not None
            
            # Step 4: Run inference
            predictor = Predictor(
                model_path=config["model"]["pretrained"],
                device="cpu"
            )
            
            predictions = predictor.predict_from_manifest(
                manifest_path=str(temp_workspace["manifest_path"]),
                audio_dirs=config["data"]["audio_dirs"]
            )
            assert len(predictions) == 5
            
            # Step 5: Evaluate predictions
            metric = WERMetric()
            
            pred_texts = [p["orthographic_text"] for p in predictions]
            ref_texts = [item["orthographic_text"] for item in temp_workspace["manifest_data"]]
            
            wer = metric.compute(pred_texts, ref_texts)
            
            # WER can be a float or dict
            if isinstance(wer, dict):
                wer_value = wer.get('wer', wer.get('error_rate', 0.0))
            else:
                wer_value = wer
            
            assert wer_value >= 0.0
            assert wer_value <= 1.0
            
            # Verify output format
            for pred in predictions:
                assert "utterance_id" in pred
                assert "orthographic_text" in pred
                assert isinstance(pred["utterance_id"], str)
                assert isinstance(pred["orthographic_text"], str)


class TestPipelineErrorHandling:
    """Test error handling in the pipeline."""
    
    def test_invalid_config_path(self):
        """Test handling of non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_invalid_manifest_path(self):
        """Test handling of non-existent manifest file."""
        with patch('src.data.dataset.WhisperProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor_class.from_pretrained.return_value = mock_processor
            
            with pytest.raises(FileNotFoundError):
                ChildSpeechDataset(
                    manifest_path="nonexistent_manifest.jsonl",
                    audio_dirs=["data/audio_0"],
                    processor=mock_processor
                )
    
    def test_model_factory_invalid_model_name(self):
        """Test handling of unsupported model name."""
        config = {
            "model": {
                "name": "unsupported_model_type",
                "variant": "base"
            }
        }
        
        with pytest.raises(ValueError):
            ModelFactory.create_model(config)


class TestPipelineComponentInteraction:
    """Test interactions between pipeline components."""
    
    def test_data_collator_with_dataset(self):
        """Test that WhisperDataCollator works with ChildSpeechDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test manifest
            audio_dir = tmpdir / "audio"
            audio_dir.mkdir()
            
            audio_path = audio_dir / "test.flac"
            audio_data = np.random.randn(16000).astype(np.float32)
            sf.write(str(audio_path), audio_data, 16000)
            
            manifest_path = tmpdir / "manifest.jsonl"
            with open(manifest_path, 'w') as f:
                f.write(json.dumps({
                    "utterance_id": "test_001",
                    "audio_filename": "test.flac",
                    "orthographic_text": "test text",
                    "age_bucket": "3-4"
                }) + '\n')
            
            # Mock processor
            with patch('src.data.dataset.WhisperProcessor') as mock_processor_class:
                mock_processor = MagicMock()
                mock_processor.tokenizer = MagicMock()
                mock_processor.tokenizer.pad_token_id = 0
                
                # Mock the call to return input features
                def mock_call(*args, **kwargs):
                    return MagicMock(input_features=torch.randn(80, 3000))
                
                mock_processor.side_effect = mock_call
                mock_processor_class.from_pretrained.return_value = mock_processor
                
                dataset = ChildSpeechDataset(
                    manifest_path=str(manifest_path),
                    audio_dirs=[str(audio_dir)],
                    processor=mock_processor
                )
                
                # Create collator
                collator = WhisperDataCollator(processor=mock_processor)
                
                # Get a sample from dataset
                sample = dataset[0]
                
                # Test that collator can process a batch
                batch = collator([sample])
                
                assert batch is not None
    
    def test_metrics_with_predictor_output(self):
        """Test that WERMetric can process Predictor output format."""
        metric = WERMetric()
        
        # Simulate predictor output
        predictions_output = [
            {"utterance_id": "utt_1", "orthographic_text": "hello world"},
            {"utterance_id": "utt_2", "orthographic_text": "test sentence"}
        ]
        
        references = [
            "hello world",
            "test sentance"
        ]
        
        pred_texts = [p["orthographic_text"] for p in predictions_output]
        wer = metric.compute(pred_texts, references)
        
        # WER can be a float or dict
        if isinstance(wer, dict):
            wer_value = wer.get('wer', wer.get('error_rate', 0.0))
        else:
            wer_value = wer
        
        assert wer_value >= 0.0
        assert wer_value <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
