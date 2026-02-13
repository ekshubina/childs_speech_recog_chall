"""
Tests for model components.

This module tests BaseASRModel interface compliance, model loading
(pretrained and checkpoint), and basic transcription functionality.
"""

import json
import pytest
import numpy as np
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from abc import ABC

from src.models.base_model import BaseASRModel
from src.models.whisper_model import WhisperModel, prepare_model_for_finetuning
from src.models.model_factory import ModelFactory


class TestBaseASRModel:
    """Tests for BaseASRModel interface compliance."""
    
    def test_basemodel_is_abstract(self):
        """Test that BaseASRModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseASRModel()
    
    def test_basemodel_requires_load_implementation(self):
        """Test that subclasses must implement load() method."""
        class IncompleteModel(BaseASRModel):
            def transcribe(self, audio_paths, **kwargs):
                pass
            def save(self, path):
                pass
            def get_model_info(self):
                pass
        
        with pytest.raises(TypeError):
            IncompleteModel()
    
    def test_basemodel_requires_transcribe_implementation(self):
        """Test that subclasses must implement transcribe() method."""
        class IncompleteModel(BaseASRModel):
            def load(self, checkpoint_path=None):
                pass
            def save(self, path):
                pass
            def get_model_info(self):
                pass
        
        with pytest.raises(TypeError):
            IncompleteModel()
    
    def test_basemodel_requires_save_implementation(self):
        """Test that subclasses must implement save() method."""
        class IncompleteModel(BaseASRModel):
            def load(self, checkpoint_path=None):
                pass
            def transcribe(self, audio_paths, **kwargs):
                pass
            def get_model_info(self):
                pass
        
        with pytest.raises(TypeError):
            IncompleteModel()
    
    def test_basemodel_requires_get_model_info_implementation(self):
        """Test that subclasses must implement get_model_info() method."""
        class IncompleteModel(BaseASRModel):
            def load(self, checkpoint_path=None):
                pass
            def transcribe(self, audio_paths, **kwargs):
                pass
            def save(self, path):
                pass
        
        with pytest.raises(TypeError):
            IncompleteModel()
    
    def test_complete_implementation_instantiates(self):
        """Test that complete BaseASRModel implementation can be instantiated."""
        class CompleteModel(BaseASRModel):
            def load(self, checkpoint_path=None):
                pass
            def transcribe(self, audio_paths, **kwargs):
                return "test"
            def save(self, path):
                pass
            def get_model_info(self):
                return {}
        
        # Should not raise
        model = CompleteModel()
        assert isinstance(model, BaseASRModel)


class TestWhisperModel:
    """Tests for WhisperModel implementation."""
    
    def test_whisper_model_inherits_from_basemodel(self):
        """Test that WhisperModel properly inherits from BaseASRModel."""
        model = WhisperModel(variant='tiny', device='cpu')
        assert isinstance(model, BaseASRModel)
    
    def test_whisper_model_initialization(self):
        """Test WhisperModel initialization with different configurations."""
        # Test with explicit device
        model = WhisperModel(variant='small', device='cpu')
        assert model.variant == 'small'
        assert model.device == 'cpu'
        assert model.model is None
        assert model.processor is None
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_device_auto_detection_cuda(self, mock_mps, mock_cuda):
        """Test automatic device detection prefers CUDA when available."""
        mock_cuda.return_value = True
        mock_mps.return_value = False
        
        model = WhisperModel(variant='tiny')
        assert model.device == 'cuda'
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_device_auto_detection_mps(self, mock_mps, mock_cuda):
        """Test automatic device detection uses MPS on Apple Silicon."""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        
        model = WhisperModel(variant='tiny')
        assert model.device == 'mps'
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_device_auto_detection_cpu(self, mock_mps, mock_cuda):
        """Test automatic device detection falls back to CPU."""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        
        model = WhisperModel(variant='tiny')
        assert model.device == 'cpu'
    
    @patch('src.models.whisper_model.WhisperForConditionalGeneration')
    @patch('src.models.whisper_model.WhisperProcessor')
    def test_load_pretrained_model(self, mock_processor_class, mock_model_class):
        """Test loading pretrained Whisper model from HuggingFace."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock model.to() to return self
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = [torch.zeros(100)]
        
        # Create and load model
        model = WhisperModel(variant='tiny', device='cpu')
        model.load('openai/whisper-tiny')
        
        # Verify model and processor loaded
        assert model.model is mock_model
        assert model.processor is mock_processor
        
        # Verify loading calls
        mock_processor_class.from_pretrained.assert_called_once_with(
            'openai/whisper-tiny',
            language='english',
            task='transcribe'
        )
        mock_model_class.from_pretrained.assert_called_once_with('openai/whisper-tiny')
        
        # Verify model moved to device
        mock_model.to.assert_called_once_with('cpu')
        
        # Verify configuration
        assert mock_model.config.forced_decoder_ids is None
        assert mock_model.generation_config.language == 'english'
        assert mock_model.generation_config.task == 'transcribe'
    
    @patch('src.models.whisper_model.WhisperForConditionalGeneration')
    @patch('src.models.whisper_model.WhisperProcessor')
    def test_load_with_default_checkpoint(self, mock_processor_class, mock_model_class):
        """Test loading model with default checkpoint path (None)."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = [torch.zeros(100)]
        
        model = WhisperModel(variant='small', device='cpu')
        model.load()  # Should default to openai/whisper-small
        
        mock_model_class.from_pretrained.assert_called_once_with('openai/whisper-small')
    
    @patch('src.models.whisper_model.WhisperForConditionalGeneration')
    @patch('src.models.whisper_model.WhisperProcessor')
    def test_load_checkpoint_failure(self, mock_processor_class, mock_model_class):
        """Test that load() raises RuntimeError on failure."""
        mock_model_class.from_pretrained.side_effect = Exception("Download failed")
        
        model = WhisperModel(variant='tiny', device='cpu')
        
        with pytest.raises(RuntimeError, match="Model loading failed"):
            model.load('openai/whisper-tiny')
    
    @patch('src.models.whisper_model.WhisperForConditionalGeneration')
    @patch('src.models.whisper_model.WhisperProcessor')
    @patch('librosa.load')
    def test_transcribe_single_audio(self, mock_librosa, mock_processor_class, mock_model_class):
        """Test transcribing a single audio file."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = [torch.zeros(100)]
        
        # Mock audio loading
        mock_audio = np.random.randn(16000).astype(np.float32)
        mock_librosa.return_value = (mock_audio, 16000)
        
        # Mock processor output
        mock_inputs = {
            'input_features': torch.randn(1, 80, 3000)
        }
        mock_processor.return_value = mock_inputs
        
        # Mock generation
        mock_generated_ids = torch.tensor([[1, 2, 3, 4]])
        mock_model.generate.return_value = mock_generated_ids
        
        # Mock decoding
        mock_processor.batch_decode.return_value = ["hello world"]
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file:
            # Create and load model
            model = WhisperModel(variant='tiny', device='cpu')
            model.load('openai/whisper-tiny')
            
            # Transcribe
            result = model.transcribe(tmp_file.name)
            
            # Verify result
            assert result == "hello world"
            assert isinstance(result, str)
            
            # Verify librosa was called
            mock_librosa.assert_called_once()
            
            # Verify processor was called
            mock_processor.assert_called_once()
            
            # Verify generate was called
            mock_model.generate.assert_called_once()
    
    @patch('src.models.whisper_model.WhisperForConditionalGeneration')
    @patch('src.models.whisper_model.WhisperProcessor')
    @patch('librosa.load')
    def test_transcribe_batch_audio(self, mock_librosa, mock_processor_class, mock_model_class):
        """Test transcribing multiple audio files in batch."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = [torch.zeros(100)]
        
        # Mock audio loading
        mock_audio = np.random.randn(16000).astype(np.float32)
        mock_librosa.return_value = (mock_audio, 16000)
        
        # Mock processor output
        mock_inputs = {
            'input_features': torch.randn(2, 80, 3000)
        }
        mock_processor.return_value = mock_inputs
        
        # Mock generation
        mock_generated_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mock_model.generate.return_value = mock_generated_ids
        
        # Mock decoding
        mock_processor.batch_decode.return_value = ["hello world", "goodbye world"]
        
        # Create temporary audio files
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file1, \
             tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file2:
            
            # Create and load model
            model = WhisperModel(variant='tiny', device='cpu')
            model.load('openai/whisper-tiny')
            
            # Transcribe batch
            result = model.transcribe([tmp_file1.name, tmp_file2.name])
            
            # Verify result
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0] == "hello world"
            assert result[1] == "goodbye world"
            
            # Verify librosa was called twice
            assert mock_librosa.call_count == 2
    
    def test_transcribe_without_loading_model(self):
        """Test that transcribe() raises RuntimeError if model not loaded."""
        model = WhisperModel(variant='tiny', device='cpu')
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.transcribe("dummy_audio.wav")
    
    @patch('src.models.whisper_model.WhisperForConditionalGeneration')
    @patch('src.models.whisper_model.WhisperProcessor')
    def test_transcribe_nonexistent_file(self, mock_processor_class, mock_model_class):
        """Test that transcribe() raises FileNotFoundError for missing files."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = [torch.zeros(100)]
        
        model = WhisperModel(variant='tiny', device='cpu')
        model.load('openai/whisper-tiny')
        
        with pytest.raises(FileNotFoundError):
            model.transcribe("nonexistent_file.wav")
    
    @patch('src.models.whisper_model.WhisperForConditionalGeneration')
    @patch('src.models.whisper_model.WhisperProcessor')
    def test_save_model(self, mock_processor_class, mock_model_class, tmp_path):
        """Test saving model checkpoint to disk."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = [torch.zeros(100)]
        
        model = WhisperModel(variant='tiny', device='cpu')
        model.load('openai/whisper-tiny')
        
        save_path = tmp_path / "checkpoint"
        model.save(save_path)
        
        # Verify save methods were called
        mock_model.save_pretrained.assert_called_once_with(str(save_path))
        mock_processor.save_pretrained.assert_called_once_with(str(save_path))
        
        # Verify directory was created
        assert save_path.exists()
    
    def test_save_without_loading_model(self, tmp_path):
        """Test that save() raises RuntimeError if model not loaded."""
        model = WhisperModel(variant='tiny', device='cpu')
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.save(tmp_path / "checkpoint")
    
    def test_get_model_info_before_loading(self):
        """Test get_model_info() returns basic info before model is loaded."""
        model = WhisperModel(variant='medium', device='cpu')
        info = model.get_model_info()
        
        assert info['name'] == 'whisper'
        assert info['variant'] == 'medium'
        assert info['device'] == 'cpu'
        assert info['sample_rate'] == 16000
        assert info['language'] == 'english'
        assert info['task'] == 'transcribe'
        assert info['model_loaded'] is False
        assert 'parameters' not in info
        assert 'trainable_parameters' not in info
    
    @patch('src.models.whisper_model.WhisperForConditionalGeneration')
    @patch('src.models.whisper_model.WhisperProcessor')
    def test_get_model_info_after_loading(self, mock_processor_class, mock_model_class):
        """Test get_model_info() returns detailed info after model is loaded."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        # Mock parameters - must return new iterator each time with requires_grad
        param1 = torch.zeros(100)
        param1.requires_grad = True
        param2 = torch.zeros(200)
        param2.requires_grad = True
        mock_params = [param1, param2]
        mock_model.parameters.side_effect = lambda: iter(mock_params)
        
        model = WhisperModel(variant='tiny', device='cpu')
        model.load('openai/whisper-tiny')
        
        info = model.get_model_info()
        
        assert info['name'] == 'whisper'
        assert info['variant'] == 'tiny'
        assert info['model_loaded'] is True
        assert info['parameters'] == 300  # 100 + 200
        assert info['trainable_parameters'] == 300


class TestPrepareModelForFinetuning:
    """Tests for prepare_model_for_finetuning function."""
    
    def test_prepare_model_basic_configuration(self):
        """Test that model is configured with correct settings."""
        mock_model = MagicMock()
        
        result = prepare_model_for_finetuning(
            mock_model,
            freeze_encoder=False,
            dropout=0.1,
            language='english',
            task='transcribe'
        )
        
        # Verify configuration
        assert mock_model.config.forced_decoder_ids is None
        assert mock_model.config.suppress_tokens == []
        assert mock_model.generation_config.language == 'english'
        assert mock_model.generation_config.task == 'transcribe'
        assert mock_model.config.dropout == 0.1
        assert mock_model.config.attention_dropout == 0.1
        assert mock_model.config.activation_dropout == 0.1
        
        assert result is mock_model
    
    def test_prepare_model_freeze_encoder(self):
        """Test that encoder can be frozen while decoder remains trainable."""
        mock_model = MagicMock()
        
        # Mock encoder parameters
        mock_param1 = MagicMock()
        mock_param2 = MagicMock()
        mock_param1.requires_grad = True
        mock_param2.requires_grad = True
        mock_model.model.encoder.parameters.return_value = [mock_param1, mock_param2]
        
        # Mock parameters for counting
        mock_trainable = MagicMock()
        mock_trainable.numel.return_value = 1000
        mock_trainable.requires_grad = True
        
        mock_frozen = MagicMock()
        mock_frozen.numel.return_value = 2000
        mock_frozen.requires_grad = False
        
        mock_model.parameters.return_value = [mock_trainable, mock_frozen]
        
        result = prepare_model_for_finetuning(
            mock_model,
            freeze_encoder=True,
            dropout=0.05
        )
        
        # Verify encoder parameters were frozen
        assert mock_param1.requires_grad is False
        assert mock_param2.requires_grad is False
    
    def test_prepare_model_different_languages(self):
        """Test preparing model for different language/task combinations."""
        mock_model = MagicMock()
        
        prepare_model_for_finetuning(
            mock_model,
            language='spanish',
            task='translate'
        )
        
        assert mock_model.generation_config.language == 'spanish'
        assert mock_model.generation_config.task == 'translate'


class TestModelFactory:
    """Tests for ModelFactory."""
    
    def test_factory_creates_whisper_model(self):
        """Test that factory correctly creates WhisperModel."""
        config = {
            'model': {
                'name': 'whisper',
                'variant': 'small',
                'pretrained': 'openai/whisper-small'
            }
        }
        
        model = ModelFactory.create_model(config)
        
        assert isinstance(model, WhisperModel)
        assert isinstance(model, BaseASRModel)
        assert model.variant == 'small'
    
    def test_factory_with_explicit_device(self):
        """Test factory creates model with explicit device."""
        config = {
            'model': {
                'name': 'whisper',
                'variant': 'small',
                'device': 'cpu'
            }
        }
        
        model = ModelFactory.create_model(config)
        
        assert model.device == 'cpu'
    
    def test_factory_case_insensitive_model_name(self):
        """Test that model name matching is case-insensitive."""
        config = {
            'model': {
                'name': 'WHISPER',
                'variant': 'tiny'
            }
        }
        
        model = ModelFactory.create_model(config)
        assert isinstance(model, WhisperModel)
    
    def test_factory_unknown_model_raises_error(self):
        """Test that factory raises ValueError for unknown model types."""
        config = {
            'model': {
                'name': 'unknown_model',
                'variant': 'large'
            }
        }
        
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create_model(config)
    
    def test_factory_missing_model_config(self):
        """Test that factory raises KeyError if model config missing."""
        config = {}
        
        with pytest.raises(KeyError):
            ModelFactory.create_model(config)
    
    def test_factory_missing_model_name(self):
        """Test that factory raises KeyError if model name missing."""
        config = {
            'model': {
                'variant': 'medium'
            }
        }
        
        with pytest.raises(KeyError):
            ModelFactory.create_model(config)
    
    def test_factory_registry_contains_whisper(self):
        """Test that model registry contains Whisper."""
        assert 'whisper' in ModelFactory._MODEL_REGISTRY
        assert ModelFactory._MODEL_REGISTRY['whisper'] is WhisperModel


class TestModelIntegration:
    """Integration tests for end-to-end model functionality."""
    
    @patch('src.models.whisper_model.WhisperForConditionalGeneration')
    @patch('src.models.whisper_model.WhisperProcessor')
    @patch('librosa.load')
    def test_full_pipeline_with_factory(
        self, mock_librosa, mock_processor_class, mock_model_class, tmp_path
    ):
        """Test complete pipeline: factory creation -> load -> transcribe -> save."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = [torch.zeros(100)]
        
        # Mock audio and transcription
        mock_audio = np.random.randn(16000).astype(np.float32)
        mock_librosa.return_value = (mock_audio, 16000)
        mock_inputs = {'input_features': torch.randn(1, 80, 3000)}
        mock_processor.return_value = mock_inputs
        mock_generated_ids = torch.tensor([[1, 2, 3]])
        mock_model.generate.return_value = mock_generated_ids
        mock_processor.batch_decode.return_value = ["test transcription"]
        
        # Create config
        config = {
            'model': {
                'name': 'whisper',
                'variant': 'tiny',
                'pretrained': 'openai/whisper-tiny',
                'device': 'cpu'
            }
        }
        
        # Create model via factory
        model = ModelFactory.create_model(config)
        assert isinstance(model, WhisperModel)
        
        # Load model
        model.load(config['model']['pretrained'])
        assert model.model is not None
        
        # Get model info
        info = model.get_model_info()
        assert info['model_loaded'] is True
        
        # Transcribe dummy audio
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file:
            result = model.transcribe(tmp_file.name)
            assert result == "test transcription"
        
        # Save model
        save_path = tmp_path / "test_checkpoint"
        model.save(save_path)
        assert save_path.exists()
        
        # Verify all operations completed
        mock_model_class.from_pretrained.assert_called()
        mock_model.generate.assert_called()
        mock_model.save_pretrained.assert_called()
