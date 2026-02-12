"""
Tests for data pipeline components.

This module tests audio loading, dataset functionality, stratified splitting,
and multi-directory file lookup.
"""

import json
import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data.audio_processor import (
    load_audio,
    resample_audio,
    convert_to_mono,
    normalize_audio
)
from src.data.dataset import (
    ChildSpeechDataset,
    create_train_val_split,
    save_manifest,
    WhisperDataCollator
)


class TestAudioProcessor:
    """Tests for audio processing functions."""
    
    def test_load_audio_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_audio("nonexistent_file.flac")
    
    def test_load_audio_with_different_formats(self, tmp_path):
        """Test loading audio files in different formats (WAV, FLAC simulation)."""
        # Create a temporary WAV file
        audio_data = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio_data, 16000)
        
        # Load the audio
        loaded_audio, sr = load_audio(str(wav_path), sr=16000)
        
        # Verify
        assert sr == 16000
        assert len(loaded_audio) > 0
        assert isinstance(loaded_audio, np.ndarray)
        assert loaded_audio.ndim == 1  # Should be mono
    
    def test_load_audio_resampling(self, tmp_path):
        """Test that audio is properly resampled to target sample rate."""
        # Create audio at 8kHz
        audio_data = np.random.randn(8000).astype(np.float32)
        audio_path = tmp_path / "test_8k.wav"
        sf.write(str(audio_path), audio_data, 8000)
        
        # Load with target sr=16000
        loaded_audio, sr = load_audio(str(audio_path), sr=16000)
        
        # Should be resampled to ~16000 samples (approximately 2x original)
        assert sr == 16000
        assert len(loaded_audio) > 8000  # Should be longer after upsampling
    
    def test_resample_audio_no_change(self):
        """Test that audio is unchanged when sample rates match."""
        audio = np.random.randn(16000)
        resampled = resample_audio(audio, orig_sr=16000, target_sr=16000)
        
        # Should be identical
        np.testing.assert_array_equal(audio, resampled)
    
    def test_resample_audio_downsample(self):
        """Test downsampling audio from 48kHz to 16kHz."""
        audio_48k = np.random.randn(48000)
        audio_16k = resample_audio(audio_48k, orig_sr=48000, target_sr=16000)
        
        # Should be approximately 1/3 the length
        assert len(audio_16k) < len(audio_48k)
        assert 15000 < len(audio_16k) < 17000  # ~16000 samples
    
    def test_resample_audio_upsample(self):
        """Test upsampling audio from 8kHz to 16kHz."""
        audio_8k = np.random.randn(8000)
        audio_16k = resample_audio(audio_8k, orig_sr=8000, target_sr=16000)
        
        # Should be approximately 2x the length
        assert len(audio_16k) > len(audio_8k)
        assert 15000 < len(audio_16k) < 17000  # ~16000 samples
    
    def test_convert_to_mono_already_mono(self):
        """Test that mono audio passes through unchanged."""
        mono_audio = np.random.randn(16000)
        result = convert_to_mono(mono_audio)
        
        np.testing.assert_array_equal(mono_audio, result)
        assert result.ndim == 1
    
    def test_convert_to_mono_stereo_channels_first(self):
        """Test converting stereo audio with shape (2, samples)."""
        stereo_audio = np.random.randn(2, 16000)
        mono_audio = convert_to_mono(stereo_audio)
        
        assert mono_audio.ndim == 1
        assert len(mono_audio) == 16000
        # Should be average of two channels
        expected = np.mean(stereo_audio, axis=0)
        np.testing.assert_array_almost_equal(mono_audio, expected)
    
    def test_convert_to_mono_stereo_channels_last(self):
        """Test converting stereo audio with shape (samples, 2)."""
        stereo_audio = np.random.randn(16000, 2)
        mono_audio = convert_to_mono(stereo_audio)
        
        assert mono_audio.ndim == 1
        assert len(mono_audio) == 16000
        # Should be average of two channels
        expected = np.mean(stereo_audio, axis=1)
        np.testing.assert_array_almost_equal(mono_audio, expected)
    
    def test_convert_to_mono_invalid_dimensions(self):
        """Test that 3D audio raises ValueError."""
        audio_3d = np.random.randn(2, 2, 16000)
        
        with pytest.raises(ValueError, match="Unexpected audio array dimensions"):
            convert_to_mono(audio_3d)
    
    def test_normalize_audio_standard(self):
        """Test audio normalization to target level."""
        audio = np.array([0.1, -0.5, 0.3, -0.2])
        normalized = normalize_audio(audio, target_level=0.95)
        
        # Max absolute value should be close to target
        max_val = np.max(np.abs(normalized))
        assert abs(max_val - 0.95) < 0.001
    
    def test_normalize_audio_empty(self):
        """Test that empty audio is returned as-is."""
        audio = np.array([])
        normalized = normalize_audio(audio)
        
        assert len(normalized) == 0
    
    def test_normalize_audio_silence(self):
        """Test that silent audio (all zeros) is returned as-is."""
        audio = np.zeros(16000)
        normalized = normalize_audio(audio)
        
        np.testing.assert_array_equal(audio, normalized)
    
    def test_normalize_audio_custom_target(self):
        """Test normalization with custom target level."""
        audio = np.array([0.2, -0.4, 0.3])
        normalized = normalize_audio(audio, target_level=0.5)
        
        max_val = np.max(np.abs(normalized))
        assert abs(max_val - 0.5) < 0.001


class TestChildSpeechDataset:
    """Tests for ChildSpeechDataset class."""
    
    @pytest.fixture
    def sample_manifest(self, tmp_path):
        """Create a sample JSONL manifest for testing."""
        manifest_path = tmp_path / "test_manifest.jsonl"
        
        samples = [
            {
                "utterance_id": "U_test001",
                "audio_path": "audio/U_test001.flac",
                "orthographic_text": "hello world",
                "age_bucket": "8-11",
                "audio_duration_sec": 1.5
            },
            {
                "utterance_id": "U_test002",
                "audio_path": "audio/U_test002.flac",
                "orthographic_text": "test speech",
                "age_bucket": "12-15",
                "audio_duration_sec": 2.0
            },
            {
                "utterance_id": "U_test003",
                "audio_path": "audio/U_test003.flac",
                "orthographic_text": "another example",
                "age_bucket": "8-11",
                "audio_duration_sec": 1.8
            }
        ]
        
        with open(manifest_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        return manifest_path
    
    @pytest.fixture
    def audio_dirs(self, tmp_path):
        """Create test audio directories."""
        audio_0 = tmp_path / "audio_0"
        audio_1 = tmp_path / "audio_1"
        audio_2 = tmp_path / "audio_2"
        
        audio_0.mkdir()
        audio_1.mkdir()
        audio_2.mkdir()
        
        # Create dummy audio files in different directories
        audio_data = np.random.randn(16000).astype(np.float32)
        
        sf.write(str(audio_0 / "U_test001.flac"), audio_data, 16000)
        sf.write(str(audio_1 / "U_test002.flac"), audio_data, 16000)
        sf.write(str(audio_2 / "U_test003.flac"), audio_data, 16000)
        
        return [audio_0, audio_1, audio_2]
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock WhisperProcessor."""
        processor = Mock()
        processor.return_value.input_features = [np.random.randn(80, 3000)]
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.return_value.input_ids = [np.array([1, 2, 3, 4])]
        processor.tokenizer = tokenizer
        
        return processor
    
    def test_dataset_initialization_with_manifest(self, sample_manifest, audio_dirs, mock_processor):
        """Test dataset initialization from manifest file."""
        dataset = ChildSpeechDataset(
            manifest_path=sample_manifest,
            audio_dirs=audio_dirs,
            processor=mock_processor
        )
        
        assert len(dataset) == 3
        assert dataset.manifest_path is not None
    
    def test_dataset_initialization_with_samples(self, mock_processor):
        """Test dataset initialization with pre-loaded samples."""
        samples = [
            {"utterance_id": "U_test001", "audio_path": "audio/U_test001.flac", "orthographic_text": "test"},
            {"utterance_id": "U_test002", "audio_path": "audio/U_test002.flac", "orthographic_text": "test2"}
        ]
        
        dataset = ChildSpeechDataset(
            samples=samples,
            audio_dirs=[],
            processor=mock_processor
        )
        
        assert len(dataset) == 2
        assert dataset.manifest_path is None
    
    def test_dataset_initialization_no_input(self, mock_processor):
        """Test that ValueError is raised when neither manifest nor samples provided."""
        with pytest.raises(ValueError, match="Must provide either manifest_path or samples"):
            ChildSpeechDataset(processor=mock_processor)
    
    def test_dataset_manifest_not_found(self, mock_processor):
        """Test that FileNotFoundError is raised for non-existent manifest."""
        with pytest.raises(FileNotFoundError):
            ChildSpeechDataset(
                manifest_path="nonexistent_manifest.jsonl",
                audio_dirs=[],
                processor=mock_processor
            )
    
    def test_find_audio_file_success(self, sample_manifest, audio_dirs, mock_processor):
        """Test finding audio file in multi-directory setup."""
        dataset = ChildSpeechDataset(
            manifest_path=sample_manifest,
            audio_dirs=audio_dirs,
            processor=mock_processor
        )
        
        # File in audio_0
        found_path = dataset._find_audio_file("audio/U_test001.flac")
        assert found_path is not None
        assert found_path.name == "U_test001.flac"
        assert found_path.exists()
        
        # File in audio_1
        found_path = dataset._find_audio_file("audio/U_test002.flac")
        assert found_path is not None
        assert found_path.name == "U_test002.flac"
        
        # File in audio_2
        found_path = dataset._find_audio_file("audio/U_test003.flac")
        assert found_path is not None
        assert found_path.name == "U_test003.flac"
    
    def test_find_audio_file_not_found(self, sample_manifest, audio_dirs, mock_processor):
        """Test handling of missing audio file."""
        dataset = ChildSpeechDataset(
            manifest_path=sample_manifest,
            audio_dirs=audio_dirs,
            processor=mock_processor
        )
        
        found_path = dataset._find_audio_file("audio/U_nonexistent.flac")
        assert found_path is None
    
    def test_dataset_getitem(self, sample_manifest, audio_dirs, mock_processor):
        """Test retrieving an item from the dataset."""
        # Setup mock processor to return proper tensors
        import torch
        mock_processor.return_value.input_features = torch.randn(1, 80, 3000)
        mock_processor.tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3, 4]])
        
        dataset = ChildSpeechDataset(
            manifest_path=sample_manifest,
            audio_dirs=audio_dirs,
            processor=mock_processor
        )
        
        # Get first item
        item = dataset[0]
        
        # Verify structure
        assert 'input_features' in item
        assert 'labels' in item
        assert 'utterance_id' in item
        assert item['utterance_id'] == 'U_test001'
        assert 'age_bucket' in item
        assert item['age_bucket'] == '8-11'
    
    def test_dataset_getitem_missing_audio(self, sample_manifest, mock_processor):
        """Test that FileNotFoundError is raised when audio file is missing."""
        # Create dataset with no audio directories
        dataset = ChildSpeechDataset(
            manifest_path=sample_manifest,
            audio_dirs=[],
            processor=mock_processor
        )
        
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            _ = dataset[0]
    
    def test_dataset_length(self, sample_manifest, audio_dirs, mock_processor):
        """Test __len__ method."""
        dataset = ChildSpeechDataset(
            manifest_path=sample_manifest,
            audio_dirs=audio_dirs,
            processor=mock_processor
        )
        
        assert len(dataset) == 3


class TestDatasetSplitting:
    """Tests for dataset splitting functionality."""
    
    @pytest.fixture
    def large_manifest(self, tmp_path):
        """Create a larger manifest for split testing."""
        manifest_path = tmp_path / "large_manifest.jsonl"
        
        samples = []
        age_buckets = ["8-11", "12-15", "16-18"]
        
        # Create 90 samples (30 per age bucket)
        for i in range(90):
            age_bucket = age_buckets[i % 3]
            sample = {
                "utterance_id": f"U_test{i:03d}",
                "audio_path": f"audio/U_test{i:03d}.flac",
                "orthographic_text": f"test speech {i}",
                "age_bucket": age_bucket,
                "audio_duration_sec": 1.5
            }
            samples.append(sample)
        
        with open(manifest_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        return manifest_path
    
    def test_create_train_val_split_basic(self, large_manifest):
        """Test basic train/val split functionality."""
        train_samples, val_samples = create_train_val_split(
            large_manifest,
            val_ratio=0.1,
            stratify_by='age_bucket',
            random_seed=42
        )
        
        # Check sizes
        total_samples = len(train_samples) + len(val_samples)
        assert total_samples == 90
        assert len(val_samples) == pytest.approx(9, abs=1)  # ~10% of 90
        assert len(train_samples) == pytest.approx(81, abs=1)
    
    def test_create_train_val_split_stratification(self, large_manifest):
        """Test that stratification maintains age bucket proportions."""
        train_samples, val_samples = create_train_val_split(
            large_manifest,
            val_ratio=0.2,
            stratify_by='age_bucket',
            random_seed=42
        )
        
        # Count age buckets in each split
        from collections import Counter
        
        train_buckets = Counter(s['age_bucket'] for s in train_samples)
        val_buckets = Counter(s['age_bucket'] for s in val_samples)
        
        # Each bucket should be roughly proportional
        for bucket in ["8-11", "12-15", "16-18"]:
            train_ratio = train_buckets[bucket] / len(train_samples)
            val_ratio = val_buckets[bucket] / len(val_samples)
            # Ratios should be similar (within 10%)
            assert abs(train_ratio - val_ratio) < 0.1
    
    def test_create_train_val_split_reproducibility(self, large_manifest):
        """Test that splits are reproducible with same random seed."""
        train1, val1 = create_train_val_split(large_manifest, random_seed=42)
        train2, val2 = create_train_val_split(large_manifest, random_seed=42)
        
        # Extract IDs for comparison
        train_ids1 = {s['utterance_id'] for s in train1}
        train_ids2 = {s['utterance_id'] for s in train2}
        
        assert train_ids1 == train_ids2
    
    def test_create_train_val_split_different_seed(self, large_manifest):
        """Test that different random seeds produce different splits."""
        train1, val1 = create_train_val_split(large_manifest, random_seed=42)
        train2, val2 = create_train_val_split(large_manifest, random_seed=99)
        
        # Extract IDs
        train_ids1 = {s['utterance_id'] for s in train1}
        train_ids2 = {s['utterance_id'] for s in train2}
        
        # Should be different
        assert train_ids1 != train_ids2
    
    def test_create_train_val_split_custom_ratio(self, large_manifest):
        """Test split with custom validation ratio."""
        train_samples, val_samples = create_train_val_split(
            large_manifest,
            val_ratio=0.3,
            random_seed=42
        )
        
        val_percentage = len(val_samples) / (len(train_samples) + len(val_samples))
        assert abs(val_percentage - 0.3) < 0.05  # Within 5%
    
    def test_create_train_val_split_no_stratification(self, large_manifest):
        """Test split without stratification."""
        train_samples, val_samples = create_train_val_split(
            large_manifest,
            val_ratio=0.2,
            stratify_by=None,
            random_seed=42
        )
        
        # Should still work
        assert len(train_samples) + len(val_samples) == 90
    
    def test_create_train_val_split_manifest_not_found(self):
        """Test that FileNotFoundError is raised for non-existent manifest."""
        with pytest.raises(FileNotFoundError):
            create_train_val_split("nonexistent.jsonl")
    
    def test_save_manifest(self, tmp_path):
        """Test saving samples to JSONL manifest."""
        samples = [
            {"utterance_id": "U_001", "text": "hello"},
            {"utterance_id": "U_002", "text": "world"}
        ]
        
        output_path = tmp_path / "output_manifest.jsonl"
        save_manifest(samples, output_path)
        
        # Verify file was created
        assert output_path.exists()
        
        # Load and verify contents
        loaded_samples = []
        with open(output_path, 'r') as f:
            for line in f:
                loaded_samples.append(json.loads(line.strip()))
        
        assert len(loaded_samples) == 2
        assert loaded_samples[0]['utterance_id'] == 'U_001'
        assert loaded_samples[1]['utterance_id'] == 'U_002'


class TestWhisperDataCollator:
    """Tests for WhisperDataCollator."""
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock WhisperProcessor."""
        processor = Mock()
        return processor
    
    def test_collator_initialization(self, mock_processor):
        """Test data collator initialization."""
        collator = WhisperDataCollator(processor=mock_processor)
        
        assert collator.processor is mock_processor
        assert collator.padding == "longest"
        assert collator.max_length is None
    
    def test_collator_batch_processing(self, mock_processor):
        """Test collating a batch of features."""
        import torch
        
        collator = WhisperDataCollator(processor=mock_processor)
        
        # Create mock features
        features = [
            {
                'input_features': torch.randn(80, 3000),
                'labels': torch.tensor([1, 2, 3]),
                'utterance_id': 'U_001',
                'age_bucket': '8-11'
            },
            {
                'input_features': torch.randn(80, 3000),
                'labels': torch.tensor([1, 2, 3, 4, 5]),
                'utterance_id': 'U_002',
                'age_bucket': '12-15'
            }
        ]
        
        batch = collator(features)
        
        # Verify batch structure
        assert 'input_features' in batch
        assert 'labels' in batch
        assert batch['input_features'].shape[0] == 2  # Batch size
        assert batch['labels'].shape[0] == 2
        
        # Labels should be padded to same length
        assert batch['labels'].shape[1] == 5  # Max label length
        
        # Metadata should be included
        assert 'utterance_ids' in batch
        assert len(batch['utterance_ids']) == 2
        assert batch['utterance_ids'][0] == 'U_001'
    
    def test_collator_label_padding(self, mock_processor):
        """Test that labels are properly padded with -100."""
        import torch
        
        collator = WhisperDataCollator(processor=mock_processor)
        
        features = [
            {
                'input_features': torch.randn(80, 3000),
                'labels': torch.tensor([1, 2]),
            },
            {
                'input_features': torch.randn(80, 3000),
                'labels': torch.tensor([1, 2, 3, 4]),
            }
        ]
        
        batch = collator(features)
        
        # First sample's labels should be padded with -100
        assert batch['labels'][0, 0] == 1
        assert batch['labels'][0, 1] == 2
        assert batch['labels'][0, 2] == -100
        assert batch['labels'][0, 3] == -100
        
        # Second sample's labels should be unchanged
        assert batch['labels'][1, 0] == 1
        assert batch['labels'][1, 1] == 2
        assert batch['labels'][1, 2] == 3
        assert batch['labels'][1, 3] == 4
    
    def test_collator_single_sample(self, mock_processor):
        """Test collating a single sample."""
        import torch
        
        collator = WhisperDataCollator(processor=mock_processor)
        
        features = [
            {
                'input_features': torch.randn(80, 3000),
                'labels': torch.tensor([1, 2, 3]),
                'utterance_id': 'U_001'
            }
        ]
        
        batch = collator(features)
        
        assert batch['input_features'].shape[0] == 1
        assert batch['labels'].shape[0] == 1
        assert len(batch['utterance_ids']) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
