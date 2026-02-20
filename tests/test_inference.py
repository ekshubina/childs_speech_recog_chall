"""
Tests for inference pipeline.

This module tests end-to-end prediction pipeline with mock data,
JSONL output format, and batch processing functionality.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from src.inference.predictor import Predictor


@pytest.fixture
def mock_whisper_model():
    """
    Mock WhisperForConditionalGeneration model.

    Returns a mock that simulates model.generate() behavior.
    """
    model = MagicMock()
    model.eval.return_value = None
    model.to.return_value = model
    model.parameters.return_value = [torch.randn(100) for _ in range(10)]

    # Mock generate to return token IDs
    def mock_generate(*args, **kwargs):
        batch_size = args[0].shape[0] if len(args) > 0 else 1
        # Return dummy token IDs (shape: [batch_size, sequence_length])
        return torch.randint(0, 1000, (batch_size, 10))

    model.generate.side_effect = mock_generate

    return model


@pytest.fixture
def mock_whisper_processor():
    """
    Mock WhisperProcessor.

    Returns a mock that simulates preprocessing and decoding.
    """
    processor = MagicMock()

    # Mock __call__ for preprocessing
    def mock_call(audio, sampling_rate, return_tensors, padding=True):
        batch_size = len(audio) if isinstance(audio, list) else 1
        # Return dummy input features
        return {"input_features": torch.randn(batch_size, 80, 3000)}

    processor.side_effect = mock_call

    # Mock batch_decode for decoding
    def mock_batch_decode(token_ids, skip_special_tokens=True):
        batch_size = token_ids.shape[0]
        return [f"transcription_{i}" for i in range(batch_size)]

    processor.batch_decode = Mock(side_effect=mock_batch_decode)

    return processor


@pytest.fixture
def sample_audio_files(tmp_path):
    """
    Create sample audio files for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Dictionary with audio directory and list of audio file paths
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    audio_files = []
    for i in range(5):
        # Create 1-second audio at 16kHz
        audio_data = np.random.randn(16000).astype(np.float32)
        audio_path = audio_dir / f"audio_{i:03d}.wav"
        sf.write(str(audio_path), audio_data, 16000)
        audio_files.append(audio_path)

    return {"audio_dir": audio_dir, "audio_files": audio_files}


@pytest.fixture
def sample_manifest(tmp_path, sample_audio_files):
    """
    Create sample JSONL manifest for testing.

    Args:
        tmp_path: Pytest temporary directory fixture
        sample_audio_files: Sample audio files fixture

    Returns:
        Path to manifest file
    """
    manifest_path = tmp_path / "test_manifest.jsonl"

    with open(manifest_path, "w") as f:
        for i, audio_file in enumerate(sample_audio_files["audio_files"]):
            entry = {
                "utterance_id": f"utt_{i:03d}",
                "audio_file": audio_file.name,
                "age_bucket": "5-6" if i % 2 == 0 else "7-8",
            }
            f.write(json.dumps(entry) + "\n")

    return manifest_path


@pytest.fixture
def mock_model_checkpoint(tmp_path):
    """
    Create a mock model checkpoint directory.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to mock checkpoint directory
    """
    checkpoint_dir = tmp_path / "model_checkpoint"
    checkpoint_dir.mkdir()

    # Create dummy files to simulate a checkpoint
    (checkpoint_dir / "config.json").write_text("{}")
    (checkpoint_dir / "pytorch_model.bin").write_text("dummy")

    return checkpoint_dir


class TestPredictorInitialization:
    """Tests for Predictor initialization."""

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predictor_init_valid_path(
        self, mock_processor_cls, mock_model_cls, mock_model_checkpoint, mock_whisper_model, mock_whisper_processor
    ):
        """Test Predictor initialization with valid model path."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu", batch_size=8)

        assert predictor.model_path == mock_model_checkpoint
        assert predictor.device == "cpu"
        assert predictor.batch_size == 8
        assert predictor.language == "english"
        assert predictor.task == "transcribe"

        # Verify model and processor were loaded
        mock_model_cls.from_pretrained.assert_called_once()
        mock_processor_cls.from_pretrained.assert_called_once()
        mock_whisper_model.to.assert_called_with("cpu")
        mock_whisper_model.eval.assert_called_once()

    def test_predictor_init_invalid_path(self):
        """Test that Predictor raises FileNotFoundError for invalid path."""
        with pytest.raises(FileNotFoundError):
            Predictor(model_path="nonexistent_model_path")

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    @patch("torch.cuda.is_available")
    def test_predictor_auto_device_detection_cuda(
        self,
        mock_cuda_available,
        mock_processor_cls,
        mock_model_cls,
        mock_model_checkpoint,
        mock_whisper_model,
        mock_whisper_processor,
    ):
        """Test automatic CUDA device detection."""
        mock_cuda_available.return_value = True
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device=None)

        assert predictor.device == "cuda"

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_predictor_auto_device_detection_mps(
        self,
        mock_mps_available,
        mock_cuda_available,
        mock_processor_cls,
        mock_model_cls,
        mock_model_checkpoint,
        mock_whisper_model,
        mock_whisper_processor,
    ):
        """Test automatic MPS device detection (Apple Silicon)."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device=None)

        assert predictor.device == "mps"

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_predictor_auto_device_detection_cpu(
        self,
        mock_mps_available,
        mock_cuda_available,
        mock_processor_cls,
        mock_model_cls,
        mock_model_checkpoint,
        mock_whisper_model,
        mock_whisper_processor,
    ):
        """Test fallback to CPU when no GPU available."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device=None)

        assert predictor.device == "cpu"


class TestPredictorSinglePrediction:
    """Tests for single file prediction."""

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predict_single_success(
        self,
        mock_processor_cls,
        mock_model_cls,
        mock_model_checkpoint,
        mock_whisper_model,
        mock_whisper_processor,
        sample_audio_files,
    ):
        """Test successful prediction on a single audio file."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu", batch_size=8)

        audio_file = sample_audio_files["audio_files"][0]
        result = predictor.predict_single(audio_file)

        assert isinstance(result, str)
        assert result == "transcription_0"

        # Verify model.generate was called
        mock_whisper_model.generate.assert_called()

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predict_single_file_not_found(
        self, mock_processor_cls, mock_model_cls, mock_model_checkpoint, mock_whisper_model, mock_whisper_processor
    ):
        """Test that FileNotFoundError is raised for non-existent file."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu")

        with pytest.raises(FileNotFoundError):
            predictor.predict_single("nonexistent_audio.wav")


class TestPredictorBatchPrediction:
    """Tests for batch prediction."""

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predict_batch_success(
        self,
        mock_processor_cls,
        mock_model_cls,
        mock_model_checkpoint,
        mock_whisper_model,
        mock_whisper_processor,
        sample_audio_files,
    ):
        """Test successful batch prediction."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu", batch_size=2)

        audio_files = sample_audio_files["audio_files"][:3]
        results = predictor.predict_batch(audio_files, show_progress=False)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
        assert results == ["transcription_0", "transcription_1", "transcription_0"]

        # Should be called 2 times (batch_size=2, 3 files → 2 batches)
        assert mock_whisper_model.generate.call_count == 2

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predict_batch_single_batch(
        self,
        mock_processor_cls,
        mock_model_cls,
        mock_model_checkpoint,
        mock_whisper_model,
        mock_whisper_processor,
        sample_audio_files,
    ):
        """Test batch prediction with all files fitting in one batch."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu", batch_size=10)  # Larger than number of files

        audio_files = sample_audio_files["audio_files"][:3]
        results = predictor.predict_batch(audio_files, show_progress=False)

        assert len(results) == 3

        # Should be called exactly once
        assert mock_whisper_model.generate.call_count == 1

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predict_batch_file_not_found(
        self,
        mock_processor_cls,
        mock_model_cls,
        mock_model_checkpoint,
        mock_whisper_model,
        mock_whisper_processor,
        sample_audio_files,
    ):
        """Test that FileNotFoundError is raised if any file doesn't exist."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu")

        audio_files = [sample_audio_files["audio_files"][0], Path("nonexistent.wav")]

        with pytest.raises(FileNotFoundError):
            predictor.predict_batch(audio_files, show_progress=False)

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predict_batch_empty_list(
        self, mock_processor_cls, mock_model_cls, mock_model_checkpoint, mock_whisper_model, mock_whisper_processor
    ):
        """Test batch prediction with empty list."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu")

        results = predictor.predict_batch([], show_progress=False)

        assert results == []
        assert mock_whisper_model.generate.call_count == 0


class TestPredictorFromManifest:
    """Tests for prediction from JSONL manifest."""

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predict_from_manifest_success(
        self,
        mock_processor_cls,
        mock_model_cls,
        mock_model_checkpoint,
        mock_whisper_model,
        mock_whisper_processor,
        sample_audio_files,
        sample_manifest,
        tmp_path,
    ):
        """Test successful prediction from manifest."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu", batch_size=2)

        output_path = tmp_path / "predictions.jsonl"
        results = predictor.predict_from_manifest(
            manifest_path=sample_manifest,
            audio_dirs=[sample_audio_files["audio_dir"]],
            output_path=output_path,
            show_progress=False,
        )

        # Verify results structure
        assert len(results) == 5
        assert all("utterance_id" in r for r in results)
        assert all("orthographic_text" in r for r in results)

        # Verify utterance IDs
        expected_ids = [f"utt_{i:03d}" for i in range(5)]
        actual_ids = [r["utterance_id"] for r in results]
        assert actual_ids == expected_ids

        # Verify output file was created
        assert output_path.exists()

        # Read and verify output file
        with open(output_path, "r") as f:
            output_data = [json.loads(line) for line in f]

        assert len(output_data) == 5
        assert output_data == results

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predict_from_manifest_no_output_file(
        self,
        mock_processor_cls,
        mock_model_cls,
        mock_model_checkpoint,
        mock_whisper_model,
        mock_whisper_processor,
        sample_audio_files,
        sample_manifest,
    ):
        """Test prediction from manifest without writing output file."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu")

        results = predictor.predict_from_manifest(
            manifest_path=sample_manifest,
            audio_dirs=[sample_audio_files["audio_dir"]],
            output_path=None,  # No output file
            show_progress=False,
        )

        assert len(results) == 5
        assert all("utterance_id" in r for r in results)
        assert all("orthographic_text" in r for r in results)

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predict_from_manifest_multiple_audio_dirs(
        self, mock_processor_cls, mock_model_cls, mock_model_checkpoint, mock_whisper_model, mock_whisper_processor, tmp_path
    ):
        """Test prediction with audio files spread across multiple directories."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        # Create audio in multiple directories
        audio_dir_0 = tmp_path / "audio_0"
        audio_dir_1 = tmp_path / "audio_1"
        audio_dir_0.mkdir()
        audio_dir_1.mkdir()

        # Create audio files
        audio_0 = audio_dir_0 / "audio_000.wav"
        audio_1 = audio_dir_1 / "audio_001.wav"

        for audio_path in [audio_0, audio_1]:
            audio_data = np.random.randn(16000).astype(np.float32)
            sf.write(str(audio_path), audio_data, 16000)

        # Create manifest
        manifest_path = tmp_path / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            f.write(json.dumps({"utterance_id": "utt_000", "audio_file": "audio_000.wav"}) + "\n")
            f.write(json.dumps({"utterance_id": "utt_001", "audio_file": "audio_001.wav"}) + "\n")

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu")

        results = predictor.predict_from_manifest(
            manifest_path=manifest_path, audio_dirs=[audio_dir_0, audio_dir_1], show_progress=False
        )

        assert len(results) == 2
        assert results[0]["utterance_id"] == "utt_000"
        assert results[1]["utterance_id"] == "utt_001"

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predict_from_manifest_file_not_found(
        self, mock_processor_cls, mock_model_cls, mock_model_checkpoint, mock_whisper_model, mock_whisper_processor, tmp_path
    ):
        """Test that FileNotFoundError is raised when audio file not found."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        # Create manifest with non-existent audio file
        manifest_path = tmp_path / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            f.write(json.dumps({"utterance_id": "utt_000", "audio_file": "nonexistent.wav"}) + "\n")

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu")

        with pytest.raises(FileNotFoundError):
            predictor.predict_from_manifest(manifest_path=manifest_path, audio_dirs=[tmp_path / "audio"], show_progress=False)

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_predict_from_manifest_invalid_manifest(
        self, mock_processor_cls, mock_model_cls, mock_model_checkpoint, mock_whisper_model, mock_whisper_processor
    ):
        """Test that FileNotFoundError is raised for invalid manifest path."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu")

        with pytest.raises(FileNotFoundError):
            predictor.predict_from_manifest(
                manifest_path="nonexistent_manifest.jsonl", audio_dirs=[Path("audio_dir")], show_progress=False
            )


class TestPredictorOutputFormat:
    """Tests for JSONL output format validation."""

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_jsonl_output_format_valid(
        self,
        mock_processor_cls,
        mock_model_cls,
        mock_model_checkpoint,
        mock_whisper_model,
        mock_whisper_processor,
        sample_audio_files,
        sample_manifest,
        tmp_path,
    ):
        """Test that output JSONL file has correct format."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu")

        output_path = tmp_path / "output.jsonl"
        predictor.predict_from_manifest(
            manifest_path=sample_manifest,
            audio_dirs=[sample_audio_files["audio_dir"]],
            output_path=output_path,
            show_progress=False,
        )

        # Read output file
        with open(output_path, "r") as f:
            lines = f.readlines()

        # Verify each line is valid JSON
        for line in lines:
            data = json.loads(line.strip())
            assert "utterance_id" in data
            assert "orthographic_text" in data
            assert isinstance(data["utterance_id"], str)
            assert isinstance(data["orthographic_text"], str)

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_jsonl_output_format_complete(
        self,
        mock_processor_cls,
        mock_model_cls,
        mock_model_checkpoint,
        mock_whisper_model,
        mock_whisper_processor,
        sample_audio_files,
        sample_manifest,
        tmp_path,
    ):
        """Test that all utterances are included in output."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(model_path=mock_model_checkpoint, device="cpu")

        output_path = tmp_path / "output.jsonl"
        predictor.predict_from_manifest(
            manifest_path=sample_manifest,
            audio_dirs=[sample_audio_files["audio_dir"]],
            output_path=output_path,
            show_progress=False,
        )

        # Load input manifest
        with open(sample_manifest, "r") as f:
            input_data = [json.loads(line) for line in f]

        # Load output
        with open(output_path, "r") as f:
            output_data = [json.loads(line) for line in f]

        # Verify counts match
        assert len(output_data) == len(input_data)

        # Verify all utterance IDs present
        input_ids = {entry["utterance_id"] for entry in input_data}
        output_ids = {entry["utterance_id"] for entry in output_data}
        assert input_ids == output_ids


class TestPredictorModelInfo:
    """Tests for model information retrieval."""

    @patch("src.inference.predictor.WhisperForConditionalGeneration")
    @patch("src.inference.predictor.WhisperProcessor")
    def test_get_model_info(
        self, mock_processor_cls, mock_model_cls, mock_model_checkpoint, mock_whisper_model, mock_whisper_processor
    ):
        """Test get_model_info returns correct metadata."""
        mock_model_cls.from_pretrained.return_value = mock_whisper_model
        mock_processor_cls.from_pretrained.return_value = mock_whisper_processor

        predictor = Predictor(
            model_path=mock_model_checkpoint, device="cpu", batch_size=16, language="english", task="transcribe"
        )

        info = predictor.get_model_info()

        assert "model_path" in info
        assert "device" in info
        assert "batch_size" in info
        assert "language" in info
        assert "task" in info
        assert "parameters" in info

        assert info["device"] == "cpu"
        assert info["batch_size"] == 16
        assert info["language"] == "english"
        assert info["task"] == "transcribe"
        assert info["parameters"] > 0
