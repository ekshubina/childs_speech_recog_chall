# Baseline Whisper-Medium Pipeline Scaffolding Implementation Plan

## Overview
Build the foundational infrastructure to enable Whisper-medium fine-tuning on children's speech data (95K+ samples). This scaffolding creates a complete, trainable baseline system with data processing, model abstraction, training loop, and inference capabilities—all configurable, testable, and aligned with DrivenData's code execution submission format. The goal is to establish a working end-to-end pipeline that can be iteratively improved rather than achieving optimal performance immediately.

## Goals
1. Create a modular codebase that supports easy experimentation with different models and configurations
2. Implement a data pipeline that handles multi-directory audio files and normalizes to 16kHz mono format
3. Build training infrastructure with WER evaluation using Whisper's text normalizer
4. Enable baseline model fine-tuning and inference on children's speech data
5. Establish code quality standards with testing and documentation

## Non-Goals
1. Achieving optimal WER performance (this is baseline/scaffolding phase)
2. Implementing data augmentation strategies (deferred to Phase 2)
3. Building ensemble models or advanced optimization techniques
4. Creating production-ready Docker containers (will be done in submission phase)
5. Implementing Wav2Vec or alternative model architectures

## Implementation Steps

**Note**: All unimplemented functions and methods must explicitly raise `NotImplementedError` with descriptive messages. This prevents silent failures and clearly identifies incomplete implementations during development and testing.

### Phase 1: Project Foundation
1. Create directory structure following the system design: [configs/](configs/), [src/data/](src/data/), [src/models/](src/models/), [src/training/](src/training/), [src/utils/](src/utils/), [scripts/](scripts/), [tests/](tests/), [notebooks/](notebooks/)
2. Initialize Python package with `__init__.py` files in all [src/](src/) subdirectories
3. Create [requirements.txt](requirements.txt) with core dependencies: `transformers>=4.35.0`, `torch>=2.0.0`, `torchaudio>=2.0.0`, `librosa>=0.10.0`, `jiwer>=3.0.0`, `pyyaml>=6.0`, `datasets>=2.14.0`, `accelerate>=0.24.0`, `numpy`, `pandas`, `soundfile`

### Phase 2: Configuration System
1. Implement [src/utils/config.py](src/utils/config.py) with `load_config()` function for YAML parsing and validation
2. Create [configs/baseline_whisper_medium.yaml](configs/baseline_whisper_medium.yaml) with model specification (`openai/whisper-medium`), data paths, training hyperparameters (batch_size=8, lr=1e-5, epochs=10), and inference settings
3. Add [src/utils/logging_utils.py](src/utils/logging_utils.py) for consistent logging across modules

### Phase 3: Data Processing Pipeline
1. Build [src/data/audio_processor.py](src/data/audio_processor.py) with functions for FLAC loading, resampling to 16kHz, converting to mono, and audio normalization
2. Implement [src/data/dataset.py](src/data/dataset.py) with `ChildSpeechDataset` class extending `torch.utils.data.Dataset` that loads JSONL manifest, searches audio files across multiple directories, preprocesses audio, and returns model-ready inputs
3. Add validation split logic with stratification by `age_bucket` to ensure representative validation set
4. Create [src/utils/text_normalizer.py](src/utils/text_normalizer.py) wrapping Whisper's `EnglishTextNormalizer` for consistent text preprocessing

### Phase 4: Model Abstraction Layer
1. Define [src/models/base_model.py](src/models/base_model.py) with `BaseASRModel` abstract class specifying interface methods: `load()`, `transcribe()`, `save()`, `get_model_info()`
2. Implement [src/models/whisper_model.py](src/models/whisper_model.py) with `WhisperModel` class wrapping `WhisperForConditionalGeneration` and `WhisperProcessor` from Hugging Face, supporting both pretrained loading and checkpoint resumption
3. Create [src/models/model_factory.py](src/models/model_factory.py) with factory pattern for model instantiation based on configuration
4. Configure model for English transcription with `forced_decoder_ids=None` and `language="en"`

### Phase 5: Training Infrastructure
1. Implement [src/training/metrics.py](src/training/metrics.py) with `WERMetric` class that computes Word Error Rate using `jiwer` library and applies Whisper's text normalizer to both predictions and references
2. Build [src/training/trainer.py](src/training/trainer.py) wrapping Hugging Face's `Seq2SeqTrainer` with custom compute_metrics, data collator for Whisper, and checkpoint management
3. Create [scripts/train.py](scripts/train.py) as entry point: parse config, initialize model, create datasets with 90/10 stratified split, setup trainer, and execute training loop with validation
4. Add checkpointing strategy: save every 1000 steps and keep best 3 checkpoints based on validation WER

### Phase 6: Inference and Evaluation
1. Build [scripts/predict.py](scripts/predict.py) for batch inference: load trained model, process audio files, generate transcriptions, and output JSONL format matching submission requirements
2. Implement [scripts/evaluate.py](scripts/evaluate.py) to compute WER on validation set, analyze errors by age group, and generate detailed metrics report
3. Create [scripts/prepare_submission.py](scripts/prepare_submission.py) to validate JSONL format and prepare submission file structure
4. Add [notebooks/eda.ipynb](notebooks/eda.ipynb) for exploratory data analysis: audio duration distribution, age group statistics, transcription length analysis

### Phase 7: Testing and Documentation
1. Write unit tests in [tests/test_data.py](tests/test_data.py) for audio loading, preprocessing, and dataset functionality
2. Create [tests/test_models.py](tests/test_models.py) for model interface compliance and basic inference
3. Add [tests/test_inference.py](tests/test_inference.py) for end-to-end prediction pipeline
4. Update [README.md](README.md) with setup instructions, usage examples, and project structure overview

## Success Criteria
1. **Trainable baseline**: Successfully fine-tune whisper-medium on training data without errors
2. **WER measurement**: Compute validation WER with Whisper normalizer matching competition scoring
3. **Valid output format**: Generate JSONL predictions matching submission format exactly
4. **Reproducible**: Training results reproducible with fixed random seed
5. **Modular code**: Easy to swap models by changing config file only
6. **Test coverage**: Core data and model functionality covered by unit tests

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory issues during training (Whisper-medium requires ~16GB VRAM) | High | Use FP16 mixed precision, gradient accumulation (4 steps), batch size 8; enable gradient checkpointing if needed |
| Audio files not found across multiple directories | Medium | Implement robust path search in dataset with clear error messages; validate all paths exist before training |
| WER computation doesn't match competition scoring | High | Use exact same normalizer (Whisper's EnglishTextNormalizer); validate with sample predictions vs expected WER |
| Long training time (95K samples) | Medium | Start with subset (10K samples) to validate pipeline; use accelerate for distributed training if available |
| Inconsistent audio formats in training data | Medium | Robust audio loading with librosa; explicit resampling to 16kHz and mono conversion; log warnings for issues |
| Configuration errors causing silent failures | Low | Add config validation in load_config(); use type hints and runtime checks; comprehensive logging |
| Package dependency conflicts | Low | Pin specific versions in requirements.txt; test in clean virtual environment; document Python version requirement (3.8+) |
