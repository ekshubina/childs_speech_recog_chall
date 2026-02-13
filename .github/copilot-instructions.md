## Project Overview
Modular ASR system for DrivenData competition. Fine-tunes models on 95K+ children's speech samples (ages 3-12+). Uses config-driven workflow, factory pattern for models, and competition-compliant WER evaluation.

## Architecture & Key Components

### Factory Pattern for Models
- Models implement `BaseASRModel` interface ([src/models/base_model.py](src/models/base_model.py))
- Instantiate via `ModelFactory.create_model(config)` ([src/models/model_factory.py](src/models/model_factory.py))
- Currently supports Whisper; registry designed for easy extension (Wav2Vec2, Hubert, etc.)
- Models are created unloaded - must call `model.load()` separately

### Data Pipeline - Multi-Directory Audio
- Audio files split across `data/audio_0/`, `data/audio_1/`, `data/audio_2/`
- `ChildSpeechDataset` searches all directories automatically ([src/data/dataset.py](src/data/dataset.py))
- JSONL manifests with `audio_path` field referencing files like `audio/{utterance_id}.flac`
- **Critical**: Use `create_train_val_split()` with `stratify_by='age_bucket'` for representative validation

### Text Normalization - Competition Compliance
- **MUST** use Whisper's `EnglishTextNormalizer` for WER computation ([src/utils/text_normalizer.py](src/utils/text_normalizer.py))
- Handles contractions, punctuation, numbers - matches competition scoring
- Apply normalization to both predictions and references before computing WER
- See `WERMetric` class in [src/training/metrics.py](src/training/metrics.py) for reference implementation

### Training Infrastructure
- Wraps HuggingFace `Seq2SeqTrainer` via `WhisperTrainer` ([src/training/trainer.py](src/training/trainer.py))
- Uses FP16 mixed precision + gradient accumulation (default: 3 steps) for memory efficiency
- Model preparation: `prepare_model_for_finetuning()` in [src/models/whisper_model.py](src/models/whisper_model.py)
- Data collation: `WhisperDataCollator` handles batching with proper padding ([src/data/dataset.py](src/data/dataset.py))

### Inference & Evaluation
```bash
# Generate predictions
python scripts/predict.py \
    --model-path checkpoints/whisper-small-finetuned \
    --input-jsonl data/test_manifest.jsonl \
    --output-jsonl predictions.jsonl

# Evaluate WER
python scripts/evaluate.py \
    --model-path checkpoints/whisper-small-finetuned \
    --val-manifest data/val_manifest.jsonl
```

### Testing
```bash
# Run all tests
pytest

# Specific test categories (see pytest.ini for markers)
pytest -m unit              # Fast unit tests only
pytest -m integration       # Integration tests
pytest -m "not slow"        # Skip slow tests
pytest tests/test_data.py   # Single test file
```

### Monitoring Training
```bash
# TensorBoard logs automatically saved during training
tensorboard --logdir checkpoints/baseline_whisper_small/runs
```

## Configuration System

All experiments controlled via YAML in `configs/`. Key sections:
```yaml
model:
  name: whisper           # Used by ModelFactory
  variant: small          # Model size (tiny/base/small/medium/large)
  pretrained: openai/whisper-small
  freeze_encoder: false   # Train both encoder/decoder
  gradient_checkpointing: true  # Memory optimization

data:
  train_manifest: data/train_word_transcripts.jsonl
  audio_dirs: [data/audio_0, data/audio_1, data/audio_2]
  val_ratio: 0.1
  stratify_by: age_bucket      # Critical for representative splits

training:
  batch_size: 12                # Per-device
  gradient_accumulation_steps: 3  # Effective batch = 36
  fp16: true                    # Mixed precision (requires CUDA)
  learning_rate: 1e-5
  warmup_steps: 500
```

## Project-Specific Patterns

### Code Style
- Black formatter with **127 character line limit** (see [pyproject.toml](pyproject.toml))
- isort with Black profile for import sorting
- Type hints encouraged but optional (mypy configured but not strict)

### Module Organization
- Entry points: `scripts/*.py` (train, predict, evaluate)
- Core logic: `src/` with clear separation: data/, models/, training/, inference/, utils/
- Tests mirror src/ structure in `tests/`
- Configs are version-controlled, checkpoints are not

### Error Handling Patterns
- Audio loading failures logged but allow training to continue (see [src/data/dataset.py](src/data/dataset.py) `__getitem__`)
- Config validation happens at load time via `load_config()` ([src/utils/config.py](src/utils/config.py))
- All scripts use `setup_logger()` for consistent logging ([src/utils/logging_utils.py](src/utils/logging_utils.py))

### Data Loading Patterns
- Dataset supports both manifest file path AND pre-loaded samples list
- Use pre-loaded samples pattern for train/val splits to avoid re-parsing JSONL
- Example:
  ```python
  train_samples, val_samples = create_train_val_split(manifest_path, val_ratio=0.1)
  train_dataset = ChildSpeechDataset(samples=train_samples, audio_dirs=[...], processor=processor)
  val_dataset = ChildSpeechDataset(samples=val_samples, audio_dirs=[...], processor=processor)
  ```

## External Dependencies & Integration

### Key Libraries
- `transformers>=4.35.0` - Whisper model, Seq2SeqTrainer, tokenizers
- `torch>=2.0.0` + `torchaudio>=2.0.0` - Training framework
- `librosa>=0.10.0` - Audio loading/preprocessing
- `jiwer>=3.0.0` - WER computation (use with Whisper normalizer)
- `openai-whisper>=20231117` - **Critical for EnglishTextNormalizer**
- `accelerate>=0.24.0` - Multi-GPU support, FP16 training

### Model Checkpoints
- Pretrained from HuggingFace Hub: `openai/whisper-{size}`
- Fine-tuned checkpoints saved in `checkpoints/{output_dir}/`
- Checkpoint structure compatible with `transformers.WhisperForConditionalGeneration`

### Competition Requirements
- Output format: JSONL with `utterance_id` and `orthographic_text` fields
- Text normalization must match competition scoring (use Whisper normalizer)
- No external data allowed except TalkBank corpus (see [docs/DrivenData Competition Rules (1).md](docs/DrivenData%20Competition%20Rules%20(1).md))

## Common Gotchas

1. **Virtual environment**: Always `source venv/bin/activate` before running scripts
2. **Multi-directory audio**: Don't assume audio files are in single directory - use `audio_dirs` list
3. **Age stratification**: Train/val splits MUST stratify by `age_bucket` for representative evaluation
4. **Text normalization**: Don't use custom normalization - breaks competition scoring alignment
5. **Model loading**: `ModelFactory.create_model()` returns uninitialized model - call `.load()` before use
6. **Debug mode**: Use `--debug` flag for quick smoke tests before full runs
7. **GPU memory**: Default config assumes 8GB VRAM - adjust batch_size + gradient_accumulation_steps if needed

## Documentation References
- System architecture: [docs/SystemDesign.md](docs/SystemDesign.md)
- Competition rules: [docs/Problem Description.md](docs/Problem%20Description.md)
- Pipeline planning: [docs/baseline-pipeline/](docs/baseline-pipeline/)
