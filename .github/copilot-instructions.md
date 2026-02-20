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
- **All training runs on RunPod** — never run `scripts/train.py` locally; always use `scripts/pod_train.sh`

### RunPod Training Workflow
All model training and heavy inference runs on RunPod GPU pods. The local machine only orchestrates.

**Prerequisites** — `.runpod.env` in repo root (git-ignored) with:
```
RUNPOD_API_KEY=<key>
NETWORK_VOLUME_ID=<volume-id>   # persists checkpoints & data across pod restarts
POD_ID=                          # auto-filled on first run
SSH_KEY_PATH=~/.ssh/id_ed25519
GPU_TYPE=NVIDIA GeForce RTX 4090
```

**Start / resume training (one command):**
```bash
# First run — creates pod, writes POD_ID back to .runpod.env
./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml

# Smoke test (100 train / 20 val samples)
./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml --debug

# Ignore existing checkpoints, restart from epoch 0
./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml --force-restart

# Subsequent runs — resumes the stopped pod automatically
./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml
```

**How it works** ([scripts/pod_train.sh](scripts/pod_train.sh)):
1. Creates a new pod (or resumes existing) via `runpodctl`
2. Polls until SSH is ready (secure-cloud proxy or direct TCP)
3. Streams [scripts/remote_train.sh](scripts/remote_train.sh) over SSH, which sets up the environment and launches training in a `tmux` session
4. Tails `/workspace/logs/current.log` live — Ctrl+C disconnects the tail but training keeps running

**Reattach after disconnect:**
```bash
ssh -i ~/.ssh/id_ed25519 <user>@ssh.runpod.io 'tail -f /workspace/logs/current.log'
ssh -i ~/.ssh/id_ed25519 <user>@ssh.runpod.io 'tmux attach -t train'
```

**Stop pod (billing stops):**
```bash
runpodctl stop pod $POD_ID
```

### Inference & Evaluation
Run on RunPod or locally against saved checkpoints pulled from the network volume:
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
# Stream live log from RunPod (run locally)
ssh -i ~/.ssh/id_ed25519 <user>@ssh.runpod.io 'tail -f /workspace/logs/current.log'

# TensorBoard via RunPod port-forward (see scripts/pod_tensorboard.sh)
./scripts/pod_tensorboard.sh
# Then open http://localhost:6006

# Or locally against synced checkpoints
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

1. **Always use RunPod for training**: Never run `scripts/train.py` locally — GPU is required and costs are managed via `pod_train.sh`
2. **`.runpod.env` is required**: Copy from `.runpod.env.example` and fill in `RUNPOD_API_KEY` and `NETWORK_VOLUME_ID` before first use
3. **POD_ID auto-saved**: First run of `pod_train.sh` creates the pod and writes `POD_ID` back to `.runpod.env`; subsequent runs resume it
4. **Stop pods when done**: Run `runpodctl stop pod $POD_ID` or billing continues
5. **Virtual environment**: Always `source venv/bin/activate` before running local scripts
6. **Multi-directory audio**: Don't assume audio files are in single directory - use `audio_dirs` list
7. **Age stratification**: Train/val splits MUST stratify by `age_bucket` for representative evaluation
8. **Text normalization**: Don't use custom normalization - breaks competition scoring alignment
9. **Model loading**: `ModelFactory.create_model()` returns uninitialized model - call `.load()` before use
10. **Debug mode**: Use `--debug` flag for quick smoke tests before full runs
11. **GPU memory**: Default config assumes 8GB VRAM - adjust `batch_size` + `gradient_accumulation_steps` if needed

## Documentation References
- System architecture: [docs/SystemDesign.md](docs/SystemDesign.md)
- Competition rules: [docs/Problem Description.md](docs/Problem%20Description.md)
- Pipeline planning: [docs/baseline-pipeline/](docs/baseline-pipeline/)
- RunPod automation: [docs/runpod-training-automation/](docs/runpod-training-automation/)
- RunPod usage guide: [docs/runpod-training-automation/runpod-usage-guide.md](docs/runpod-training-automation/runpod-usage-guide.md)
