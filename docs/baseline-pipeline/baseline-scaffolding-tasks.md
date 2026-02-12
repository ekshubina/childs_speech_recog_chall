# Baseline Pipeline Scaffolding Task Checklist

## Summary
- **Total Tasks**: 28 implementation tasks + 6 testing tasks + 2 documentation tasks = 36 tasks
- **Dependencies**: None (starting from scratch)
- **Estimated Time**: 3-5 days for full scaffolding implementation

---

## Project Foundation

- [x] **Create directory structure** - Create [configs/](configs/), [src/data/](src/data/), [src/models/](src/models/), [src/training/](src/training/), [src/inference/](src/inference/), [src/utils/](src/utils/), [scripts/](scripts/), [tests/](tests/), [notebooks/](notebooks/) directories

- [x] **Initialize Python packages** - Add `__init__.py` files to [src/](src/), [src/data/](src/data/), [src/models/](src/models/), [src/training/](src/training/), [src/inference/](src/inference/), [src/utils/](src/utils/)

- [x] **Create requirements.txt** - List core dependencies with pinned versions: `transformers>=4.35.0`, `torch>=2.0.0`, `torchaudio>=2.0.0`, `librosa>=0.10.0`, `jiwer>=3.0.0`, `pyyaml>=6.0`, `datasets>=2.14.0`, `accelerate>=0.24.0`, `openai-whisper>=20231117`, `soundfile`, `numpy`, `pandas`, `pytest`

- [x] **Setup .gitignore** - Exclude checkpoints/, `__pycache__/`, `*.pyc`, `.DS_Store`, `.vscode/`, `*.ipynb_checkpoints`, data/ (large audio files)

---

## Configuration System

- [x] **Implement config loader** - Create [src/utils/config.py](src/utils/config.py) with `load_config(yaml_path)` function that parses YAML and validates required fields

- [x] **Create baseline config** - Build [configs/baseline_whisper_medium.yaml](configs/baseline_whisper_medium.yaml) with model settings (name: whisper, variant: medium, pretrained: openai/whisper-medium), data paths, training hyperparameters (batch_size: 8, learning_rate: 1e-5, num_epochs: 10, gradient_accumulation: 4, fp16: true), evaluation settings

- [x] **Add logging utilities** - Implement [src/utils/logging_utils.py](src/utils/logging_utils.py) with `setup_logger(name, log_file)` function using Python's logging module for consistent formatting

- [x] **Create text normalizer wrapper** - Build [src/utils/text_normalizer.py](src/utils/text_normalizer.py) wrapping `whisper.normalizers.EnglishTextNormalizer` with convenience methods

---

## Data Processing Pipeline

- [x] **Build audio processor** - Create [src/data/audio_processor.py](src/data/audio_processor.py) with functions: `load_audio(path, sr=16000)` using librosa, `resample_audio(audio, orig_sr, target_sr)`, `convert_to_mono(audio)`, `normalize_audio(audio)` for amplitude normalization

- [x] **Implement dataset class** - Build [src/data/dataset.py](src/data/dataset.py) with `ChildSpeechDataset(Dataset)` class: `__init__` loads JSONL manifest, `_find_audio_file` searches across audio_0/1/2 directories, `__getitem__` loads audio + processes with WhisperProcessor + returns input_features and labels, `__len__` returns dataset size

- [x] **Add dataset splitting** - Implement `create_train_val_split(manifest_path, val_ratio=0.1, stratify_by='age_bucket')` in [src/data/dataset.py](src/data/dataset.py) using sklearn's `train_test_split` with stratification

- [x] **Create data collator** - Implement `WhisperDataCollator` in [src/data/dataset.py](src/data/dataset.py) for dynamic padding and label preparation for Seq2Seq training

---

## Model Abstraction Layer

- [x] **Define base model interface** - Create [src/models/base_model.py](src/models/base_model.py) with `BaseASRModel(ABC)` defining abstract methods: `load(checkpoint_path)`, `transcribe(audio_paths)`, `save(path)`, `get_model_info()`

- [x] **Implement WhisperModel** - Build [src/models/whisper_model.py](src/models/whisper_model.py) with `WhisperModel(BaseASRModel)`: `__init__` sets variant and device, `load` initializes WhisperForConditionalGeneration and WhisperProcessor with forced_decoder_ids=None, `transcribe` performs batch inference, `save` saves model and processor, `get_model_info` returns metadata

- [x] **Create model factory** - Implement [src/models/model_factory.py](src/models/model_factory.py) with `ModelFactory.create_model(config)` that instantiates models based on config['model']['name']

- [x] **Add model initialization helper** - Create `prepare_model_for_finetuning(model)` in [src/models/whisper_model.py](src/models/whisper_model.py) to configure model settings (language, task, dropout)

---

## Training Infrastructure

- [x] **Build WER metric** - Create [src/training/metrics.py](src/training/metrics.py) with `WERMetric` class: `__init__` initializes EnglishTextNormalizer, `compute(predictions, references)` normalizes texts and computes WER using jiwer

- [x] **Create compute_metrics function** - Implement `compute_metrics(pred)` in [src/training/metrics.py](src/training/metrics.py) for Seq2SeqTrainer callback that decodes predictions and computes WER

- [x] **Build trainer wrapper** - Create [src/training/trainer.py](src/training/trainer.py) with `WhisperTrainer(Seq2SeqTrainer)` that configures training arguments, data collator, compute_metrics, and checkpoint strategy

- [x] **Implement training loop** - Build [scripts/train.py](scripts/train.py): parse arguments (--config, --resume), load config, create model via factory, load and split dataset, initialize trainer, execute training with `trainer.train()`, save final model

---

## Inference Pipeline

- [x] **Create predictor class** - Build [src/inference/predictor.py](src/inference/predictor.py) with `Predictor` class: `__init__` loads model and processor, `predict_batch(audio_paths)` performs batch inference, `predict_from_manifest(jsonl_path)` processes full dataset

- [x] **Implement prediction script** - Create [scripts/predict.py](scripts/predict.py): parse arguments (--model-path, --input-jsonl, --output-jsonl), initialize Predictor, generate predictions, write JSONL output with utterance_id and orthographic_text

- [x] **Build evaluation script** - Create [scripts/evaluate.py](scripts/evaluate.py): load validation set, run predictions, compute WER overall and by age_bucket, generate detailed metrics report, save results

- [x] **Add submission validator** - Implement [scripts/prepare_submission.py](scripts/prepare_submission.py): validate JSONL format, check required fields (utterance_id, orthographic_text), verify all test utterances covered, report any issues

---

## Testing

- [x] **Write data pipeline tests** - Create [tests/test_data.py](tests/test_data.py): test audio loading with different formats, test dataset with sample JSONL, test stratified splitting, test multi-directory file lookup

- [x] **Write model tests** - Create [tests/test_models.py](tests/test_models.py): test BaseASRModel interface compliance for WhisperModel, test model loading (pretrained and checkpoint), test basic transcription on dummy audio

- [x] **Write inference tests** - Create [tests/test_inference.py](tests/test_inference.py): test end-to-end prediction pipeline with mock data, test JSONL output format, test batch processing

- [x] **Write metrics tests** - Add tests in [tests/test_metrics.py](tests/test_metrics.py): test WER computation with known examples, test text normalization (contractions, punctuation), verify matches jiwer library

- [x] **Write integration test** - Create [tests/test_integration.py](tests/test_integration.py): test full pipeline from config loading → dataset creation → model init → training step → inference → evaluation

- [x] **Add CI/linting setup** - Create [.github/workflows/test.yml](.github/workflows/test.yml) for automated testing (optional) and setup pytest configuration

---

## Documentation

- [x] **Write comprehensive README** - Update [README.md](README.md) with project overview, installation instructions, usage examples for training/inference/evaluation, directory structure explanation, troubleshooting tips

- [x] **Create EDA notebook** - Build [notebooks/eda.ipynb](notebooks/eda.ipynb): load training data, analyze audio duration distribution, visualize age_bucket distribution, plot transcription length statistics, identify outliers, examine sample audio spectrograms

---

## Implementation Order

**Day 1: Foundation & Configuration**
1. Project foundation tasks (directory structure, requirements.txt, .gitignore)
2. Configuration system tasks (config loader, baseline config, logging)

**Day 2: Data Pipeline**
3. Data processing pipeline tasks (audio processor, dataset class, splitting, collator)
4. Begin testing with sample data

**Day 3: Model Layer**
5. Model abstraction layer tasks (base interface, WhisperModel, factory)
6. Test model loading and basic inference

**Day 4: Training & Inference**
7. Training infrastructure tasks (metrics, trainer wrapper, training script)
8. Inference pipeline tasks (predictor, scripts)

**Day 5: Testing & Documentation**
9. Testing tasks (unit tests, integration tests)
10. Documentation tasks (README, EDA notebook)
11. Validation run: train small model on subset to verify full pipeline

---

## Acceptance Criteria

Each task should be:
- **Testable**: Has clear success criteria that can be verified (e.g., "script runs without error", "test passes", "config loads successfully")
- **Atomic**: Can be completed independently without requiring multiple unrelated changes
- **Specific**: Focused on a single file or coherent functionality rather than vague multi-component work
- **Actionable**: Has clear implementation steps defined in context and plan documents

### Implementation Standards

**All unimplemented functions and methods must raise `NotImplementedError`:**
```python
def placeholder_function():
    """Brief description of what this will do."""
    raise NotImplementedError("placeholder_function needs implementation")

class BaseModel(ABC):
    @abstractmethod
    def method(self):
        """Abstract method description."""
        raise NotImplementedError("Subclasses must implement method()")
```

This ensures:
- Clear runtime errors when calling unfinished code
- Easy tracking of what still needs implementation
- No silent failures during development and testing
