# Baseline Pipeline Scaffolding Context & References

## Key Files

### Existing Code to Modify
| File | Purpose | Changes Needed |
|------|---------|----------------|
| N/A | No existing code | Starting from scratch |

### New Files to Create

#### Configuration & Utils
| File | Purpose |
|------|---------|
| [configs/baseline_whisper_medium.yaml](configs/baseline_whisper_medium.yaml) | Model config, data paths, training hyperparameters |
| [src/utils/config.py](src/utils/config.py) | YAML config loading and validation |
| [src/utils/text_normalizer.py](src/utils/text_normalizer.py) | Wrapper for Whisper's EnglishTextNormalizer |
| [src/utils/logging_utils.py](src/utils/logging_utils.py) | Logging utilities for training/inference |

#### Data Pipeline
| File | Purpose |
|------|---------|
| [src/data/audio_processor.py](src/data/audio_processor.py) | Audio loading, resampling to 16kHz mono, normalization |
| [src/data/dataset.py](src/data/dataset.py) | PyTorch Dataset for children's speech with multi-directory support |
| [src/data/__init__.py](src/data/__init__.py) | Package initialization |

#### Model Layer
| File | Purpose |
|------|---------|
| [src/models/base_model.py](src/models/base_model.py) | Abstract base class defining ASR model interface |
| [src/models/whisper_model.py](src/models/whisper_model.py) | Whisper implementation with fine-tuning support |
| [src/models/model_factory.py](src/models/model_factory.py) | Factory pattern for model instantiation |
| [src/models/__init__.py](src/models/__init__.py) | Package initialization |

#### Training Infrastructure
| File | Purpose |
|------|---------|
| [src/training/metrics.py](src/training/metrics.py) | WER computation with text normalization |
| [src/training/trainer.py](src/training/trainer.py) | Training loop wrapper using HF Seq2SeqTrainer |
| [src/training/__init__.py](src/training/__init__.py) | Package initialization |

#### Scripts
| File | Purpose |
|------|---------|
| [scripts/train.py](scripts/train.py) | Main training entry point |
| [scripts/predict.py](scripts/predict.py) | Batch inference for generating predictions |
| [scripts/evaluate.py](scripts/evaluate.py) | Validation WER computation and error analysis |
| [scripts/prepare_submission.py](scripts/prepare_submission.py) | Submission format validation |

#### Testing & Documentation
| File | Purpose |
|------|---------|
| [tests/test_data.py](tests/test_data.py) | Unit tests for data pipeline |
| [tests/test_models.py](tests/test_models.py) | Unit tests for model layer |
| [tests/test_inference.py](tests/test_inference.py) | Integration tests for inference |
| [notebooks/eda.ipynb](notebooks/eda.ipynb) | Exploratory data analysis |
| [requirements.txt](requirements.txt) | Python dependencies |
| [README.md](README.md) | Project documentation and setup instructions |

### Reference Implementations
| File | Relevance |
|------|-----------|
| [docs/SystemDesign.md](docs/SystemDesign.md) | Complete system architecture and component specifications |
| [docs/Problem Description.md](docs/Problem%20Description.md) | Competition requirements and evaluation criteria |
| [data/train_word_transcripts.jsonl](data/train_word_transcripts.jsonl) | Training data format and schema (95,572 samples) |

## Architecture Decisions

### Decision 1: Use Hugging Face Transformers for Whisper
- **Context**: Need to fine-tune Whisper-medium on children's speech data with efficient training infrastructure
- **Decision**: Use `transformers.WhisperForConditionalGeneration` and `Seq2SeqTrainer` from Hugging Face
- **Rationale**: 
  - Well-maintained library with extensive documentation
  - Built-in support for gradient accumulation, FP16, checkpointing
  - Easy integration with datasets and accelerate libraries
  - Consistent API for loading pretrained models and fine-tuning
- **Alternatives Considered**: 
  - OpenAI's official Whisper repo (lower-level, requires more custom code)
  - PyTorch Lightning (additional abstraction layer, more complex for this use case)

### Decision 2: Abstract Base Model Interface
- **Context**: System design calls for easy model switching between Whisper variants and potentially Wav2Vec
- **Decision**: Define `BaseASRModel` abstract class with standard interface methods
- **Rationale**:
  - Enables configuration-driven model selection via factory pattern
  - Facilitates testing with mock implementations
  - Future-proofs codebase for additional model architectures
  - Encapsulates model-specific loading/inference logic
- **Alternatives Considered**: 
  - Direct model usage without abstraction (less flexible, harder to test)
  - Protocol/structural typing (less explicit, no enforcement)

### Decision 3: Stratified Train/Val Split by Age Bucket
- **Context**: Training data spans age groups 3-4, 5-7, 8-11, 12+, and unknown; need representative validation set
- **Decision**: Use 90/10 stratified split ensuring all age groups proportionally represented in validation
- **Rationale**:
  - Children's speech characteristics vary significantly by age
  - Proportional representation prevents bias toward dominant age groups
  - Enables age-specific error analysis for debugging
  - Matches real-world test distribution better than random split
- **Alternatives Considered**:
  - Random 90/10 split (simpler but may miss age groups)
  - K-fold cross-validation (more robust but much slower for large dataset)

### Decision 4: Multi-Directory Audio File Lookup
- **Context**: Audio files split across `audio_0/`, `audio_1/`, `audio_2/` directories; JSONL only provides relative path
- **Decision**: Implement sequential search across base paths with caching for performance
- **Rationale**:
  - JSONL `audio_path` doesn't specify which directory contains the file
  - Searching all directories ensures all files found
  - Caching prevents repeated filesystem searches during training
  - Clear error messages when files not found
- **Alternatives Considered**:
  - Require manual path specification (error-prone, user unfriendly)
  - Preprocess to create single merged directory (requires data copying, storage overhead)

### Decision 5: Whisper's EnglishTextNormalizer for WER
- **Context**: Competition scoring uses Whisper's text normalizer; WER must match exactly for valid submissions
- **Decision**: Use `whisper.normalizers.EnglishTextNormalizer` for all WER computations and validation
- **Rationale**:
  - Competition explicitly states scoring methodology
  - Ensures local validation WER matches leaderboard WER
  - Handles punctuation, contractions, numbers consistently
  - Available directly from openai-whisper package
- **Alternatives Considered**:
  - Custom normalizer (high risk of mismatch with competition)
  - jiwer's built-in normalization (different rules, incompatible)

### Decision 6: Start Without Data Augmentation
- **Context**: System design includes augmentation strategies; need to balance complexity vs quick baseline
- **Decision**: Implement clean baseline first; defer augmentation to Phase 2 optimization
- **Rationale**:
  - Establishes performance floor without augmentation
  - Simplifies debugging (fewer variables)
  - Whisper-medium already trained on diverse data, may not need augmentation initially
  - Allows measuring augmentation impact independently
- **Alternatives Considered**:
  - Include basic augmentation from start (adds complexity, delays baseline)
  - Use SpecAugment only (still adds dependencies and tuning requirements)

### Decision 7: NotImplementedError for Placeholder Methods
- **Context**: Scaffolding phase creates many function stubs and class methods that need implementation
- **Decision**: All unimplemented functions and methods must explicitly raise `NotImplementedError` with descriptive message
- **Rationale**:
  - Makes it immediately clear when calling unfinished code
  - Prevents silent failures or unexpected behavior
  - Facilitates tracking implementation progress
  - Standard Python practice for abstract methods and stubs
  - Helps during testing to identify missing implementations
- **Alternatives Considered**:
  - Pass statements (silent failures, hard to debug)
  - TODO comments only (no runtime enforcement)
  - Return None (ambiguous, could be valid return value)

## Dependencies

### Internal Dependencies
- **EnglishTextNormalizer**: From `openai-whisper` package for text normalization matching competition
- **WhisperProcessor**: Combines feature extractor and tokenizer for consistent preprocessing
- **WhisperForConditionalGeneration**: Core model for fine-tuning and inference
- **Seq2SeqTrainer**: Training loop with built-in gradient accumulation and checkpointing

### External Dependencies
- **transformers** (≥4.35.0): Whisper model and training infrastructure
- **torch** (≥2.0.0): Deep learning framework
- **torchaudio** (≥2.0.0): Audio processing utilities
- **librosa** (≥0.10.0): Audio loading and resampling
- **jiwer** (≥3.0.0): WER computation library
- **pyyaml** (≥6.0): Configuration file parsing
- **datasets** (≥2.14.0): Data loading and processing
- **accelerate** (≥0.24.0): Distributed training support
- **soundfile**: Audio file reading backend for librosa
- **numpy**, **pandas**: Data manipulation

## Related Documentation
- [SystemDesign.md](docs/SystemDesign.md): Complete system architecture, component specifications, and implementation phases
- [Problem Description.md](docs/Problem%20Description.md): Competition goals, evaluation metrics (WER), submission format requirements
- [DrivenData Competition Rules.md](docs/DrivenData%20Competition%20Rules%20(1).md): Code execution format, external data restrictions, competition timeline

## Open Questions
1. **GPU availability**: What GPU resources are available? Whisper-medium needs ~16GB VRAM for batch_size=8. If only 12GB available, need to adjust batch size and increase gradient accumulation.

2. **Training subset for validation**: Should we start with a 10K subset to validate the full pipeline before committing to full 95K training run? This would catch bugs faster.

3. **Checkpoint storage**: Where should model checkpoints be saved? Large models (whisper-medium ~1.5GB per checkpoint) need significant storage. Consider cloud storage or local SSD.

4. **Python version**: Target Python 3.8, 3.9, 3.10, or 3.11? Affects compatibility with some dependencies. Recommend 3.10 for good balance of features and library support.

5. **Logging backend**: Use TensorBoard, Weights & Biases, or simple file logging? W&B offers better visualization but requires account setup.

6. **Development environment**: Local development with GPU, cloud notebooks (Colab/SageMaker), or dedicated ML workstation? Affects setup instructions in README.
