# Child Speech Recognition Challenge - Baseline Pipeline

A modular, production-ready automatic speech recognition (ASR) system for transcribing children's speech using fine-tuned Whisper models. This baseline pipeline provides complete infrastructure for data processing, model training, inference, and evaluation on the DrivenData Child Speech Recognition Challenge dataset.

## Overview

Children's speech presents unique challenges for ASR systems due to developing vocal tracts, pronunciation variations, and diverse age-related characteristics. This project implements a baseline system using OpenAI's Whisper-small model, fine-tuned on 95K+ children's speech samples spanning ages 3-12+.

**Key Features:**
- 🎯 Modular architecture supporting easy model experimentation
- 📊 Stratified train/val splitting by age group for representative evaluation
- 🚀 Efficient training with FP16 mixed precision and gradient accumulation
- 📈 WER evaluation using Whisper's text normalizer (competition-compliant)
- 🔧 Configuration-driven workflow (no code changes for hyperparameter tuning)
- ✅ Comprehensive testing suite for data pipeline, models, and inference

## Quick Start

**🚀 Try in Google Colab (No Setup Required):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/childs_speech_recog_chall/blob/main/colab_training.ipynb)

Click the badge above to open a ready-to-use notebook with GPU support!

**📋 Local Setup:**

```bash
# Clone repository
git clone <repository-url>
cd childs_speech_recog_chall

# Install dependencies
pip install -r requirements.txt

# Quick test - validate setup (100 samples, ~5 minutes)
python scripts/train.py --config configs/baseline_whisper_small.yaml --debug

# Full training - fine-tune baseline model (86K samples, several hours)
python scripts/train.py --config configs/baseline_whisper_small.yaml

# Generate predictions
python scripts/predict.py \
    --model-path checkpoints/whisper-small-finetuned \
    --input-jsonl data/test_manifest.jsonl \
    --output-jsonl predictions.jsonl

# Evaluate on validation set
python scripts/evaluate.py \
    --model-path checkpoints/whisper-small-finetuned \
    --val-manifest data/val_manifest.jsonl
```

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU with 8GB+ VRAM (for training)
- 50GB+ disk space for audio data and model checkpoints

### Option 1: Google Colab (Recommended for Quick Start)

Google Colab provides free GPU access, making it ideal for training without local hardware. Here's how to set up:

**Step 1: Create a new Colab notebook**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `New notebook`
3. Enable GPU: `Runtime` → `Change runtime type` → `Hardware accelerator: T4 GPU`

**Step 2: Clone repository and setup**
```python
# Clone repository
!git clone https://github.com/your-username/childs_speech_recog_chall.git
%cd childs_speech_recog_chall

# Install dependencies
!pip install -q -r requirements.txt

# Verify installation
!python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Step 3: Upload data to Google Drive (one-time setup)**

Since Colab has limited session storage, mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Create project directory in Drive (first time only)
!mkdir -p /content/drive/MyDrive/child_speech_data
```

Upload your data files to `MyDrive/child_speech_data/`:
- Upload `train_word_transcripts.jsonl`
- Upload audio directories: `audio_0/`, `audio_1/`, `audio_2/`

Then link to your Colab workspace:
```python
# Create symlinks to access Drive data
!ln -s /content/drive/MyDrive/child_speech_data/train_word_transcripts.jsonl data/train_word_transcripts.jsonl
!ln -s /content/drive/MyDrive/child_speech_data/audio_0 data/audio_0
!ln -s /content/drive/MyDrive/child_speech_data/audio_1 data/audio_1
!ln -s /content/drive/MyDrive/child_speech_data/audio_2 data/audio_2
```

**Alternative: Download data from URL (if available)**
```python
# Download and extract data
!wget https://your-data-url.com/child_speech_data.tar.gz
!tar -xzf child_speech_data.tar.gz -C data/
```

**Step 4: Run training**
```python
# Quick test (5-10 minutes on T4 GPU)
!python scripts/train.py --config configs/baseline_whisper_small.yaml --debug

# Full training (will take several hours)
!python scripts/train.py --config configs/baseline_whisper_small.yaml
```

**Step 5: Save checkpoints to Drive**
```python
# Copy checkpoints to Drive for persistence
!cp -r checkpoints/baseline_whisper_small /content/drive/MyDrive/child_speech_checkpoints/
```

**Step 6: Generate predictions**
```python
!python scripts/predict.py \
    --model-path checkpoints/baseline_whisper_small/final_model \
    --input-jsonl data/test_manifest.jsonl \
    --output-jsonl predictions.jsonl

# Download predictions
from google.colab import files
files.download('predictions.jsonl')
```

**Colab Tips:**
- **Session timeout:** Colab disconnects after ~90 minutes of inactivity. Keep browser open or use [Colab Pro](https://colab.research.google.com/signup) for longer sessions
- **GPU quota:** Free tier has daily limits. Upgrade to Colab Pro for more GPU time
- **Checkpoints:** Always save to Drive! Local storage is lost when session ends
- **Monitor training:** Use TensorBoard in Colab:
  ```python
  %load_ext tensorboard
  %tensorboard --logdir logs/baseline_whisper_small
  ```
- **Debug mode first:** Always run `--debug` to verify setup before full training

**Comparison: Colab vs Local:**

| Feature | Google Colab Free | Google Colab Pro | Local (CUDA GPU) |
|---------|-------------------|------------------|------------------|
| GPU | T4 (16GB VRAM) | T4/V100 | Your hardware |
| Training time (full) | 4-6 hours | 3-4 hours | 3-8 hours |
| Session length | 12 hours max | 24 hours max | Unlimited |
| Data persistence | Via Drive only | Via Drive only | Local disk |
| Cost | Free | $10/month | Hardware cost |
| Setup time | 5 minutes | 5 minutes | 15-30 minutes |

**Complete Colab notebook template:**
```python
# === Setup ===
!git clone https://github.com/your-username/childs_speech_recog_chall.git
%cd childs_speech_recog_chall
!pip install -q -r requirements.txt

# === Mount Drive ===
from google.colab import drive
drive.mount('/content/drive')

# === Link data (adjust paths to your Drive structure) ===
!ln -sf /content/drive/MyDrive/child_speech_data/train_word_transcripts.jsonl data/train_word_transcripts.jsonl
!ln -sf /content/drive/MyDrive/child_speech_data/audio_0 data/audio_0
!ln -sf /content/drive/MyDrive/child_speech_data/audio_1 data/audio_1
!ln -sf /content/drive/MyDrive/child_speech_data/audio_2 data/audio_2

# === Verify GPU ===
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# === Quick test ===
!python scripts/train.py --config configs/baseline_whisper_small.yaml --debug

# === Full training ===
!python scripts/train.py --config configs/baseline_whisper_small.yaml

# === Save to Drive ===
!cp -r checkpoints/baseline_whisper_small /content/drive/MyDrive/child_speech_checkpoints/

# === Generate predictions ===
!python scripts/predict.py \
    --model-path checkpoints/baseline_whisper_small/final_model \
    --input-jsonl data/test_manifest.jsonl \
    --output-jsonl predictions.jsonl

# === Download results ===
from google.colab import files
files.download('predictions.jsonl')
```

### Option 2: Local Setup

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Core dependencies:
- `transformers>=4.35.0` - Hugging Face Transformers for Whisper models
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchaudio>=2.0.0` - Audio processing utilities
- `librosa>=0.10.0` - Audio loading and preprocessing
- `jiwer>=3.0.0` - Word Error Rate computation
- `openai-whisper>=20231117` - Whisper's text normalizer
- `accelerate>=0.24.0` - Distributed training support
- `tensorboard` - Training visualization and monitoring

3. **Prepare data:**
```bash
# Ensure audio files are in data/ directory:
# data/audio_0/, data/audio_1/, data/audio_2/
# data/train_word_transcripts.jsonl

# Verify data integrity
python -c "import json; data = [json.loads(l) for l in open('data/train_word_transcripts.jsonl')]; print(f'Loaded {len(data)} samples')"
```

## Project Structure

```
childs_speech_recog_chall/
├── colab_training.ipynb              # Google Colab notebook (click to open in Colab)
├── configs/                          # Configuration files
│   ├── baseline_whisper_small.yaml   # Baseline model config (default)
│   └── baseline_whisper_medium.yaml  # Alternative larger model config
├── data/                             # Data directory (audio files + manifests)
│   ├── audio_0/                      # Audio files (part 1)
│   ├── audio_1/                      # Audio files (part 2)
│   ├── audio_2/                      # Audio files (part 3)
│   ├── train_word_transcripts.jsonl  # Training manifest
│   └── submission_format_*.jsonl     # Submission templates
├── src/                              # Source code
│   ├── data/                         # Data processing modules
│   │   ├── audio_processor.py        # Audio loading, resampling, normalization
│   │   └── dataset.py                # PyTorch Dataset implementation
│   ├── models/                       # Model abstraction layer
│   │   ├── base_model.py             # Abstract base model interface
│   │   ├── whisper_model.py          # Whisper implementation
│   │   └── model_factory.py          # Model instantiation factory
│   ├── training/                     # Training infrastructure
│   │   ├── metrics.py                # WER computation with normalization
│   │   └── trainer.py                # Training loop wrapper
│   ├── inference/                    # Inference pipeline
│   │   └── predictor.py              # Batch prediction engine
│   └── utils/                        # Utility functions
│       ├── config.py                 # YAML config loader
│       ├── logging_utils.py          # Logging utilities
│       └── text_normalizer.py        # Text normalization wrapper
├── scripts/                          # Entry point scripts
│   ├── train.py                      # Training script
│   ├── predict.py                    # Inference script
│   ├── evaluate.py                   # Evaluation script
│   └── prepare_submission.py         # Submission format validator
├── tests/                            # Test suite
│   ├── test_data.py                  # Data pipeline tests
│   ├── test_models.py                # Model layer tests
│   ├── test_inference.py             # Inference pipeline tests
│   ├── test_metrics.py               # Metrics computation tests
│   └── test_integration.py           # End-to-end integration tests
├── notebooks/                        # Jupyter notebooks
│   └── eda.ipynb                     # Exploratory data analysis
├── docs/                             # Documentation
│   ├── SystemDesign.md               # Architecture documentation
│   └── Problem Description.md        # Competition requirements
├── requirements.txt                  # Python dependencies
├── requirements-dev.txt              # Development dependencies
├── pytest.ini                        # Pytest configuration
└── README.md                         # This file
```

## Usage

### Configuration

All experiments are controlled via YAML configuration files in `configs/`. Key parameters:

```yaml
model:
  name: whisper                       # Model type
  variant: small                      # Model size (tiny/base/small/medium/large)
  pretrained: openai/whisper-small    # Hugging Face model ID

data:
  train_manifest: data/train_word_transcripts.jsonl
  audio_directories:
    - data/audio_0
    - data/audio_1
    - data/audio_2
  val_split: 0.1                      # 10% validation split
  stratify_by: age_bucket             # Stratification field

training:
  batch_size: 12                      # Per-device batch size (optimized for small variant)
  learning_rate: 1.0e-05              # Learning rate
  num_epochs: 10                      # Training epochs
  gradient_accumulation_steps: 3      # Gradient accumulation
  fp16: true                          # Mixed precision training
  warmup_steps: 500                   # LR warmup steps
  save_steps: 1000                    # Checkpoint frequency
  eval_steps: 1000                    # Evaluation frequency
  output_dir: checkpoints/whisper-small-finetuned
```

Create custom configs by copying `baseline_whisper_small.yaml` and modifying parameters. A `baseline_whisper_medium.yaml` config is also available for larger-scale experiments.

### Training

**Quick test (recommended first step):**

Before running full training, validate your setup with a quick test on 100 samples:
```bash
python scripts/train.py --config configs/baseline_whisper_small.yaml --debug
```
This runs training on 100 training samples and 20 validation samples, completing in just a few minutes. Use this to:
- Verify all dependencies are installed correctly
- Test the data pipeline and model loading
- Ensure your environment has sufficient memory
- Validate the training configuration

**Full training:**
```bash
python scripts/train.py --config configs/baseline_whisper_small.yaml
```
Training 86K+ samples will take several hours depending on your hardware.

**Resume from checkpoint:**
```bash
python scripts/train.py \
    --config configs/baseline_whisper_small.yaml \
    --resume checkpoints/baseline_whisper_small/checkpoint-5000
```

**Custom validation split:**
```bash
python scripts/train.py \
    --config configs/baseline_whisper_small.yaml \
    --val-ratio 0.15  # Use 15% for validation instead of default 10%
```

**Monitor training:**

Training progress is logged to both console and `logs/train_*.log` files. Key metrics:
- `train_loss`: Training loss per batch
- `eval_loss`: Validation loss
- `eval_wer`: Validation Word Error Rate (primary metric)

Monitor in real-time:
```bash
# Watch latest log file
tail -f logs/train_*.log

# View with TensorBoard
tensorboard --logdir logs/baseline_whisper_small
```

Checkpoints are saved to `checkpoints/baseline_whisper_small/` every 1000 steps.

### Inference

**Generate predictions from trained model:**
```bash
python scripts/predict.py \
    --model-path checkpoints/whisper-small-finetuned \
    --input-jsonl data/test_manifest.jsonl \
    --output-jsonl predictions.jsonl \
    --batch-size 16
```

**Output format (JSONL):**
```json
{"utterance_id": "utt_001", "orthographic_text": "i want to go to the park"}
{"utterance_id": "utt_002", "orthographic_text": "the cat is sleeping"}
```

### Evaluation

**Evaluate model on validation set:**
```bash
python scripts/evaluate.py \
    --model-path checkpoints/whisper-small-finetuned \
    --val-manifest data/val_manifest.jsonl
```

**Sample output:**
```
Overall WER: 12.34%
WER by age group:
  - Age 3-4: 18.56%
  - Age 5-7: 13.21%
  - Age 8-11: 9.87%
  - Age 12+: 8.45%
  - Unknown: 15.32%
```

**Prepare submission:**
```bash
# Validate submission format
python scripts/prepare_submission.py \
    --submission-file predictions.jsonl \
    --template data/submission_format_*.jsonl
```

## Data Format

### Training Manifest (JSONL)
Each line contains one training sample:
```json
{
  "utterance_id": "001_a1001_014",
  "audio_path": "001/001_a1001_014.flac",
  "orthographic_text": "it is time for dinner",
  "age_bucket": "5-7"
}
```

**Fields:**
- `utterance_id`: Unique identifier
- `audio_path`: Relative path to audio file (search across audio_0/1/2 directories)
- `orthographic_text`: Ground truth transcription
- `age_bucket`: Age group (`3-4`, `5-7`, `8-11`, `12+`, `unknown`)

### Audio Files
- **Format:** FLAC (lossless compression)
- **Sample rate:** Variable (16kHz-48kHz, resampled to 16kHz during loading)
- **Channels:** Mono or stereo (converted to mono during loading)
- **Duration:** 0.5s - 30s (most samples 2-8 seconds)

## Testing

Run the test suite to verify implementation:

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_data.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html
```

**Test coverage:**
- `test_data.py`: Audio loading, dataset implementation, stratified splitting
- `test_models.py`: Model interface compliance, loading, basic inference
- `test_inference.py`: End-to-end prediction pipeline
- `test_metrics.py`: WER computation, text normalization
- `test_integration.py`: Full training/inference flow

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory (OOM)
**Symptoms:** Training crashes with `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce `batch_size` in config (try 8 or 4)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable gradient checkpointing (reduces memory at cost of speed)
- Use smaller model variant (base or tiny instead of small)

```yaml
training:
  batch_size: 8  # Reduce from 12
  gradient_accumulation_steps: 6  # Increase from 3
```

#### 2. Audio File Not Found
**Symptoms:** `FileNotFoundError: Audio file not found in any directory`

**Solutions:**
- Verify all audio directories exist: `data/audio_0/`, `data/audio_1/`, `data/audio_2/`
- Check audio file permissions
- Ensure audio_path in manifest matches actual file structure
- Debug with:
```python
from src.data.dataset import ChildSpeechDataset
dataset = ChildSpeechDataset('data/train_word_transcripts.jsonl', ['data/audio_0', 'data/audio_1', 'data/audio_2'])
# Will show which files are missing
```

#### 3. WER Mismatch with Leaderboard
**Symptoms:** Local validation WER differs significantly from leaderboard WER

**Solutions:**
- Ensure using Whisper's `EnglishTextNormalizer` (not custom normalization)
- Verify predictions are properly formatted (lowercase, no punctuation after normalization)
- Check that evaluation set matches competition test set distribution
- Run verification:
```python
from src.utils.text_normalizer import normalize_text
from src.training.metrics import WERMetric

metric = WERMetric()
pred = ["helo world"]
ref = ["hello world"]
wer = metric.compute(pred, ref)
print(f"WER: {wer:.2%}")  # Should match jiwer with normalization
```

#### 4. Slow Training
**Symptoms:** Training takes too long (>24 hours for 95K samples)

**Solutions:**
- Verify FP16 is enabled: `training.fp16: true` in config
- Check GPU utilization: `nvidia-smi` (should be >90%)
- Increase `batch_size` if memory allows
- Use `accelerate` for multi-GPU training
- Reduce `eval_steps` to evaluate less frequently

#### 5. Import Errors
**Symptoms:** `ModuleNotFoundError: No module named 'src'`

**Solutions:**
- Run scripts from repository root: `python scripts/train.py` (not `cd scripts/ && python train.py`)
- Ensure virtual environment is activated
- Verify all `__init__.py` files exist in src/ subdirectories
- Add to PYTHONPATH if needed:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 6. Checkpoint Loading Issues
**Symptoms:** `RuntimeError: Error(s) in loading state_dict`

**Solutions:**
- Verify checkpoint path is correct
- Ensure checkpoint version matches current model architecture
- Check for partial checkpoint writes (corrupted files)
- Load with strict=False for debugging:
```python
model.load_state_dict(checkpoint, strict=False)  # Shows missing/unexpected keys
```

#### 7. Text Normalization Inconsistencies
**Symptoms:** Predictions have different formatting than expected

**Solutions:**
- Use `normalize_text()` from `src.utils.text_normalizer` consistently
- Check that Whisper normalizer is properly initialized
- Verify input text encoding (should be UTF-8)
- Test normalization manually:
```python
from src.utils.text_normalizer import normalize_text
print(normalize_text("I'm going to the store!"))  # Expected: "i am going to the store"
```

### Performance Tips

1. **Use FP16 mixed precision** - Cuts memory usage by ~40% with minimal accuracy impact
2. **Gradient accumulation** - Simulate larger batch sizes without OOM
3. **Freeze encoder layers** - Fine-tune only decoder for faster training (add to config)
4. **Data caching** - Dataset caches audio file paths after first epoch
5. **Batch size tuning** - Find maximum batch size that fits in memory
6. **Distributed training** - Use `accelerate` for multi-GPU training

### Getting Help

- Check [docs/SystemDesign.md](docs/SystemDesign.md) for architecture details
- Review [docs/Problem Description.md](docs/Problem%20Description.md) for competition rules
- Search [issues](issues) for similar problems
- Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Evaluation Metrics

The competition uses **Word Error Rate (WER)** as the primary metric:

$$
\text{WER} = \frac{S + D + I}{N}
$$

Where:
- $S$ = Substitutions (wrong words)
- $D$ = Deletions (missing words)
- $I$ = Insertions (extra words)
- $N$ = Total words in reference

**Text normalization** (using Whisper's EnglishTextNormalizer):
- Convert to lowercase
- Expand contractions (I'm → i am)
- Remove punctuation
- Normalize whitespace
- Spell out numbers (1 → one)

**Examples:**
```python
Reference: "I'm going to the store."
Hypothesis: "im going to the store"
Normalized Ref: "i am going to the store"
Normalized Hyp: "im going to the store"
WER: 1/6 = 16.67% (one substitution: "im" vs "i am")
```

## Development

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Docstrings for all public functions/classes
- Keep functions focused and testable

### Adding a New Model

1. **Create model implementation:**
```python
# src/models/new_model.py
from src.models.base_model import BaseASRModel

class NewModel(BaseASRModel):
    def load(self, checkpoint_path=None):
        # Implementation
        pass
    
    def transcribe(self, audio_paths, **kwargs):
        # Implementation
        pass
    
    # ... implement all abstract methods
```

2. **Register in model factory:**
```python
# src/models/model_factory.py
def create_model(config):
    if config['model']['name'] == 'new_model':
        return NewModel(config)
    # ...
```

3. **Add configuration:**
```yaml
# configs/new_model_config.yaml
model:
  name: new_model
  variant: base
  # model-specific parameters
```

4. **Write tests:**
```python
# tests/test_models.py
def test_new_model_interface():
    model = NewModel(config)
    assert isinstance(model, BaseASRModel)
    # Test all interface methods
```

### Running Experiments

Track experiments systematically:

1. **Create experiment config:**
```bash
cp configs/baseline_whisper_small.yaml configs/experiment_01_higher_lr.yaml
# Edit hyperparameters
```

2. **Run training:**
```bash
python scripts/train.py --config configs/experiment_01_higher_lr.yaml
```

3. **Log results:**
```bash
# Results automatically saved to checkpoints/<output_dir>/results.json
```

4. **Compare experiments:**
```python
# notebooks/compare_experiments.ipynb
import json
exp1 = json.load(open('checkpoints/exp1/results.json'))
exp2 = json.load(open('checkpoints/exp2/results.json'))
print(f"Exp1 WER: {exp1['best_wer']:.2%}")
print(f"Exp2 WER: {exp2['best_wer']:.2%}")
```

## Roadmap

### Phase 1: Baseline (Current)
- ✅ Data pipeline with multi-directory support
- ✅ Whisper-small fine-tuning infrastructure
- ✅ WER evaluation with text normalization
- ✅ Training scripts and inference pipeline
- ✅ Comprehensive testing suite

### Phase 2: Optimization (Planned)
- [ ] Data augmentation (SpecAugment, speed perturbation)
- [ ] Hyperparameter tuning (learning rate, batch size)
- [ ] Advanced training techniques (knowledge distillation)
- [ ] Model ensemble approaches
- [ ] Age-specific model fine-tuning

### Phase 3: Production (Planned)
- [ ] Docker containerization for code submission
- [ ] Optimized inference pipeline (batch processing)
- [ ] Model compression (quantization, pruning)
- [ ] Deployment documentation
- [ ] Final submission preparation

## License

This project is for educational and competition purposes. Please refer to the DrivenData Competition Rules for usage restrictions.

## Acknowledgments

- OpenAI for Whisper model
- Hugging Face for Transformers library
- DrivenData for hosting the competition
- Contributors and community feedback

## Citation

```bibtex
@misc{child-speech-recognition-challenge,
  title={Child Speech Recognition Challenge - Baseline Pipeline},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/your-username/childs_speech_recog_chall}}
}
```

---

**Last Updated:** February 2026  
**Version:** 1.0.0  
**Status:** Baseline Implementation Complete
