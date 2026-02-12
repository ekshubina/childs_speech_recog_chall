# System Design: Children's Speech Recognition Pipeline

**Project:** On Top of Pasketti - Children's Speech Recognition Challenge  
**Date:** February 12, 2026  
**Baseline Model:** Whisper-Medium  
**Design Version:** 1.0

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONFIG MANAGEMENT LAYER                     │
│  - YAML-based configuration                                      │
│  - Model selection (whisper-medium, whisper-large-v3, etc.)     │
│  - Training hyperparameters                                      │
│  - Data paths and processing options                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING PIPELINE                      │
│  - Audio loading & normalization                                 │
│  - Dataset management (95K+ training samples)                   │
│  - Data augmentation (noise injection for classroom scenarios)  │
│  - Text normalization (Whisper's English normalizer)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL ABSTRACTION LAYER                     │
│  - BaseModel interface                                           │
│  - WhisperModel implementation (medium, large-v3)               │
│  - Wav2VecModel implementation (future extension)               │
│  - Model factory pattern for easy switching                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING & EVALUATION ENGINE                  │
│  - Fine-tuning pipeline                                          │
│  - WER metric computation                                        │
│  - Checkpointing & model versioning                             │
│  - Validation on noisy classroom samples                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      INFERENCE & SUBMISSION                      │
│  - Batch inference on test data                                  │
│  - JSONL submission generation                                   │
│  - Code execution format compatibility                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
childs_speech_recog_chall/
├── configs/
│   ├── baseline_whisper_medium.yaml
│   ├── whisper_large_v3.yaml
│   └── wav2vec2_base.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset class
│   │   ├── audio_processor.py   # Audio loading/preprocessing
│   │   └── augmentation.py      # Data augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py        # Abstract base class
│   │   ├── whisper_model.py     # Whisper implementation
│   │   ├── wav2vec_model.py     # Wav2Vec implementation
│   │   └── model_factory.py     # Factory pattern
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop
│   │   └── metrics.py           # WER calculation
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py         # Inference pipeline
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Config management
│       ├── text_normalizer.py   # Whisper normalizer wrapper
│       └── logging_utils.py     # Logging utilities
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   ├── predict.py               # Prediction script
│   └── prepare_submission.py    # Submission formatter
├── notebooks/
│   ├── eda.ipynb               # Exploratory data analysis
│   └── model_comparison.ipynb  # Compare different models
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_inference.py
├── requirements.txt
├── README.md
└── submission/                  # For code execution format
    ├── main.py
    ├── model/                   # Trained model artifacts
    └── requirements.txt
└── data/                        # Audio data directories (not included in repo)
    ├── audio_0/
    ├── audio_1/
    └── audio_2/
```

---

## 3. Core Components Design

### 3.1 Configuration System

**File:** `configs/baseline_whisper_medium.yaml`

```yaml
model:
  name: "whisper"
  variant: "medium"
  pretrained: "openai/whisper-medium"
  language: "en"
  task: "transcribe"

data:
  train_manifest: "train_word_transcripts.jsonl"
  audio_base_path: "audio_{0,1,2}/"
  sample_rate: 16000
  normalize_audio: true
  
augmentation:
  enabled: true
  noise_prob: 0.3
  speed_perturb: [0.9, 1.0, 1.1]
  add_classroom_noise: true

training:
  batch_size: 8
  learning_rate: 1e-5
  num_epochs: 10
  gradient_accumulation_steps: 4
  warmup_steps: 500
  fp16: true
  
  optimizer: "adamw"
  scheduler: "linear"
  
  checkpoint_dir: "checkpoints/"
  save_steps: 1000
  eval_steps: 500
  
evaluation:
  metric: "wer"
  use_normalizer: true  # Whisper English normalizer
  split_ratio: 0.1      # For validation
  
inference:
  batch_size: 16
  beam_size: 5
  temperature: 0.0      # Greedy decoding
```

### 3.2 Base Model Interface

**File:** `src/models/base_model.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseASRModel(ABC):
    """Abstract base class for ASR models"""
    
    @abstractmethod
    def load(self, checkpoint_path: str = None) -> None:
        """Load model from checkpoint or pretrained"""
        pass
    
    @abstractmethod
    def transcribe(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files to text"""
        pass
    
    @abstractmethod
    def fine_tune(self, train_data: Any, val_data: Any, config: Dict) -> None:
        """Fine-tune the model"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return model metadata"""
        pass
```

### 3.3 Whisper Model Implementation

**File:** `src/models/whisper_model.py`

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from .base_model import BaseASRModel

class WhisperModel(BaseASRModel):
    """Whisper model implementation"""
    
    def __init__(self, variant: str = "medium", device: str = "cuda"):
        self.variant = variant
        self.device = device
        self.model_name = f"openai/whisper-{variant}"
        self.processor = None
        self.model = None
        
    def load(self, checkpoint_path: str = None) -> None:
        if checkpoint_path:
            self.model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
            self.processor = WhisperProcessor.from_pretrained(checkpoint_path)
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.config.forced_decoder_ids = None
        
    def transcribe(self, audio_paths: List[str]) -> List[str]:
        """Batch transcription of audio files"""
        # Load and preprocess audio
        # Run inference
        # Return transcriptions
        pass
    
    def fine_tune(self, train_data, val_data, config: Dict) -> None:
        """Fine-tune Whisper on children's speech data"""
        # Setup training
        # Training loop
        # Validation
        pass
    
    def save(self, path: str) -> None:
        """Save model and processor"""
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
    
    def get_model_info(self) -> Dict:
        return {
            "name": "Whisper",
            "variant": self.variant,
            "parameters": self.model.num_parameters(),
            "language": "en"
        }
```

### 3.4 Model Factory

**File:** `src/models/model_factory.py`

```python
from .base_model import BaseASRModel
from .whisper_model import WhisperModel
from .wav2vec_model import Wav2VecModel

class ModelFactory:
    """Factory for creating ASR models"""
    
    @staticmethod
    def create_model(model_config: Dict) -> BaseASRModel:
        model_name = model_config.get("name", "").lower()
        
        if model_name == "whisper":
            variant = model_config.get("variant", "medium")
            return WhisperModel(variant=variant)
        
        elif model_name == "wav2vec":
            variant = model_config.get("variant", "base")
            return Wav2VecModel(variant=variant)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
```

### 3.5 Dataset Implementation

**File:** `src/data/dataset.py`

```python
import json
import torch
from torch.utils.data import Dataset
import librosa
from pathlib import Path

class ChildSpeechDataset(Dataset):
    """Dataset for children's speech recognition"""
    
    def __init__(self, manifest_path: str, audio_base_paths: List[str], 
                 processor, sample_rate: int = 16000, augment: bool = False):
        self.manifest = self._load_manifest(manifest_path)
        self.audio_base_paths = audio_base_paths
        self.processor = processor
        self.sample_rate = sample_rate
        self.augment = augment
        
    def _load_manifest(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    def _find_audio_file(self, audio_path: str) -> str:
        """Find audio file in multiple directories"""
        for base_path in self.audio_base_paths:
            full_path = Path(base_path) / audio_path
            if full_path.exists():
                return str(full_path)
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.manifest[idx]
        audio_path = self._find_audio_file(item['audio_path'])
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Apply augmentation if enabled
        if self.augment:
            audio = self._augment_audio(audio)
        
        # Process with model processor
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        labels = self.processor.tokenizer(item['orthographic_text']).input_ids
        
        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": torch.tensor(labels),
            "utterance_id": item['utterance_id']
        }
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        # Speed perturbation
        # Add noise
        # Volume adjustment
        return audio
```

### 3.6 Training Script

**File:** `scripts/train.py`

```python
import argparse
from src.utils.config import load_config
from src.models.model_factory import ModelFactory
from src.data.dataset import ChildSpeechDataset
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create model
    model = ModelFactory.create_model(config['model'])
    if args.resume:
        model.load(args.resume)
    else:
        model.load()
    
    # Load datasets
    train_dataset = ChildSpeechDataset(
        manifest_path=config['data']['train_manifest'],
        audio_base_paths=config['data']['audio_base_path'],
        processor=model.processor,
        augment=config['augmentation']['enabled']
    )
    
    val_dataset = ChildSpeechDataset(
        manifest_path=config['data']['val_manifest'],
        audio_base_paths=config['data']['audio_base_path'],
        processor=model.processor,
        augment=False
    )
    
    # Train
    trainer = Trainer(model, train_dataset, val_dataset, config)
    trainer.train()

if __name__ == "__main__":
    main()
```

### 3.7 Metrics Computation

**File:** `src/training/metrics.py`

```python
import jiwer
from whisper.normalizers import EnglishTextNormalizer

class WERMetric:
    """Word Error Rate metric with text normalization"""
    
    def __init__(self, use_normalizer: bool = True):
        self.use_normalizer = use_normalizer
        self.normalizer = EnglishTextNormalizer() if use_normalizer else None
    
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute WER between predictions and references"""
        
        # Normalize texts if enabled
        if self.normalizer:
            predictions = [self.normalizer(p) for p in predictions]
            references = [self.normalizer(r) for r in references]
        
        # Compute WER
        wer = jiwer.wer(references, predictions)
        
        # Additional metrics
        mer = jiwer.mer(references, predictions)  # Match Error Rate
        wil = jiwer.wil(references, predictions)  # Word Information Lost
        
        return {
            "wer": wer,
            "mer": mer,
            "wil": wil
        }
```

---

## 4. Key Features

### 4.1 Model Switching
- **Configuration-based**: Switch models by changing config file
- **Unified Interface**: All models implement BaseASRModel
- **No Code Changes**: Test different models without modifying code
- **Example Usage**:
  ```bash
  python scripts/train.py --config configs/baseline_whisper_medium.yaml
  python scripts/train.py --config configs/whisper_large_v3.yaml
  ```

### 4.2 Data Augmentation
Specialized for children's speech and classroom environments:
- **Speed Perturbation**: 0.9x, 1.0x, 1.1x
- **Noise Injection**: Classroom noise, background chatter
- **Volume Adjustments**: Simulate different recording conditions
- **Time Stretching**: Handle varied speech rates

### 4.3 Text Normalization
- **Whisper's English Normalizer**: Same as competition scoring
- **Handles**:
  - Punctuation removal
  - Contraction expansion (can't → cannot)
  - Number normalization
  - Special character removal
  - Whitespace standardization

### 4.4 Evaluation
- **Primary Metric**: Word Error Rate (WER)
- **Secondary Metric**: Noisy WER (classroom samples)
- **Additional Metrics**: Match Error Rate (MER), Word Information Lost (WIL)
- **Validation Strategy**: 10% train/val split, stratified by age group

### 4.5 Code Execution Format
Structure for DrivenData submission:
```
submission/
├── main.py              # Entry point
├── predict.py           # Inference logic
├── model/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
└── requirements.txt
```

---

## 5. Model Selection Rationale

### Why Whisper-Medium as Baseline?

**Advantages:**
1. **Proven Performance**: 680K hours of diverse training data
2. **Children's Speech**: Better handles non-standard pronunciations
3. **Robustness**: Strong on noisy environments (classroom bonus)
4. **Text Normalization**: Built-in normalizer matches competition scoring
5. **Size**: Medium variant balances accuracy and inference speed

**Alternatives Considered:**
| Model | Pros | Cons | Use Case |
|-------|------|------|----------|
| Whisper-Large-v3 | Best accuracy | Slower, larger | Final optimization |
| Wav2Vec 2.0 | Fast fine-tuning | Requires more preprocessing | Quick iteration |
| Whisper-Small | Fast inference | Lower accuracy | Resource-constrained |

---

## 6. Implementation Phases

### Phase 1: Foundation (Days 1-3)
- [x] System design document
- [ ] Directory structure setup
- [ ] Configuration system
- [ ] Base model interface
- [ ] Whisper model implementation

### Phase 2: Data Pipeline (Days 4-7)
- [ ] Dataset class implementation
- [ ] Audio preprocessing
- [ ] Data augmentation
- [ ] Text normalization wrapper
- [ ] Data validation

### Phase 3: Training Infrastructure (Days 8-10)
- [ ] Training loop
- [ ] WER metric computation
- [ ] Checkpointing system
- [ ] Logging and monitoring
- [ ] Early stopping

### Phase 4: Baseline Training (Days 11-14)
- [ ] Train whisper-medium baseline
- [ ] Validate on held-out set
- [ ] Analyze errors
- [ ] Test submission format
- [ ] Document baseline WER

### Phase 5: Optimization (Days 15-21)
- [ ] Train whisper-large-v3
- [ ] Experiment with augmentation strategies
- [ ] Focus on noisy classroom samples
- [ ] Hyperparameter tuning
- [ ] Ensemble methods (if needed)

### Phase 6: Submission (Days 22-25)
- [ ] Prepare code execution package
- [ ] Test containerized inference
- [ ] Generate final predictions
- [ ] Create submission file
- [ ] Write documentation

---

## 7. Success Metrics

### Technical Metrics
- **Primary Goal**: WER < 15% on validation set
- **Stretch Goal**: WER < 10% on validation set
- **Noisy Classroom**: Noisy WER < 20% (for $5K bonus)

### Performance Metrics
- **Inference Speed**: < 1.0 RTF (Real-Time Factor)
- **Memory Usage**: < 8GB GPU VRAM for inference
- **Code Quality**: 80%+ test coverage

### Competitive Metrics
- **Leaderboard**: Top 20 for noisy classroom bonus eligibility
- **Final Rank**: Top 10 overall

---

## 8. Risk Mitigation

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|---------------------|
| Model too large for submission | High | Medium | Model quantization (int8), distillation to smaller model |
| Poor performance on noisy data | High | Medium | Heavy augmentation with classroom noise samples |
| Overfitting to training data | Medium | High | Cross-validation, early stopping, dropout, weight decay |
| Long training time | Medium | Medium | FP16 mixed precision, gradient accumulation, smaller batches |
| Out-of-vocabulary words | Low | Medium | Subword tokenization, character-level fallback |
| Non-standard pronunciations | Medium | High | Fine-tune on children's speech, use phonetic features |
| Hardware limitations | Medium | Low | Use gradient checkpointing, smaller batch sizes |

---

## 9. Testing Strategy

### Unit Tests
- Audio loading and preprocessing
- Text normalization
- Model interface compliance
- Metric computation

### Integration Tests
- End-to-end training pipeline
- Inference pipeline
- Submission format validation

### Performance Tests
- Memory profiling
- Inference speed benchmarking
- GPU utilization

---

## 10. Monitoring and Logging

### Training Monitoring
- WER on validation set (every 500 steps)
- Loss curves (train and validation)
- Learning rate schedule
- Gradient norms
- GPU memory usage

### Inference Monitoring
- Inference time per sample
- Batch processing throughput
- Memory consumption
- Error analysis by age group

### Tools
- **TensorBoard**: Training visualization
- **Weights & Biases**: Experiment tracking
- **Python logging**: Debug and info logs

---

## 11. Deployment Considerations

### Code Execution Requirements
- **Environment**: Docker container
- **Dependencies**: requirements.txt with pinned versions
- **Entry Point**: main.py with predict function
- **Output Format**: JSONL with utterance_id and orthographic_text

### Model Artifacts
- **Model Weights**: Saved in submission/model/
- **Configuration**: JSON config for reproducibility
- **Processor**: Tokenizer and feature extractor

---

## 12. Future Enhancements

### Short-term
- Implement Wav2Vec 2.0 for comparison
- Add SpecAugment for better generalization
- Ensemble multiple models

### Long-term
- Custom child-speech specific model
- Age-specific model routing
- Real-time streaming inference
- Multi-lingual support

---

## 13. References

### Competition Resources
- [Problem Description](../Problem%20Description.md)
- [Competition Rules](../DrivenData%20Competition%20Rules%20(1).md)
- Training data: `train_word_transcripts.jsonl` (95,573 samples)

### Technical References
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Whisper Repository](https://github.com/openai/whisper)
- [Wav2Vec 2.0 Paper](https://arxiv.org/abs/2006.11477)
- [WER Computation](https://github.com/jitsi/jiwer)

### Best Practices
- [ASR Fine-tuning Guide](https://huggingface.co/blog/fine-tune-whisper)
- [Children's Speech Recognition Challenges](https://www.isca-speech.org/archive/)
- [Data Augmentation for ASR](https://arxiv.org/abs/1904.08779)

---

**Document Status:** Initial Draft  
**Last Updated:** February 12, 2026  
**Next Review:** After Phase 1 completion
