# Training Enhancements Context & References

## Key Files

### Existing Code to Modify

| File | Purpose | Changes Needed |
|------|---------|----------------|
| [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml) | Main training config | Fix `save_steps`, add `lr_scheduler_type`, `early_stopping_patience`, `freeze_encoder_steps`, SpecAugment params, curriculum flag |
| [src/training/trainer.py](../../src/training/trainer.py) | Wraps `Seq2SeqTrainer`, builds `Seq2SeqTrainingArguments` | Wire missing YAML keys, add `EarlyStoppingCallback`, add `EncoderUnfreezeCallback` |
| [src/data/dataset.py](../../src/data/dataset.py) | `ChildSpeechDataset` + `WhisperDataCollator` | Add `augment` flag and SpecAugment transforms to `__getitem__` |
| [scripts/train.py](../../scripts/train.py) | Training entry point | Pass `augment=True` to train dataset, add curriculum sort |

### New Files to Create

| File | Purpose |
|------|---------|
| [src/training/callbacks.py](../../src/training/callbacks.py) | `EncoderUnfreezeCallback` — freezes encoder at train start, unfreezes at target step |

### Reference Implementations

| File | Relevance |
|------|-----------|
| [src/models/whisper_model.py](../../src/models/whisper_model.py) | `prepare_model_for_finetuning()` — existing encoder freeze pattern; `EncoderUnfreezeCallback` mirrors its freeze logic at runtime |
| [src/training/metrics.py](../../src/training/metrics.py) | `WERMetric` — how `compute_metrics` closure is wired; early stopping depends on it returning `"wer"` key |

---

## Architecture Decisions

### Decision 1: SpecAugment at Dataset level, not Collator level
- **Context**: SpecAugment can be applied per-sample (in `__getitem__`) or per-batch (in `WhisperDataCollator.__call__`).
- **Decision**: Apply in `__getitem__`, gated by `augment: bool` flag on `ChildSpeechDataset`.
- **Rationale**: Dataset-level is cleaner — `augment=True` only for train, default `False` for val. Collator level would require passing a train/eval flag into the shared collator.
- **Alternatives Considered**: Batch-level in `WhisperDataCollator` — rejected because the same collator is shared between train and eval dataloaders; distinguishing train vs. eval batches at that level is fragile.

### Decision 2: SpecAugment parameter values
- **Context**: Whisper's mel spectrogram is always `80 mel-frequency bins × 3000 time frames` (= 30 seconds at 10 ms hop). SpecAugment masks contiguous blocks along each axis independently.
- **Decision**: Start with `freq_mask_param: 27` and `time_mask_param: 100`.
- **Rationale**:
  - `freq_mask_param=27`: each mask can hide at most 27 / 80 = **33% of mel bins**. Conservative — wide enough to force the model to rely on spectral context rather than isolated frequency peaks, but narrow enough not to destroy formant information critical for children's vowels.
  - `time_mask_param=100`: each mask can hide at most 100 / 3000 = **3.3% of the time axis** (~1 second). Very conservative — children's utterances are short (1–5 s) so hiding more would erase significant content.
- **Tuning guidance**: If WER does not improve after one full run with these values, increase to `freq_mask_param: 40` (50% of bins) and `time_mask_param: 150` (~1.5 s). Do not exceed `freq_mask_param: 54` (67%) or `time_mask_param: 300` (3 s) — beyond these thresholds augmentation degrades WER on children's speech.
- **Alternatives Considered**: Applying two frequency masks and two time masks (as in the original SpecAugment paper, `num_freq_masks=2`, `num_time_masks=2`). Deferred — `torchaudio` `FrequencyMasking`/`TimeMasking` apply a single mask per call; multiple masks would require calling the transform twice. Can be added as a follow-up.

### Decision 3: `EncoderUnfreezeCallback` via `TrainerCallback`, not scheduled in `prepare_model_for_finetuning`
- **Context**: Encoder should be frozen for the first `freeze_encoder_steps` steps, then unfrozen for the rest of training.
- **Decision**: Implement as a `transformers.TrainerCallback` registered in `create_trainer()`.
- **Rationale**: `TrainerCallback` has access to `state.global_step` and `model` at each step — the cleanest integration point. Doing it in `train.py` would require a custom training loop.
- **Alternatives Considered**: Custom training loop in `scripts/train.py` — rejected as too invasive.

### Decision 3: Curriculum only for epoch 1 (simple list sort, not custom Sampler)
- **Context**: Age-based curriculum could be applied every epoch (hard) or just epoch 1 (easy).
- **Decision**: Simple `sorted()` on `train_samples` list before `ChildSpeechDataset` construction.
- **Rationale**: HF `Trainer` shuffles the dataset internally from epoch 2 onward (via its `DataLoader`). Epoch-1-only ordering is sufficient to test the hypothesis with minimal code. A custom `Sampler` / `get_train_dataloader()` override can be added later if multi-epoch curriculum is needed.
- **Alternatives Considered**: `torch.utils.data.Sampler` subclass overriding `WhisperTrainer.get_train_dataloader()` — deferred as future work.

### Decision 4: Raise `num_epochs: 20`, let early stopping terminate
- **Context**: Best WER was at epoch 6.1 in the first run. With only 10 epochs configured, there's no room to see if improvement continues.
- **Decision**: Set `num_epochs: 20` and `early_stopping_patience: 5` (= 5 × 500 steps = 2.5 epochs of patience).
- **Rationale**: Early stopping with patience 5 will terminate ~2.5 epochs after the best WER, which is safe. If stopping fires too early, increase patience in config without code changes.

---

## Dependencies

### Internal Dependencies
- `WhisperTrainer.create_trainer()`: Central configuration point — all callbacks and argument changes live here
- `ChildSpeechDataset.__getitem__()`: SpecAugment insertion point; must not affect the val set
- `create_train_val_split()`: Returns `(train_samples, val_samples)` list — curriculum sort applied after this call
- `WERMetric` / `compute_metrics`: Must return `{"wer": float}` for `EarlyStoppingCallback` to work (already does)

### External Dependencies
- `torchaudio >= 2.0.0`: Provides `FrequencyMasking` and `TimeMasking` transforms (already in `requirements.txt`)
- `transformers.EarlyStoppingCallback`: Built into `transformers >= 4.35.0` (already installed)
- `transformers.TrainerCallback`: Base class for `EncoderUnfreezeCallback` (built-in)

---

## Key Findings from Training Run Analysis

### Checkpoint Bug (H1)
- `eval_steps: 500` / `save_steps: 1000` → best WER of **0.1519** at step 5500 was never saved
- The trainer's `best_model_checkpoint` points to `checkpoint-5000` which evaluates to **WER 0.1719**
- Fix: `save_steps: 500` — trivial one-line YAML change, no code required

### Overfitting Pattern
- Train loss: 7.4 → 0.027 (step 8000); Eval loss: 0.3216 (step 1500 min) → 0.5062 (step 8000)
- Eval WER improved from step 1500 to 5500 despite rising eval loss — decoder output distributions sharpen
- WER plateau after step 5500: oscillates in [0.152–0.168] for 3 epochs with no clear trend
- Verdict: training beyond epoch 6 gives diminishing returns; early stopping + SpecAugment regularization address this

### Missing `Seq2SeqTrainingArguments` Forwards (bugs)
These YAML keys are parsed but **never passed** to HF Trainer:

| YAML key | Value | Fix location |
|---|---|---|
| `optim` | `adamw_torch` | `Seq2SeqTrainingArguments` in `trainer.py` |
| `weight_decay` | `0.01` | `Seq2SeqTrainingArguments` in `trainer.py` |
| `max_grad_norm` | `1.0` | `Seq2SeqTrainingArguments` in `trainer.py` |
| `logging_dir` | `logs/baseline_whisper_small` | `Seq2SeqTrainingArguments` in `trainer.py` |

### GPU Utilization
- Peak VRAM: 22.01 GB / 25.26 GB (87.1%) — healthy, 2.5 GB headroom
- With SpecAugment (CPU-side transform), no VRAM impact expected

---

## Open Questions

1. Should `EncoderUnfreezeCallback` live in `src/training/trainer.py` (keeps everything in one file) or a new `src/training/callbacks.py` (better separation)? → Recommend `callbacks.py` for extensibility.
2. Should `freq_mask_param` and `time_mask_param` be tunable per-run via CLI args, or YAML-only? → YAML-only is sufficient for now.
3. Should curriculum sorting apply a single pass (epoch 1 only, current plan) or implement a decreasing temperature schedule across epochs? → Epoch-1-only for this iteration.

---

## Related Documentation
- [docs/SystemDesign.md](../SystemDesign.md): Overall system architecture
- [docs/runpod-training-automation/runpod-usage-guide.md](../runpod-training-automation/runpod-usage-guide.md): How to run training on RunPod
- [.github/copilot-instructions.md](../../.github/copilot-instructions.md): Project patterns and gotchas
