# Training Enhancements Implementation Plan

## Overview

Five targeted improvements to the `baseline_whisper_small` training pipeline based on analysis of the first full training run (WER 15.19% best / 17.19% recoverable). The enhancements address a critical checkpoint bug, add regularization via SpecAugment, improve convergence with cosine LR + early stopping, speed up warmup via encoder freezing, and expose harder samples earlier via age-based curriculum. No model architecture changes are made (Whisper-small is kept).

## Goals

1. Fix the `save_steps` / `eval_steps` misalignment so the true-best checkpoint is always persisted (H1)
2. Add cosine LR schedule and early stopping to stop training at the optimal WER point (H2)
3. Add SpecAugment (frequency + time masking) to reduce overfitting on children's speech (H4)
4. Implement step-based encoder freeze warmup to stabilize early training (H3)
5. Add age-based curriculum sorting to expose easier samples first in epoch 1 (H5)

## Non-Goals

- Switching to `whisper-medium` or any other model variant
- Changes to inference, prediction, or submission scripts
- Hyperparameter search / sweeps
- Multi-epoch curriculum learning via custom sampler

---

## Implementation Steps

### Phase 1 — Config Fixes (H1 + H2 + bug fixes)

1. **Fix checkpoint alignment (H1):** Set `save_steps: 500` (was `1000`) in [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml). Prevents the true-best checkpoint from being silently unrecoverable.

2. **Add early stopping + cosine LR config (H2):** Add `early_stopping_patience: 5`, `lr_scheduler_type: cosine`, and `num_epochs: 20` (let early stopping decide) to the `training:` block in [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml). Raise `save_total_limit: 5` — with `load_best_model_at_end: true` already set, HF Trainer always protects the best checkpoint from deletion, so in practice this yields 5 most-recent saves + 1 best.

3. **Add curriculum + SpecAugment config keys:** Under `data:`, add `specaugment: true`, `freq_mask_param: 27`, `time_mask_param: 100`, `curriculum_learning: true` in [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml).

4. **Add `freeze_encoder_steps` config key:** Under `model:`, add `freeze_encoder_steps: 1000` (0 = disabled) in [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml).

### Phase 2 — Trainer Fixes + Callbacks (H2 + H3)

5. **Wire missing YAML keys into `Seq2SeqTrainingArguments`:** In `create_trainer()` in [src/training/trainer.py](../../src/training/trainer.py), add the currently-ignored kwargs: `lr_scheduler_type`, `optim`, `weight_decay`, `max_grad_norm`, `logging_dir`.

6. **Add `lr_scheduler_type` to `WhisperTrainingConfig` dataclass:** Add the field (default `"linear"`) to the dataclass in [src/training/trainer.py](../../src/training/trainer.py).

7. **Add `EarlyStoppingCallback` (H2):** In `create_trainer()`, read `early_stopping_patience` from config; instantiate `EarlyStoppingCallback` if `> 0` and pass via `callbacks=` to the `WhisperTrainer` constructor.

8. **Implement `EncoderUnfreezeCallback` (H3):** Create `EncoderUnfreezeCallback(TrainerCallback)` in [src/training/trainer.py](../../src/training/trainer.py) (or a new `src/training/callbacks.py`). It freezes `model.model.encoder.parameters()` at `on_train_begin` and unfreezes at `on_step_end` when `state.global_step >= freeze_encoder_steps`. Register it in `create_trainer()` when `freeze_encoder_steps > 0`.

### Phase 3 — SpecAugment (H4)

9. **Add `augment` flag to `ChildSpeechDataset`:** In [src/data/dataset.py](../../src/data/dataset.py), add `augment: bool = False` to `__init__`. At the end of `__getitem__`, apply `torchaudio.transforms.FrequencyMasking(freq_mask_param)` and `TimeMasking(time_mask_param)` on the `input_features` tensor (shape `80×3000`) when `augment=True`.

10. **Pass augmentation params from config:** Accept `freq_mask_param: int = 27` and `time_mask_param: int = 100` in `__init__` and use them in the transforms. Parameter rationale:
    - `freq_mask_param=27` → masks up to 27/80 = **33% of mel-frequency bins**; conservative enough to preserve children's formant structure
    - `time_mask_param=100` → masks up to 100/3000 = **3.3% of time frames** (~1 second); conservative given short children's utterances (1–5 s)
    - **If WER does not improve after one run**, increase to `freq_mask_param: 40` (50% of bins) and `time_mask_param: 150` (~1.5 s). Do not exceed `freq_mask_param: 54` or `time_mask_param: 300`.

11. **Enable `augment=True` only for train dataset:** In [scripts/train.py](../../scripts/train.py), pass `augment=True`, `freq_mask_param`, `time_mask_param` (from config) to `train_dataset = ChildSpeechDataset(...)`. Leave `val_dataset` with default `augment=False`.

### Phase 4 — Age Curriculum (H5)

12. **Sort train samples by age bucket:** In [scripts/train.py](../../scripts/train.py), define `AGE_ORDER = {'3-4': 0, '5-6': 1, '7-8': 2, '9-10': 3, '11-12': 4, '13+': 5}` and after `create_train_val_split()`, conditionally sort: `train_samples = sorted(train_samples, key=lambda s: AGE_ORDER.get(s.get('age_bucket', ''), 99))` when `curriculum_learning` is `true` in config.

---

## Success Criteria

1. `save_steps == eval_steps == 500` — every evaluated checkpoint is saved
2. Training stops automatically when WER stops improving for 5 consecutive evals
3. `eval_wer` improves beyond 15.19% (absolute) in the next run with augmentation active
4. `EncoderUnfreezeCallback` logs unfreeze event at the correct step
5. All previously-ignored YAML keys (`optim`, `weight_decay`, `max_grad_norm`, `logging_dir`) are forwarded to the HF `Seq2SeqTrainingArguments`
6. No regression on existing tests (`pytest -m "not slow"` passes)

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SpecAugment on mel spectrogram degrades WER | Medium | Use conservative params (freq=27, time=100); gate behind `specaugment: true` flag so it can be disabled |
| Encoder freeze for first 1000 steps slows convergence | Low | Callback logs unfreeze clearly; `freeze_encoder_steps: 0` disables it instantly |
| More frequent saves (500-step) exceed disk quota | Low | Raise `save_total_limit: 10`; network volume has ~200 GB |
| `torchaudio` transforms not available on RunPod pod | Low | `torchaudio` is already in `requirements.txt` |
| Curriculum sort + HF Trainer's internal shuffle interact | Low | Curriculum only affects epoch 1 ordering; Trainer reshuffles from epoch 2 onward (expected behavior) |
