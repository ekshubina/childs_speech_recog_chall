# Training Enhancements Task Checklist

## Summary
- **Goal**: Fix checkpoint bug + add SpecAugment, cosine LR, early stopping, encoder freeze warmup, age curriculum
- **Dependencies**: None — all changes are additive; no external packages needed beyond existing requirements
- **Last Updated**: 2026-02-21

---

## Phase 1 — Config Fixes (no code changes)

- [x] **Fix `save_steps` alignment (H1)** — Change `save_steps: 1000` → `save_steps: 500` in [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml) to match `eval_steps: 500`
- [x] **Raise `save_total_limit`** — Change `save_total_limit: 3` → `save_total_limit: 5` in [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml); HF Trainer always preserves the best checkpoint (guarded by `load_best_model_at_end: true`) so in practice you get 5 recent + 1 best
- [x] **Add cosine LR schedule (H2)** — Add `lr_scheduler_type: cosine` under `training:` in [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml)
- [x] **Add early stopping config (H2)** — Add `early_stopping_patience: 5` under `training:` and set `num_epochs: 20` in [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml)
- [x] **Add SpecAugment config (H4)** — Add `specaugment: true`, `freq_mask_param: 27`, `time_mask_param: 100` under `data:` in [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml). Starting values: `freq=27` = 33% of 80 mel bins; `time=100` = 3.3% of 3000 frames (~1 s). If WER does not improve after one run, raise to `freq=40, time=150`. See [context doc](training-enhancements-context.md) for full parameter rationale.
- [x] **Add curriculum config (H5)** — Add `curriculum_learning: true` under `data:` in [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml)
- [x] **Add encoder freeze config (H3)** — Add `freeze_encoder_steps: 1000` under `model:` in [configs/baseline_whisper_small.yaml](../../configs/baseline_whisper_small.yaml)

---

## Phase 2 — Trainer Fixes (bug fixes + H2 + H3)

- [x] **Add `lr_scheduler_type` to `WhisperTrainingConfig` dataclass** — Add field `lr_scheduler_type: str = "linear"` to the `@dataclass` in [src/training/trainer.py](../../src/training/trainer.py)
- [x] **Wire `lr_scheduler_type` into `Seq2SeqTrainingArguments`** — In `create_trainer()`, pass `lr_scheduler_type=training_config.get("lr_scheduler_type", "linear")` to `Seq2SeqTrainingArguments` in [src/training/trainer.py](../../src/training/trainer.py)
- [x] **Wire missing YAML keys (bug fix)** — Add `optim`, `weight_decay`, `max_grad_norm`, `logging_dir` as explicit kwargs to `Seq2SeqTrainingArguments` in `create_trainer()` in [src/training/trainer.py](../../src/training/trainer.py)
- [x] **Add `EarlyStoppingCallback` (H2)** — Import `EarlyStoppingCallback` from `transformers`; read `early_stopping_patience` from config in `create_trainer()`; instantiate and pass via `callbacks=` to `WhisperTrainer(...)` if patience `> 0` in [src/training/trainer.py](../../src/training/trainer.py)
- [x] **Create `EncoderUnfreezeCallback` (H3)** — Create [src/training/callbacks.py](../../src/training/callbacks.py) with `EncoderUnfreezeCallback(TrainerCallback)` that: freezes `model.model.encoder.parameters()` at `on_train_begin`, logs the freeze; unfreezes at `on_step_end` when `state.global_step >= self.freeze_steps`, logs the unfreeze
- [x] **Register `EncoderUnfreezeCallback` in `create_trainer()` (H3)** — Read `freeze_encoder_steps` from `config["model"]`; if `> 0`, append `EncoderUnfreezeCallback` to the callbacks list in [src/training/trainer.py](../../src/training/trainer.py)

---

## Phase 3 — SpecAugment in Dataset (H4)

- [x] **Add `augment` flag + params to `ChildSpeechDataset.__init__()`** — Add `augment: bool = False`, `freq_mask_param: int = 27`, `time_mask_param: int = 100` parameters in [src/data/dataset.py](../../src/data/dataset.py)
- [x] **Apply SpecAugment transforms in `__getitem__()`** — After `input_features = self.processor(...).input_features[0]`, when `self.augment is True`, apply `torchaudio.transforms.FrequencyMasking(self.freq_mask_param)(input_features)` then `torchaudio.transforms.TimeMasking(self.time_mask_param)(input_features)` in [src/data/dataset.py](../../src/data/dataset.py). Both transforms are stochastic and applied per-sample. Default params (`freq=27`, `time=100`) are conservative — see [context doc](training-enhancements-context.md) §Decision 2 for tuning guidance.
- [x] **Enable augmentation on train dataset only** — In [scripts/train.py](../../scripts/train.py), read `specaugment`, `freq_mask_param`, `time_mask_param` from `config["data"]` and pass `augment=True`, `freq_mask_param=...`, `time_mask_param=...` only to `train_dataset = ChildSpeechDataset(...)`; leave `val_dataset` with defaults

---

## Phase 4 — Age Curriculum Sort (H5)

- [x] **Define `AGE_ORDER` mapping** — Add `AGE_ORDER = {'3-4': 0, '5-6': 1, '7-8': 2, '9-10': 3, '11-12': 4, '13+': 5}` near the top of [scripts/train.py](../../scripts/train.py)
- [x] **Apply curriculum sort conditionally** — After `create_train_val_split()` returns `train_samples`, read `curriculum_learning` from `config["data"]`; if `True`, sort: `train_samples = sorted(train_samples, key=lambda s: AGE_ORDER.get(s.get('age_bucket', ''), 99))` in [scripts/train.py](../../scripts/train.py)
- [x] **Log curriculum sort** — Add a logger message when curriculum sort is applied, listing the sorted age-bucket distribution in [scripts/train.py](../../scripts/train.py)

---

## Phase 5 — Tests

- [x] **Test SpecAugment flag** — Add test in [tests/test_data.py](../../tests/test_data.py) asserting that `ChildSpeechDataset(augment=True)` returns different `input_features` tensors across two calls on the same sample (stochastic transforms)
- [x] **Test `EncoderUnfreezeCallback`** — Add unit test in [tests/test_models.py](../../tests/test_models.py) (or a new `tests/test_callbacks.py`) that instanciates the callback, calls `on_train_begin` and verifies encoder params have `requires_grad=False`, then calls `on_step_end` at the target step and verifies `requires_grad=True`
- [ ] **Smoke test full training script** — Run `./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml --debug` to verify no crashes with new config keys and callbacks active

---

## Implementation Order

1. **Phase 1** (YAML config) — No code risk; do all config changes first
2. **Phase 2 — Trainer bug fixes + early stopping** — Wire missing keys, add `EarlyStoppingCallback`
3. **Phase 2 — `EncoderUnfreezeCallback`** — Create `callbacks.py`, register in trainer
4. **Phase 3 — SpecAugment** — Dataset changes + train.py wiring
5. **Phase 4 — Curriculum sort** — Single-file change in `train.py`
6. **Phase 5 — Tests** — Verify callbacks + augmentation behave correctly
7. **Smoke test** — `--debug` run on RunPod before a full training run

---

## Acceptance Criteria

Each task should be:
- **Testable**: New behavior is verifiable (callback logs, augmented vs. non-augmented tensors differ, WER checkpoint saved at correct step)
- **Atomic**: Each checkbox is completable independently without breaking other tasks
- **Specific**: Focused on a single file/function change
- **Reversible**: All new behavior gated behind config keys (`specaugment: false`, `curriculum_learning: false`, `freeze_encoder_steps: 0`, `early_stopping_patience: 0`) so the baseline run can be reproduced exactly
