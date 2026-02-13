# Whisper-Small Migration Implementation Plan

## Overview
Migrate the baseline ASR model from Whisper-medium (769M parameters, ~16GB VRAM) to Whisper-small (244M parameters, ~8GB VRAM) to enable faster training iterations, reduce resource requirements, and maintain competitive accuracy for the children's speech recognition competition. This change leverages the project's factory pattern architecture for a clean model swap while optimizing batch configuration to take advantage of the smaller memory footprint.

## Goals
1. Establish Whisper-small as the new baseline model with optimized training configuration
2. Maintain backward compatibility by preserving medium config for future experiments
3. Update all documentation, tests, and examples to reflect the new baseline
4. Archive existing medium model checkpoints for historical reference
5. Validate the new configuration with a smoke test before full training runs

## Non-Goals
1. Training a full model immediately (validation only)
2. Adding a comprehensive model comparison section to documentation
3. Deleting or modifying the medium model configuration
4. Retraining existing medium model checkpoints with the new config

## Implementation Steps

### Phase 1: Configuration Setup
1. **Create new baseline config** - Copy [configs/baseline_whisper_medium.yaml](configs/baseline_whisper_medium.yaml) to `configs/baseline_whisper_small.yaml` and update:
   - `model.variant: small`
   - `model.pretrained: openai/whisper-small`
   - `training.batch_size: 12` (up from 8, with explanatory comment)
   - `training.gradient_accumulation_steps: 3` (down from 4, maintains effective batch ~36)
   - `training.output_dir: checkpoints/baseline_whisper_small`
   - `training.logging_dir: logs/baseline_whisper_small`
   - `evaluation.predictions_file: predictions/baseline_whisper_small_val.jsonl`

2. **Add configuration documentation** - Insert YAML comment explaining batch size optimization: "Increased from 8 to 12 due to Whisper-small's lower memory footprint (~8GB vs ~16GB for medium)"

### Phase 2: Test Updates
1. **Update default model test** - In [tests/test_models.py](tests/test_models.py), update the default loading test (around line 196-198) to expect `openai/whisper-small` instead of `openai/whisper-medium`

2. **Update factory test config** - In [tests/test_models.py](tests/test_models.py), update the factory test configuration (around line 494) to use `openai/whisper-small`

### Phase 3: Documentation Updates
1. **Update README.md** - In [README.md](README.md):
   - Replace `baseline_whisper_medium.yaml` references with `baseline_whisper_small.yaml` in training commands
   - Update VRAM requirement from ~16GB to ~8GB
   - Update batch_size examples from 8 to 12
   - Add note that medium config is preserved for experimentation

2. **Update copilot instructions** - In [.github/copilot-instructions.md](.github/copilot-instructions.md):
   - Replace config examples using medium with small
   - Update memory assumptions from 16GB to 8GB VRAM
   - Update batch configuration examples

### Phase 4: Checkpoint Management
1. **Archive medium checkpoints** - Create archive directory structure and move existing runs:
   ```bash
   mkdir -p checkpoints/archived
   mv checkpoints/baseline_whisper_medium checkpoints/archived/
   ```

2. **Document archive location** - Ensure archived runs are preserved with TensorBoard logs intact for future comparison

### Phase 5: Validation
1. **Smoke test configuration** - Run debug training to validate:
   ```bash
   source venv/bin/activate
   python scripts/train.py --config configs/baseline_whisper_small.yaml --debug
   ```

2. **Verify expectations**:
   - Config loads without errors
   - Model loads as `WhisperForConditionalGeneration` with small variant
   - Processor initializes as `WhisperProcessor` for small
   - Training starts without OOM errors
   - Batch size of 12 works on target hardware

3. **Run test suite** - Execute `pytest tests/test_models.py` to confirm updated assertions pass

## Success Criteria
1. **Configuration valid**: New `baseline_whisper_small.yaml` loads successfully with all required fields
2. **Model loads correctly**: Whisper-small model and processor initialize from HuggingFace Hub
3. **No OOM errors**: Smoke test completes at least one training step with batch_size=12
4. **Tests pass**: All model-related tests pass with updated small variant expectations
5. **Documentation consistent**: All baseline examples reference small variant and correct memory requirements
6. **Archive preserved**: Medium model checkpoints and TensorBoard logs accessible in archive directory

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Batch size increase causes OOM | High | Start with debug mode (10 samples), monitor GPU memory usage, have fallback to batch_size=8 |
| Lower accuracy with small model | Medium | Accept trade-off for faster iteration; medium config preserved for final optimization |
| Breaking existing scripts referencing medium config | Low | Keep medium config file unchanged; update only documentation defaults |
| Test failures due to hardcoded expectations | Low | Update tests proactively as part of migration; run full test suite before committing |
| Loss of medium model training history | Low | Archive checkpoints instead of deleting; preserve TensorBoard logs |
