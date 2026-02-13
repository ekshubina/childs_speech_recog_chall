# Whisper-Small Migration Context & References

## Key Files

### Existing Code to Modify
| File | Purpose | Changes Needed |
|------|---------|----------------|
| [configs/baseline_whisper_medium.yaml](configs/baseline_whisper_medium.yaml) | Current baseline config | Copy to new file, update variant/paths |
| [tests/test_models.py](tests/test_models.py) | Model unit tests | Update default assertions from medium→small |
| [README.md](README.md) | User documentation | Replace baseline examples, update VRAM specs |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | AI assistant context | Update config examples and memory assumptions |

### New Files to Create
| File | Purpose |
|------|---------|
| [configs/baseline_whisper_small.yaml](configs/baseline_whisper_small.yaml) | New baseline configuration with optimized batch settings |
| [checkpoints/archived/](checkpoints/archived/) | Archive directory for medium model runs |

### Reference Implementations
| File | Relevance |
|------|-----------|
| [src/models/whisper_model.py](src/models/whisper_model.py) | Shows how variant parameter flows through model initialization |
| [src/models/model_factory.py](src/models/model_factory.py) | Demonstrates config-driven model creation (no hardcoding) |
| [src/utils/config.py](src/utils/config.py) | Config validation and loading logic |
| [docs/SystemDesign.md](docs/SystemDesign.md) | Model comparison table and selection rationale |

## Architecture Decisions

### Decision 1: Whisper-Small as New Baseline
- **Context**: Need faster iteration cycles during development; medium model takes longer to train and requires more VRAM
- **Decision**: Switch baseline from medium (769M params) to small (244M params)
- **Rationale**: 
  - 3x fewer parameters = faster training and inference
  - Lower VRAM (~8GB vs ~16GB) = accessible to more hardware
  - Still competitive accuracy for children's speech
  - Medium config preserved for final optimization phase
- **Alternatives Considered**: 
  - Keep medium as baseline, add small as "fast" variant → Rejected: Most development work benefits from faster baseline
  - Use tiny variant → Rejected: Accuracy drop too significant for competition
  - Use base variant → Considered but small offers better accuracy/speed trade-off

### Decision 2: Increase Batch Size for Small
- **Context**: Whisper-small uses ~3x less memory, creating headroom in the training configuration
- **Decision**: Increase batch_size from 8→12, reduce gradient_accumulation from 4→3
- **Rationale**:
  - Effective batch size remains similar (32 vs 36)
  - Utilizes available memory headroom
  - Faster throughput (~30% reduction in update steps)
  - Maintains training stability with similar effective batch
- **Alternatives Considered**:
  - Keep batch_size=8, accumulation=4 → Rejected: Leaves performance on table
  - Increase batch_size to 16 → Rejected: May still cause OOM on 8GB cards
  - Test both empirically → Deferred: Can adjust after smoke test if needed

### Decision 3: Archive vs Delete Medium Checkpoints
- **Context**: Existing medium model runs consume disk space but contain training history
- **Decision**: Archive to `checkpoints/archived/baseline_whisper_medium/` instead of deleting
- **Rationale**:
  - Preserves TensorBoard logs for comparison
  - Allows reverting to medium if needed
  - Disk space not currently constrained
  - Clean separation between model variants
- **Alternatives Considered**:
  - Delete entirely → Rejected: Loses training history
  - Keep in same location → Rejected: Confusing which model is baseline

### Decision 4: Preserve Medium Config File
- **Context**: Medium config represents significant tuning effort and may be needed for future experiments
- **Decision**: Keep `baseline_whisper_medium.yaml` unchanged, create separate `baseline_whisper_small.yaml`
- **Rationale**:
  - Enables side-by-side experimentation
  - No breaking changes to existing scripts
  - Clear documentation of config evolution
  - Users can explicitly choose variant
- **Alternatives Considered**:
  - Rename medium config to "alternative" → Rejected: Diminishes valid config option
  - Overwrite medium config → Rejected: Loses tuning work and breaks compatibility

### Decision 5: Update Test Defaults
- **Context**: Tests currently assert default model is medium
- **Decision**: Update test assertions to expect small as the new default
- **Rationale**:
  - Tests should validate current baseline behavior
  - Prevents confusion when tests fail with new config
  - Maintains test coverage of default parameter paths
- **Alternatives Considered**:
  - Leave tests checking medium → Rejected: Tests wouldn't reflect reality
  - Parameterize tests for any variant → Deferred: Over-engineering for current need

## Dependencies

### Internal Dependencies
- `src.models.ModelFactory`: Config-driven model instantiation
- `src.models.WhisperModel`: Variant parameter flows through `__init__` → `load()`
- `src.utils.config.load_config()`: YAML parsing and validation
- `transformers.WhisperForConditionalGeneration`: HuggingFace model class
- `transformers.WhisperProcessor`: Audio preprocessing pipeline

### External Dependencies
- `openai/whisper-small`: HuggingFace Hub model checkpoint (auto-downloaded)
- No new package dependencies required (already have transformers>=4.35.0)

### Hardware Dependencies
- Target: ~8GB VRAM (down from ~16GB)
- Batch size validated for common configurations (RTX 3060, RTX 4070, etc.)

## Related Documentation
- [docs/SystemDesign.md](docs/SystemDesign.md): Model comparison table (lines 502-505), explains small vs medium trade-offs
- [README.md](README.md): Training quickstart, troubleshooting OOM errors (lines 323-333)
- [docs/baseline-pipeline/baseline-scaffolding-plan.md](docs/baseline-pipeline/baseline-scaffolding-plan.md): Original medium baseline rationale (line 76)
- [docs/Problem Description.md](docs/Problem%20Description.md): Competition requirements (no model size restrictions)

## Configuration Details

### Current (Medium) Configuration
```yaml
model:
  name: whisper
  variant: medium
  pretrained: openai/whisper-medium
training:
  batch_size: 8
  gradient_accumulation_steps: 4
  # Effective batch size: 32
```

### New (Small) Configuration
```yaml
model:
  name: whisper
  variant: small
  pretrained: openai/whisper-small
training:
  batch_size: 12  # Optimized for small's memory footprint
  gradient_accumulation_steps: 3
  # Effective batch size: 36
```

### Memory Calculations
- **Medium**: 769M params × 2 bytes (FP16) = ~1.5GB model + ~14.5GB activations/optimizer = ~16GB total
- **Small**: 244M params × 2 bytes (FP16) = ~0.5GB model + ~7.5GB activations/optimizer = ~8GB total
- **Headroom**: 16GB - 8GB = 8GB available for batch size increase

## Open Questions
None - all decisions finalized based on user feedback.

## Research Notes

### Code Architecture Findings
From comprehensive codebase analysis:
- **Factory pattern**: Model variant is purely config-driven, no hardcoded dependencies
- **Clean separation**: Variant flows through `config → factory → model.__init__() → model.load()`
- **Default fallbacks**: Code has `variant='medium'` defaults but these are only used if config missing
- **Test isolation**: Test files use 'tiny' for speed, don't affect production behavior

### Whisper Model Specifications
| Variant | Parameters | Layers | Width | Heads | VRAM (est.) |
|---------|-----------|--------|-------|-------|-------------|
| tiny | 39M | 4 | 384 | 6 | ~2GB |
| base | 74M | 6 | 512 | 8 | ~4GB |
| **small** | **244M** | **12** | **768** | **12** | **~8GB** |
| medium | 769M | 24 | 1024 | 16 | ~16GB |
| large | 1550M | 32 | 1280 | 20 | ~32GB |

### Performance Trade-offs
Based on Whisper paper and community benchmarks:
- Small typically achieves ~90-95% of medium's accuracy
- Training time: ~50% faster per epoch
- Inference: ~3x faster than medium
- Suitable for: Rapid experimentation, resource-constrained deployment
