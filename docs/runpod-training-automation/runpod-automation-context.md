# RunPod Training Automation Context & References

## Key Files

### Existing Code to Modify

| File | Purpose | Changes Needed |
|------|---------|----------------|
| [.gitignore](.gitignore) | Git ignore rules | Add `.runpod.env` entry alongside existing `.env` entries |

### New Files to Create

| File | Purpose |
|------|---------|
| [.runpod.env.example](.runpod.env.example) | Committed template with placeholder values; users copy to `.runpod.env` |
| [scripts/pod_train.sh](scripts/pod_train.sh) | Local entry point — creates/resumes Pod, polls SSH, launches remote training, tails log |
| [scripts/remote_train.sh](scripts/remote_train.sh) | Pod-side script — detects checkpoints, launches tmux session, stops Pod on exit |
| [scripts/pod_sync_data.sh](scripts/pod_sync_data.sh) | One-time rsync of `data/` to network volume |
| [scripts/pod_tensorboard.sh](scripts/pod_tensorboard.sh) | SSH tunnel for local TensorBoard against running Pod |

### Reference Implementations

| File | Relevance |
|------|-----------|
| [scripts/train.py](scripts/train.py) | Already supports `--config` (required) and `--resume <path>` (optional); no changes needed |
| [scripts/tensorboard_local.sh](scripts/tensorboard_local.sh) | Pattern reference for local TensorBoard launch (Google Drive–based; `pod_tensorboard.sh` is the SSH-tunnel equivalent) |
| [scripts/start_bore_server.sh](scripts/start_bore_server.sh) | Pattern reference for SSH-adjacent data transfer; `pod_sync_data.sh` uses rsync instead |
| [configs/baseline_whisper_small.yaml](configs/baseline_whisper_small.yaml) | `output_dir: checkpoints/baseline_whisper_small`, `logging_dir: logs/baseline_whisper_small`, `audio_dirs: [data/audio_0, ...]`, `save_steps: 1000` — all relative paths, resolve correctly under `/workspace/childs_speech_recog_chall/` on Pod |
| [src/training/trainer.py](src/training/trainer.py) | `WhisperTrainer.create_trainer()` accepts `resume_from_checkpoint`; actual resume happens via `trainer.train(resume_from_checkpoint=args.resume)` — no changes required |
| [docs/runpod-training-automation.md](docs/runpod-training-automation.md) | Full system design document — authoritative reference for all script logic and Pod configuration |

## Architecture Decisions

### Decision 1: Dynamic `OUTPUT_DIR` Derivation in `remote_train.sh`

- **Context**: The design doc hardcodes `OUTPUT_DIR="$REPO/checkpoints/baseline_whisper_small"`, which breaks when using `--config configs/baseline_whisper_medium.yaml`.
- **Decision**: Extract `output_dir` from the YAML config at runtime using `grep`/`sed` inside `remote_train.sh`.
- **Rationale**: Enables any config to be passed via `--config` without script changes — matches Requirement 9 ("no code changes to the existing training pipeline" extended to scripts).
- **Alternatives Considered**: Hardcode per-config; pass `OUTPUT_DIR` as an additional env var from `pod_train.sh`.

### Decision 2: Upload `remote_train.sh` via stdin over SSH

- **Context**: `remote_train.sh` needs to run on the Pod, but the Pod may not have it pre-installed on first run.
- **Decision**: Use `ssh … bash -s < scripts/remote_train.sh` to stream the script over SSH — no SCP step needed.
- **Rationale**: Simpler, avoids file placement concerns; after the first `git pull` inside tmux the script will be present in the repo for inspection.
- **Alternatives Considered**: SCP the script first; keep the script content inline in `pod_train.sh`.

### Decision 3: venv Bootstrap in `remote_train.sh`

- **Context**: The design doc assumes `venv` already exists at `/workspace/childs_speech_recog_chall/venv`, but the First-Time Setup Checklist notes it must be created manually.
- **Decision**: Add `[[ ! -d venv ]] && python -m venv venv` before `pip install -r requirements.txt` inside the tmux session.
- **Rationale**: Makes the script self-healing on first run without requiring a separate setup step; adds negligible overhead on subsequent runs.
- **Alternatives Considered**: Separate `pod_setup.sh` script; document as a manual prerequisite only.

### Decision 4: `--cost` Flag Verification Deferred

- **Context**: `runpodctl create pod --cost` flag name varies across CLI versions (may be `--bidPerGpu` or `--startBid`).
- **Decision**: Add a comment in `pod_train.sh` to run `runpodctl create pod --help` and note both known flag names; use the most common one as default.
- **Rationale**: Can't test at plan time; guard with clear error message if the flag is rejected.
- **Alternatives Considered**: Use RunPod GraphQL API directly; use `runpodctl` only for start/stop and create via UI.

## Dependencies

### Internal Dependencies

- `scripts/train.py`: consumed by `remote_train.sh` via `python scripts/train.py --config ... --resume ...`
- `configs/baseline_whisper_small.yaml` / `configs/baseline_whisper_medium.yaml`: passed as `$CONFIG` from `pod_train.sh` to `remote_train.sh`
- `.runpod.env`: sourced by all local scripts (`pod_train.sh`, `pod_sync_data.sh`, `pod_tensorboard.sh`)

### External Dependencies

- `runpodctl`: RunPod CLI — `brew install runpod/runpodctl/runpodctl`; must be configured with `runpodctl config --apiKey`
- `rsync`: standard macOS utility; used in `pod_sync_data.sh`
- `tmux`: must be available on the RunPod container image (`runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` includes it)
- `ssh`, `tail`: standard macOS utilities for log streaming in `pod_train.sh`

## RunPod Infrastructure

### Pod Creation Command (Reference)

```bash
runpodctl create pod \
  --name "whisper-training" \
  --gpuType "NVIDIA GeForce RTX 4090" \
  --communityCloud \
  --cost 0.50 \                # verify flag name: may be --bidPerGpu
  --imageName "runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204" \
  --containerDiskSize 50 \
  --networkVolumeId $NETWORK_VOLUME_ID \
  --ports "22/tcp" \
  --env "HF_HOME=/workspace/.cache/huggingface" \
  --env "PYTHONPATH=/workspace/childs_speech_recog_chall"
```

### Network Volume Layout

```
/workspace/
├── data/                               # uploaded once via pod_sync_data.sh
│   ├── audio_0/  audio_1/  audio_2/
│   └── train_word_transcripts.jsonl
├── childs_speech_recog_chall/          # git clone; git pull on each run
│   ├── venv/                           # bootstrapped by remote_train.sh
│   ├── checkpoints/baseline_whisper_small/
│   └── logs/baseline_whisper_small/
├── logs/
│   └── current.log                     # all train stdout+stderr, appended
└── .cache/huggingface/                 # model weights, downloaded once
```

## Related Documentation

- [docs/runpod-training-automation.md](docs/runpod-training-automation.md): Full system design — authoritative source for all script logic
- [docs/SystemDesign.md](docs/SystemDesign.md): Overall project architecture
- [docs/CI-CD-SETUP.md](docs/CI-CD-SETUP.md): CI/CD context

## Open Questions

1. Confirm `runpodctl create pod` flag for bid price: `--cost`, `--bidPerGpu`, or `--startBid`? Run `runpodctl create pod --help` to verify before scripting.
2. Does the `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` image include `tmux`? (Likely yes, but worth confirming on first Pod start.)
3. Should `pod_sync_data.sh` stop the Pod after upload, or leave it running for immediate training?
