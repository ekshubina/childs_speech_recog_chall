# RunPod Training Automation Task Checklist

## Summary

- **Dependencies**: `runpodctl` CLI installed and configured locally; RunPod account with API key and network volume created; SSH public key added to RunPod account settings.
- **Zero changes** to `scripts/train.py`, `src/`, or any YAML config.

## Phase 1: Repository Hygiene

- [x] **Add `.runpod.env` to `.gitignore`** — append `.runpod.env` entry in [.gitignore](.gitignore) alongside existing `.env` and `.env.local` entries
- [x] **Create `.runpod.env.example`** — committed template at [.runpod.env.example](.runpod.env.example) with:
  ```
  RUNPOD_API_KEY=your_api_key_here
  NETWORK_VOLUME_ID=your_volume_id_here
  POD_ID=                          # written automatically after first pod_train.sh run
  SSH_KEY_PATH=~/.ssh/id_ed25519
  GPU_TYPE=NVIDIA GeForce RTX 4090
  ```

## Phase 2: Pod-Side Script

- [x] **Create `scripts/remote_train.sh`** — Pod-side script at [scripts/remote_train.sh](scripts/remote_train.sh):
  - Accept `$CONFIG` and `$FORCE_RESTART` as env vars (passed over SSH)
  - Extract `OUTPUT_DIR` from the YAML config dynamically: `grep 'output_dir' "$REPO/$CONFIG" | sed 's/.*: //'`
  - Bootstrap venv if absent: `[[ ! -d venv ]] && python -m venv venv && pip install -r requirements.txt`
  - Auto-detect latest `checkpoint-*` under `OUTPUT_DIR`; skip if `FORCE_RESTART=1`
  - Write timestamped run header to `/workspace/logs/current.log`
  - Launch detached tmux session `train` with: `git pull → pip install → python scripts/train.py --config "$CONFIG" $RESUME_FLAG 2>&1 | tee -a $LOG → EXIT_CODE → runpodctl stop pod`
  - Return immediately after `tmux new-session -d`
  - Mark executable: `chmod +x scripts/remote_train.sh`

## Phase 3: Local Entry Point

- [x] **Create `scripts/pod_train.sh`** — local orchestration script at [scripts/pod_train.sh](scripts/pod_train.sh):
  - Parse `--config <path>` (required) and `--force-restart` (optional flag)
  - Source `.runpod.env`; validate `RUNPOD_API_KEY` and `NETWORK_VOLUME_ID` are non-empty
  - Branch on `POD_ID`: if unset → `runpodctl create pod` with all flags + write new ID to `.runpod.env`; if set → `runpodctl start pod $POD_ID`
  - Poll `runpodctl get pod $POD_ID` every 10 s until TCP-22 port appears; print dots while waiting
  - SSH in and execute `remote_train.sh` via `ssh … -o StrictHostKeyChecking=no bash -s < scripts/remote_train.sh` with `CONFIG` and `FORCE_RESTART` env vars set
  - Open second SSH connection: `ssh … "tail -f /workspace/logs/current.log"`
  - Trap SIGINT: kill only the tail SSH process; print reattach commands before exiting
  - Mark executable: `chmod +x scripts/pod_train.sh`

## Phase 4: Utility Scripts

- [x] **Create `scripts/pod_sync_data.sh`** — one-time data upload at [scripts/pod_sync_data.sh](scripts/pod_sync_data.sh):
  - Source `.runpod.env`; start Pod if stopped; poll for SSH readiness (reuse polling logic from `pod_train.sh`)
  - Run `rsync -avz --progress -e "ssh -p $PORT -i $SSH_KEY_PATH" data/audio_0 data/audio_1 data/audio_2 data/train_word_transcripts.jsonl root@$IP:/workspace/data/`
  - Print estimated time and stop Pod after completion
  - Mark executable: `chmod +x scripts/pod_sync_data.sh`

- [x] **Create `scripts/pod_tensorboard.sh`** — SSH TensorBoard tunnel at [scripts/pod_tensorboard.sh](scripts/pod_tensorboard.sh):
  - Source `.runpod.env`; fetch live Pod IP and port from `runpodctl get pod`
  - SSH into Pod: `tmux has-session -t tb 2>/dev/null || tmux new-session -d -s tb 'tensorboard --logdir /workspace/childs_speech_recog_chall/logs/baseline_whisper_small --host 0.0.0.0 --port 6006'`
  - Open local tunnel: `ssh -N -L ${LOCAL_PORT:-6006}:localhost:6006 root@$IP -p $PORT`
  - Print `http://localhost:$LOCAL_PORT` and tunnel-close instructions
  - Accept optional first arg for local port: `./scripts/pod_tensorboard.sh 6007`
  - Mark executable: `chmod +x scripts/pod_tensorboard.sh`

## Phase 5: Validation

- [ ] **Smoke-test `.gitignore`** — confirm `git status` does not show `.runpod.env` as tracked after creating it locally
- [ ] **Dry-run `pod_train.sh` arg parsing** — run `./scripts/pod_train.sh --help` (or with bad args) and confirm error messages are clear
- [ ] **First-time Pod creation** — run `./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml` with `POD_ID=` unset; confirm Pod is created, ID written to `.runpod.env`, SSH polling works, tmux session starts, log streams
- [ ] **Resume run** — re-run same command after Pod stops; confirm latest `checkpoint-*` is detected and `--resume` is passed; check log header shows `RESUMING FROM:`
- [ ] **Force-restart run** — run with `--force-restart`; confirm log header shows `FORCE RESTART: checkpoints ignored`
- [ ] **Pod self-stop** — wait for training to end; confirm Pod transitions to stopped state and `EXIT_CODE=0` appears in log
- [ ] **TensorBoard tunnel** — run `./scripts/pod_tensorboard.sh` while training; confirm `http://localhost:6006` loads

## Phase 6: Documentation

- [x] **Document full end-to-end workflow in README or docs** — write a usage guide covering the complete cycle:
  1. First-time setup: install `runpodctl`, create network volume, fill `.runpod.env`, run `pod_sync_data.sh`
  2. Launch training: `./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml`
  3. Monitor live: log streaming in terminal; reattach with `tail -f` or `tmux attach`
  4. Track progress: `./scripts/pod_tensorboard.sh` → `http://localhost:6006`
  5. Pod self-stops: verify `EXIT_CODE=0` in log; checkpoints persisted on volume
  6. Resume after preemption or manual stop: re-run same `pod_train.sh` command — auto-resumes
  7. Force fresh run: add `--force-restart` flag
  8. Scale to larger model: change `--config` and GPU type in `.runpod.env`

## Implementation Order

1. Phase 1: Repository hygiene (`.gitignore`, `.runpod.env.example`) — no dependencies, safe to do immediately
2. Phase 2: `remote_train.sh` — must exist before `pod_train.sh` references it
3. Phase 3: `pod_train.sh` — depends on `remote_train.sh`
4. Phase 4: `pod_sync_data.sh` and `pod_tensorboard.sh` — independent of each other; depend on polling logic from `pod_train.sh` (can extract to a shared helper or duplicate)
5. Phase 5: Validation — requires RunPod account, network volume, and data to be available
6. Phase 6: Documentation — write after validation confirms the end-to-end workflow works as designed

## Acceptance Criteria

Each task should be:
- **Testable**: `pod_train.sh` streams a log line within 60 s of SSH being ready
- **Atomic**: each script is a standalone executable
- **Specific**: each script has a single, well-defined responsibility
- **Actionable**: all referenced flags and paths come directly from the design doc at [docs/runpod-training-automation.md](docs/runpod-training-automation.md)
