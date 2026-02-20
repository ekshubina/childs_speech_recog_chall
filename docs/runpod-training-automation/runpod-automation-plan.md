# RunPod Training Automation Implementation Plan

## Overview

Implement a one-command remote training system: `scripts/pod_train.sh --config configs/baseline_whisper_small.yaml` starts or resumes a RunPod GPU Pod, syncs the latest code, runs training inside a detached tmux session, persists all stdout/stderr to the network volume, and self-stops the Pod when training completes. The local terminal can be closed at any time. The existing `scripts/train.py` and all YAML configs require zero changes.

## Goals

1. Single local command launches a full remote training run on a GPU Pod.
2. Pod self-stops (billing ends) after training completes, succeeds, or fails.
3. All outputs (`checkpoints/`, `logs/`) persist on the network volume across Pod restarts.
4. Local terminal can be disconnected at any time; training continues uninterrupted.
5. Auto-resume from the latest checkpoint on re-run; `--force-restart` to skip resume.
6. One-time data upload utility and live TensorBoard tunneling via SSH.

## Non-Goals

1. Changes to `scripts/train.py`, `src/`, or any YAML config file.
2. Orchestrating multi-Pod or distributed training.
3. Automated hyperparameter sweeps or experiment scheduling.

## Implementation Steps

### Phase 1: Repository Hygiene

1. Add `.runpod.env` to [.gitignore](.gitignore) alongside existing `.env` entries to prevent API key leaks.
2. Create [`.runpod.env.example`](.runpod.env.example) at the repo root with placeholder values for `RUNPOD_API_KEY`, `NETWORK_VOLUME_ID`, `POD_ID`, `SSH_KEY_PATH`, and `GPU_TYPE`.

### Phase 2: Pod-Side Script

3. Create [`scripts/remote_train.sh`](scripts/remote_train.sh) — executed over SSH on the Pod:
   - Reads `$CONFIG` and `$FORCE_RESTART` env vars passed over SSH.
   - Derives `OUTPUT_DIR` by extracting `output_dir` from the YAML config using `grep`/`sed` (avoids hardcoding the run name).
   - Auto-detects the latest `checkpoint-*` folder under `OUTPUT_DIR`; sets `RESUME_FLAG` accordingly; skips detection if `FORCE_RESTART=1`.
   - Writes a timestamped header to `/workspace/logs/current.log`.
   - Launches a detached tmux session (`tmux new-session -d -s train`) that: runs `git pull`, activates the venv (creating it if absent with `[[ ! -d venv ]] && python -m venv venv && pip install -r requirements.txt`), installs requirements, runs `python scripts/train.py --config "$CONFIG" $RESUME_FLAG 2>&1 | tee -a $LOG`, writes `EXIT_CODE=$?` to the log, then calls `runpodctl stop pod $RUNPOD_POD_ID`.
   - Returns immediately after `tmux new-session -d` so the SSH session closes.

### Phase 3: Local Entry Point

4. Create [`scripts/pod_train.sh`](scripts/pod_train.sh) — local orchestration script:
   - Parses `--config <path>` (required) and optional `--force-restart` flag.
   - Sources `.runpod.env`; validates that `RUNPOD_API_KEY` and `NETWORK_VOLUME_ID` are set.
   - If `POD_ID` is unset: calls `runpodctl create pod` with all flags from the design doc and writes the new ID back into `.runpod.env`.
   - If `POD_ID` is set: calls `runpodctl start pod $POD_ID`.
   - Polls `runpodctl get pod $POD_ID` every 10 seconds, parsing IP and TCP-22 port until the port appears.
   - SSHes in to upload and execute `remote_train.sh` (via `ssh … bash -s < scripts/remote_train.sh`), passing `CONFIG` and optionally `FORCE_RESTART=1` as remote env vars.
   - Opens a second SSH connection running `tail -f /workspace/logs/current.log` to stream logs; traps SIGINT to kill only the tail, printing reattach instructions.

### Phase 4: Utility Scripts

5. Create [`scripts/pod_sync_data.sh`](scripts/pod_sync_data.sh) — one-time data upload:
   - Sources `.runpod.env`, starts the Pod if stopped, polls for SSH readiness.
   - Runs `rsync -avz --progress` to copy `data/audio_0`, `data/audio_1`, `data/audio_2`, and `data/train_word_transcripts.jsonl` to `/workspace/data/` on the network volume.
   - Stops the Pod after sync completes.

6. Create [`scripts/pod_tensorboard.sh`](scripts/pod_tensorboard.sh) — SSH TensorBoard tunnel:
   - Sources `.runpod.env`, fetches live Pod IP and port.
   - On the Pod: checks for an existing `tb` tmux session; starts one with `tensorboard --logdir /workspace/childs_speech_recog_chall/logs/baseline_whisper_small --host 0.0.0.0 --port 6006` if absent.
   - Opens SSH `-L ${LOCAL_PORT:-6006}:localhost:6006` tunnel locally; Ctrl+C closes the tunnel only.

## Success Criteria

1. `./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml` starts a Pod, launches training, streams logs, and exits on Ctrl+C while training continues.
2. Re-running the same command auto-resumes from the latest checkpoint with no manual flags.
3. `--force-restart` skips auto-resume and logs `FORCE RESTART: checkpoints ignored`.
4. Pod stops automatically after training finishes (check `EXIT_CODE=0` in log).
5. `./scripts/pod_tensorboard.sh` opens `http://localhost:6006` against the running Pod.
6. `.runpod.env` is git-ignored; `.runpod.env.example` is committed.

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `runpodctl create pod --cost` flag name differs across CLI versions | High | Verify with `runpodctl create pod --help` before scripting; use `--bidPerGpu` as fallback |
| SSH port changes on every Pod start | Medium | Always re-read port from `runpodctl get pod` output — never cache in `~/.ssh/config` |
| Spot preemption mid-step | Medium | `Seq2SeqTrainer` saves every `save_steps: 1000`; re-run auto-resumes from last checkpoint |
| venv absent on first run | Low | `remote_train.sh` bootstraps venv with `[[ ! -d venv ]] && python -m venv venv` |
| `remote_train.sh` path hardcodes run name | Low | Derive `OUTPUT_DIR` from YAML `output_dir` field dynamically |
