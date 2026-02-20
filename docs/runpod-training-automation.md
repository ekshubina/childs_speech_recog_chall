# RunPod Training Automation

## System Design: One-Command RunPod Training Automation

Run `scripts/pod_train.sh --config configs/baseline_whisper_small.yaml` locally → a GPU Pod is created (or resumed), code is synced, training runs inside a **tmux session** on the Pod, all stdout/stderr is persisted to the network volume, and the Pod self-stops when done. You can close your terminal at any time — training continues uninterrupted on the Pod.

---

## Requirements

1. A single local shell command launches a full training run on a remote GPU Pod.
2. Pod self-stops (GPU billing ends) after training completes, whether it succeeds or fails.
3. All outputs — `checkpoints/`, `logs/`, `predictions/` — persist on the network volume across Pod deletions.
4. Training stdout/stderr is persisted to `/workspace/logs/current.log` on the network volume.
5. Local terminal can be closed at any time — training continues running on the Pod.
6. Local script tails the remote log live; Ctrl+C disconnects the tail without killing training.
7. Re-running `pod_train.sh` automatically resumes from the latest checkpoint if one exists — no manual path needed.
8. Subsequent runs reuse the same stopped Pod (faster resume vs. cold create).
8. TensorBoard can be opened locally against the running Pod with a single command.
9. No code changes to the existing training pipeline.

---

## High-Level Architecture

```
Local machine                          RunPod
──────────────────────                 ──────────────────────────────────
pod_train.sh --config ...
  │
  ├─ .runpod.env                       Network Volume (persistent, /workspace)
  │   POD_ID, VOLUME_ID, API_KEY         ├── data/
  │                                      │    ├── audio_0/  audio_1/  audio_2/
  ├─ runpodctl start pod $POD_ID         │    └── train_word_transcripts.jsonl
  │   (or create pod on first run)       ├── childs_speech_recog_chall/  ← repo
  │                                      │    └── (git pull on each run)
  ├─ poll: runpodctl get pod $POD_ID     ├── checkpoints/baseline_whisper_small/
  │   until TCP port 22 is live          ├── logs/
  │                                      │    └── current.log  ← stdout+stderr
  ├─ ssh → remote_train.sh              └── .cache/huggingface/  ← model weights
  │    └─ tmux new-session -d
  │         git pull                   Pod (ephemeral GPU)
  │         python train.py              ├── mounts /workspace
  │           | tee current.log          ├── tmux session "train" (detached)
  │         runpodctl stop pod           │    ├── python scripts/train.py
  │                                      │    ├── stdout → /workspace/logs/current.log
  └─ ssh tail -f current.log            │    └── runpodctl stop pod  (on EXIT)
      ↑ Ctrl+C anytime,
        training keeps running
```

---

## Components

### 1. `scripts/pod_train.sh` — Local Entry Point

Reads `.runpod.env` for `RUNPOD_API_KEY`, `POD_ID`, `NETWORK_VOLUME_ID`.

- If `POD_ID` is unset → calls `runpodctl create pod` with all flags and writes the new ID back to `.runpod.env`
- If `POD_ID` is set → calls `runpodctl start pod $POD_ID` to resume the stopped Pod
- Polls `runpodctl get pod $POD_ID` every 10 seconds until a TCP 22 port mapping appears
- SSHes in and executes `remote_train.sh "$CONFIG"` — which starts a detached tmux session and returns immediately
- Passes `FORCE_RESTART=1` env var over SSH when `--force-restart` flag is given
- Attaches a second SSH connection to `tail -f /workspace/logs/current.log` to stream logs locally
- **Ctrl+C on the local terminal** kills only the tail — training keeps running on the Pod
- To reattach to live logs later: `ssh root@$IP -p $PORT "tail -f /workspace/logs/current.log"`
- To attach to the tmux session interactively: `ssh root@$IP -p $PORT "tmux attach -t train"`

**Usage:**
```bash
# Normal run — auto-resumes from latest checkpoint if one exists
./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml

# Force fresh start — ignores existing checkpoints, starts from epoch 0
./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml --force-restart
```

---

### 2. `scripts/remote_train.sh` — Runs on Pod

Executed over SSH. Starts a **detached tmux session** and returns immediately so closing the local terminal has no effect.

The tmux session runs this sequence:

```bash
#!/usr/bin/env bash
LOG=/workspace/logs/current.log
REPO=/workspace/childs_speech_recog_chall
OUTPUT_DIR="$REPO/checkpoints/baseline_whisper_small"
mkdir -p /workspace/logs

# Auto-detect latest checkpoint — skipped if FORCE_RESTART=1
if [[ "${FORCE_RESTART:-0}" == "1" ]]; then
  RESUME_FLAG=""
else
  LATEST_CKPT=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
  RESUME_FLAG=${LATEST_CKPT:+--resume "$LATEST_CKPT"}
fi

# Log run header
echo "" >> $LOG
echo "====== $(date '+%Y-%m-%d %H:%M:%S') ======" >> $LOG
echo "CONFIG: $CONFIG" >> $LOG
[[ "${FORCE_RESTART:-0}" == "1" ]] && echo "FORCE RESTART: checkpoints ignored" >> $LOG
[[ -n "$RESUME_FLAG" ]] \
  && echo "RESUMING FROM: $LATEST_CKPT" >> $LOG \
  || echo "STARTING FRESH" >> $LOG

tmux new-session -d -s train "
  cd $REPO && git fetch origin && git checkout ${BRANCH:-main} && git pull origin ${BRANCH:-main}
  source venv/bin/activate
  pip install -q -r requirements.txt
  python scripts/train.py --config '$CONFIG' $RESUME_FLAG \
    2>&1 | tee -a $LOG
  echo \"EXIT_CODE=\$?\" >> $LOG
  runpodctl stop pod \$RUNPOD_POD_ID
"
```

Key properties:
- **Auto-resume**: before launching `train.py`, detects the latest `checkpoint-N` folder in `output_dir`; if found, passes `--resume` automatically; if not, starts from scratch — no manual intervention needed
- **`--force-restart`**: when passed via `pod_train.sh`, sets `FORCE_RESTART=1` over SSH — auto-resume is skipped entirely and training starts from epoch 0; existing checkpoints are **not deleted** and remain on the volume
- **Code + deps sync**: runs `git fetch/checkout/pull` then `pip install -r requirements.txt` so every run uses the latest pushed code and dependencies
- **All stdout + stderr** are written to `/workspace/logs/current.log` (persisted on network volume) via `tee -a`
- Logs accumulate across runs — each new run appends a timestamped header so runs are distinguishable
- `runpodctl stop pod` is the **last command inside tmux** — fires on clean exit, Python crash, or OOM kill
- `EXIT_CODE=<N>` is written to the log so the outcome is inspectable after the Pod stops
- `remote_train.sh` itself exits immediately after `tmux new-session -d` — SSH session closes, tmux keeps running

---

### 3. `.runpod.env` — Local Config (git-ignored)

```bash
RUNPOD_API_KEY=your_api_key_here
NETWORK_VOLUME_ID=your_volume_id_here
POD_ID=                          # written automatically after first pod_train.sh run
SSH_KEY_PATH=~/.ssh/id_ed25519
```

Makes `pod_train.sh` idempotent — running it twice reuses the same stopped Pod instead of creating a new one. Add `.runpod.env` to `.gitignore`.

---

### 4. `runpodctl create pod` — First-Time Pod Creation

```bash
runpodctl create pod \
  --name "whisper-training" \
  --gpuType "NVIDIA GeForce RTX 4090" \
  --communityCloud \
  --cost 0.50 \
  --imageName "runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204" \
  --containerDiskSize 50 \
  --networkVolumeId $NETWORK_VOLUME_ID \
  --ports "22/tcp" \
  --env "HF_HOME=/workspace/.cache/huggingface" \
  --env "PYTHONPATH=/workspace/childs_speech_recog_chall"
```

**GPU options:**

| GPU | VRAM | Spot price | Best for |
|-----|------|-----------|----------|
| RTX 4090 (`NVIDIA GeForce RTX 4090`) | 24 GB | ~$0.20/hr | `whisper-small`, `whisper-medium` |
| A100 PCIe (`NVIDIA A100 80GB PCIe`) | 80 GB | ~$0.60/hr | `whisper-large` (final training) |
| A100 SXM4 (`NVIDIA A100-SXM4-80GB`) | 80 GB | ~$0.79/hr | `whisper-large` (faster, NVLink) |

---

### 5. Network Volume — Persistent Storage

- **Size:** ~100 GB (`$0.07/GB/month ≈ $7/month`)
- **Mount point:** `/workspace` (automatic on Pod start)
- **Created once in the RunPod console**, reused forever
- **Contents:**

```
/workspace/
├── data/
│   ├── audio_0/               # 95K+ audio files (uploaded once)
│   ├── audio_1/
│   ├── audio_2/
│   └── train_word_transcripts.jsonl
├── childs_speech_recog_chall/ # repo clone
├── checkpoints/
│   └── baseline_whisper_small/
│       ├── checkpoint-1000/
│       ├── checkpoint-2000/
│       └── final_model/
├── logs/
│   └── current.log            # all train.py stdout+stderr, appended each run
└── .cache/huggingface/        # model weights (downloaded once)
```

All paths in `configs/baseline_whisper_small.yaml` (`output_dir`, `logging_dir`, `audio_dirs`) resolve correctly under `/workspace/childs_speech_recog_chall/` with **no config changes**.

---

### 6. `scripts/pod_sync_data.sh` — One-Time Data Upload

Run once to populate the network volume before the first training run:

```bash
# Start pod, get SSH port, rsync data, stop pod
./scripts/pod_sync_data.sh
```

Internally uses:
```bash
rsync -avz --progress \
  -e "ssh -p $PORT -i $SSH_KEY_PATH" \
  data/audio_0 data/audio_1 data/audio_2 \
  data/train_word_transcripts.jsonl \
  root@$POD_IP:/workspace/data/
```

Subsequent `pod_train.sh` runs skip data transfer entirely — the volume already has everything.

---

### 7. `scripts/pod_tensorboard.sh` — Local TensorBoard via SSH Tunnel

Starts TensorBoard on the Pod inside a tmux session (if not already running), then opens an SSH port-forward tunnel so `http://localhost:6006` works in your local browser. Safe to run while training is in progress.

```bash
#!/usr/bin/env bash
# Usage: ./scripts/pod_tensorboard.sh
# Opens TensorBoard at http://localhost:6006

set -euo pipefail
source "$(dirname "$0")/../.runpod.env"

# Resolve live SSH connection details
POD_INFO=$(runpodctl get pod "$POD_ID" 2>/dev/null)
POD_IP=$(echo "$POD_INFO"  | grep -oP '(?<=IP: )\S+')
POD_PORT=$(echo "$POD_INFO" | grep -oP '(?<=Port: )\d+')
SSH_CMD="ssh -i ${SSH_KEY_PATH:-~/.ssh/id_ed25519} -o StrictHostKeyChecking=no root@$POD_IP -p $POD_PORT"

TB_LOGDIR="/workspace/childs_speech_recog_chall/logs/baseline_whisper_small"
LOCAL_PORT="${1:-6006}"

echo "Starting TensorBoard on Pod (if not running)..."
$SSH_CMD "tmux has-session -t tb 2>/dev/null || \
  tmux new-session -d -s tb \
  'tensorboard --logdir $TB_LOGDIR --host 0.0.0.0 --port 6006'"

echo "Tunnel open: http://localhost:$LOCAL_PORT  (Ctrl+C to close tunnel, TensorBoard keeps running)"
ssh -N -L "${LOCAL_PORT}:localhost:6006" \
  -i "${SSH_KEY_PATH:-~/.ssh/id_ed25519}" \
  -o StrictHostKeyChecking=no \
  root@"$POD_IP" -p "$POD_PORT"
```

**What it does:**
- Checks if a `tb` tmux session already exists on the Pod; starts one if not
- Opens an SSH `-L` port-forward tunnel: local `6006` → Pod `6006`
- Ctrl+C closes the tunnel only — TensorBoard keeps running on the Pod
- Re-run the script anytime to reopen the tunnel (e.g. after closing the terminal)
- Optional: pass a different local port as the first argument: `./scripts/pod_tensorboard.sh 6007`

---

## Billing Model

| State | What's billed |
|-------|--------------|
| Pod running (training) | GPU + container disk (per second) |
| Pod stopped | Volume disk only (`$0.20/GB/month`) |
| Pod terminated | Nothing |
| Network volume (always) | `$0.07/GB/month` |

**Typical cost estimate (whisper-small, RTX 4090 spot):**
- ~4 hours training × $0.20/hr ≈ **$0.80 per training run**
- Network volume 100 GB × $0.07 ≈ **$7/month** (one-time fixed cost)

---

## Further Considerations

### Spot Preemption Recovery

`Seq2SeqTrainer` checkpoints every `save_steps: 1000` steps (configured in `configs/baseline_whisper_small.yaml`). On preemption:
- RunPod sends SIGTERM with 5 seconds warning — not enough to finish a step, but the last saved checkpoint is intact on the network volume
- The partial log up to the moment of preemption is preserved in `/workspace/logs/current.log`
- Simply re-run `./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml` — `remote_train.sh` auto-detects the latest checkpoint and resumes from it with no extra flags needed

### Checking Training Status Without Reconnecting

Since `EXIT_CODE=<N>` is written to the log at the end of each run, you can check outcome after the Pod has stopped:
```bash
# After pod_train.sh is done, rsync just the log:
rsync -avz -e "ssh -p $PORT" root@$IP:/workspace/logs/current.log ./logs/
grep "EXIT_CODE" logs/current.log
```

### Force-Stopping a Running Training Session

If training is in progress and you need to stop it immediately:

```bash
# Option A — kill tmux session on Pod (training stops, Pod self-stops via trap)
ssh root@$IP -p $PORT "tmux kill-session -t train"

# Option B — stop the Pod directly from your machine (fastest)
runpodctl stop pod $POD_ID
```

Both options leave all checkpoints intact on the network volume.

### Starting Fresh (Ignoring Existing Checkpoints)

```bash
./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml --force-restart
```

- Skips auto-resume — training starts from epoch 0 with fresh optimizer/scheduler state
- Existing `checkpoint-*` folders are **preserved** on the volume, not deleted
- The log header will show `FORCE RESTART: checkpoints ignored` for traceability
- Use this when: changing model architecture, switching to a very different config, or when a checkpoint is corrupted

### SSH Port Churn

`runpodctl start pod` on a stopped Pod assigns a new public port for TCP 22 each time. `pod_train.sh` always re-reads the live port from `runpodctl get pod` output — never caches it — so stale `~/.ssh/config` entries are never an issue.

### Scaling to Larger Models

The only change needed to train `whisper-medium` or `whisper-large`:
1. Point to a different config: `--config configs/baseline_whisper_medium.yaml`
2. Set `GPU_TYPE=NVIDIA A100 80GB PCIe` (or `NVIDIA A100-SXM4-80GB`) in `.runpod.env`
3. Increase `batch_size: 24` + `gradient_accumulation_steps: 1` in the config (same effective batch of 40, but ~4× faster on A100)

No script changes required.

---

## First-Time Setup Checklist

- [ ] Create a RunPod account and fund it
- [ ] Generate a RunPod API key (Account Settings → API Keys)
- [ ] Install `runpodctl`: `brew install runpod/runpodctl/runpodctl`
- [ ] Configure: `runpodctl config --apiKey YOUR_KEY`
- [ ] Create a network volume of **100 GB** in the RunPod console, note its ID
- [ ] Add your SSH public key in RunPod Account Settings
- [ ] Copy `.runpod.env.example` → `.runpod.env`, fill in values
- [ ] Run `scripts/pod_sync_data.sh` to upload audio data (one time, ~hours)
- [ ] Clone repo on the Pod to `/workspace/childs_speech_recog_chall` and set up venv (one time)
- [ ] Run `scripts/pod_train.sh --config configs/baseline_whisper_small.yaml`
- [ ] (Optional) Run `scripts/pod_tensorboard.sh` in a separate terminal to monitor training at `http://localhost:6006`
