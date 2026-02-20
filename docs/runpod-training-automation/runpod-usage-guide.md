# RunPod Training — Usage Guide

One-command remote training on a GPU Pod. Close your terminal at any time — training continues uninterrupted.

---

## Prerequisites (First Time Only)

1. **Create a RunPod account** and fund it at [runpod.io](https://www.runpod.io)
2. **Generate an API key**: Account Settings → API Keys
3. **Install runpodctl**:
   ```bash
   brew install runpod/runpodctl/runpodctl
   runpodctl config --apiKey YOUR_KEY
   ```
4. **Create a network volume** of **100 GB** in the RunPod console. Note its ID.
   - Covers: audio data ~12 GB + venv ~8 GB + HF model cache up to ~15 GB (whisper-large) + checkpoints + buffer
   - Cost: `$0.07/GB/month` → ~$7/month
5. **Add your SSH public key** in RunPod Account Settings
6. **Configure `.runpod.env`**:
   ```bash
   cp .runpod.env.example .runpod.env
   # Edit .runpod.env — fill in RUNPOD_API_KEY, NETWORK_VOLUME_ID, SSH_KEY_PATH
   ```
7. **Upload training data** (one time):
   ```bash
   # Install aws CLI if needed
   brew install awscli

   # Fill in S3 credentials in .runpod.env first:
   # RUNPOD_S3_BUCKET, RUNPOD_S3_ENDPOINT, RUNPOD_S3_REGION,
   # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
   # (from: RunPod console → Network Volumes → your volume → S3 API Access)

   ./scripts/pod_sync_data.sh
   ```
   Uploads directly via the **RunPod S3 API** — no Pod needs to be running, no GPU billing during transfer. Uses `aws s3 sync` with parallel multipart uploads and automatic resume. Files land at `/workspace/data/` when any Pod mounts the volume.

---

## Full End-to-End Workflow

### Step 1 — Launch Training

```bash
./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml
```

What happens:
- Reads `.runpod.env` for credentials and `POD_ID`
- **First run** (`POD_ID` unset): creates a new Pod and writes its ID to `.runpod.env`
- **Subsequent runs** (`POD_ID` set): resumes the stopped Pod (faster than cold create)
- Polls until SSH port 22 is live, then uploads and executes `remote_train.sh` over SSH
- `remote_train.sh` starts training inside a **detached tmux session** on the Pod and returns immediately
- Local script then tails `/workspace/logs/current.log` live in your terminal

### Step 2 — Monitor Live

Log output streams to your terminal automatically. To disconnect without stopping training:

```
Ctrl+C
```

The log tail closes; training continues on the Pod. To reattach later:

```bash
# Reattach log stream
ssh -i ~/.ssh/id_ed25519 root@<IP> -p <PORT> "tail -f /workspace/logs/current.log"

# Attach to interactive tmux session
ssh -i ~/.ssh/id_ed25519 root@<IP> -p <PORT> "tmux attach -t train"
```

### Step 3 — Track Progress with TensorBoard

Open a **second terminal** and run:

```bash
./scripts/pod_tensorboard.sh
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

- TensorBoard starts on the Pod (inside a `tb` tmux session) if not already running
- SSH port-forward tunnel opens: local `6006` → Pod `6006`
- Ctrl+C closes the tunnel only — TensorBoard keeps running
- Re-run the script anytime to reopen the tunnel
- Optional: use a different local port: `./scripts/pod_tensorboard.sh 6007`

### Step 4 — Pod Self-Stops After Training

When training finishes (or crashes), the Pod automatically stops. GPU billing ends immediately. To confirm:

```bash
# Check the log for exit code (after Pod stops, rsync just the log):
rsync -avz -e "ssh -p <PORT>" root@<IP>:/workspace/logs/current.log ./logs/
grep "EXIT_CODE" logs/current.log
```

`EXIT_CODE=0` means training completed successfully.

### Step 5 — Resume After Preemption or Manual Stop

Simply re-run the same command:

```bash
./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml
```

`remote_train.sh` automatically detects the latest `checkpoint-*` folder on the network volume and passes `--resume` to `train.py`. The log header will show:

```
RESUMING FROM: /workspace/childs_speech_recog_chall/checkpoints/baseline_whisper_small/checkpoint-5000
```

No manual flags needed.

### Step 6 — Force a Fresh Run (Ignore Checkpoints)

```bash
./scripts/pod_train.sh --config configs/baseline_whisper_small.yaml --force-restart
```

- Skips auto-resume — training starts from epoch 0
- Existing `checkpoint-*` folders are **preserved** on the volume
- Log header shows `FORCE RESTART: checkpoints ignored`

Use this when: changing model architecture, switching configs significantly, or recovering from a corrupted checkpoint.

---

## Scaling to Larger Models

The only changes needed for `whisper-medium` or `whisper-large`:

1. Point to a different config: `--config configs/baseline_whisper_medium.yaml`
2. Update `GPU_TYPE` in `.runpod.env`:
   ```
   # A100 PCIe — $0.60/hr community spot:
   GPU_TYPE=NVIDIA A100 80GB PCIe
   # A100 SXM4 — $0.79/hr, faster NVLink:
   # GPU_TYPE=NVIDIA A100-SXM4-80GB
   ```
3. Adjust `batch_size` and `gradient_accumulation_steps` in the config for VRAM headroom

No script changes required.

---

## Force-Stopping a Running Session

```bash
# Option A — kill tmux on Pod (training stops; Pod self-stops via trap)
ssh -i ~/.ssh/id_ed25519 root@<IP> -p <PORT> "tmux kill-session -t train"

# Option B — stop Pod immediately from local machine
runpodctl stop pod $POD_ID
```

Both options leave all checkpoints intact on the network volume.

---

## Billing Reference

| Pod State | What's Billed |
|-----------|--------------|
| Running (training) | GPU + container disk (per second) |
| Stopped | Network volume only (`$0.07/GB/month`) |
| Terminated | Nothing |

Typical cost for `whisper-small` on RTX 4090 spot (~$0.20/hr): **~$0.80 per full training run** (~4 hrs).
Network volume: ~$7/month (100 GB).
