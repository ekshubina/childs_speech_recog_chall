#!/usr/bin/env python3
"""
pod_storage.py — manage RunPod network volume storage via S3 API.

Replaces pod_get_results.sh and pod_clean_storage.sh.

Usage:
  python scripts/pod_storage.py list
  python scripts/pod_storage.py get [--logs-only | --checkpoints-only] [--run NAME]
  python scripts/pod_storage.py clean --old-checkpoints [--run NAME] [--yes]
  python scripts/pod_storage.py clean --all-checkpoints [--run NAME] [--yes]
  python scripts/pod_storage.py clean --hf-cache [--yes]
  python scripts/pod_storage.py clean --pycache [--yes]
  python scripts/pod_storage.py clean --logs [--yes]

Requires: .runpod.env with RUNPOD_S3_BUCKET, RUNPOD_S3_ENDPOINT,
          RUNPOD_S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

── Why not aws s3 / aws s3api list-objects-v2? ──────────────────────────────
RunPod's S3 returns IsTruncated=True with a bogus NextContinuationToken even
when there is only one page of results, and repeated calls with that token
return the same token again.  The AWS CLI and its Paginator follow this token
blindly and hang or crash.

Fix: NEVER use recursive listing (no Prefix-only list_objects_v2 loops).
Instead walk the tree level by level with Delimiter="/" so each API call sees
only direct children (< 1000 for our structure). Single-page, no pagination.
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import threading
from pathlib import Path

# ── Locate repo root and load .runpod.env ────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".runpod.env"


def load_env():
    if not ENV_FILE.exists():
        sys.exit(f"Error: {ENV_FILE} not found.")
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            # Strip inline comments (e.g. "eu-ro-1  # adjust region")
            val = val.split("#")[0].strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), val)

    missing = [
        v for v in [
            "RUNPOD_S3_BUCKET", "RUNPOD_S3_ENDPOINT", "RUNPOD_S3_REGION",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
        ]
        if not os.environ.get(v)
    ]
    if missing:
        sys.exit(
            f"Error: missing in {ENV_FILE}: {', '.join(missing)}\n"
            "Get credentials: RunPod console → Network Volumes → your volume → S3 API Access"
        )


# ── S3 client ─────────────────────────────────────────────────────────────────

def make_client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        sys.exit(
            "Error: boto3 not installed.\n"
            "Run: source .venv/bin/activate && pip install -r requirements-dev.txt"
        )
    return boto3.client(
        "s3",
        region_name=os.environ["RUNPOD_S3_REGION"],
        endpoint_url=os.environ["RUNPOD_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(
            connect_timeout=10,
            read_timeout=120,
            retries={"max_attempts": 1},
            s3={"addressing_style": "path"},  # RunPod S3 requires path-style; virtual-hosted → 307
        ),
    )


# ── Core S3 helpers (shallow only — Delimiter="/" at each level) ──────────────

def _bucket() -> str:
    return os.environ["RUNPOD_S3_BUCKET"]


def list_subdirs(s3, prefix: str) -> list[str]:
    """
    Return immediate sub-prefixes (virtual dirs) under prefix/.
    Delimiter="/" means one level only — never paginates for our structure.
    """
    try:
        resp = s3.list_objects_v2(
            Bucket=_bucket(),
            Prefix=prefix.rstrip("/") + "/",
            Delimiter="/",
            MaxKeys=1000,
        )
        return [
            cp["Prefix"].rstrip("/").split("/")[-1]
            for cp in resp.get("CommonPrefixes", [])
        ]
    except Exception as e:
        print(f"  Warning: list_subdirs('{prefix}') failed: {e}", file=sys.stderr)
        return []


def list_files_shallow(s3, prefix: str) -> list[tuple[str, int]]:
    """
    Return (key, size) for files directly inside prefix/ (non-recursive).
    Delimiter="/" → at most ~20 direct files per checkpoint dir; no pagination.
    Skips directory-marker objects (key ends with '/').
    """
    try:
        resp = s3.list_objects_v2(
            Bucket=_bucket(),
            Prefix=prefix.rstrip("/") + "/",
            Delimiter="/",
            MaxKeys=1000,
        )
        return [
            (obj["Key"], obj["Size"])
            for obj in resp.get("Contents", [])
            if not obj["Key"].endswith("/")
        ]
    except Exception as e:
        print(f"  Warning: list_files_shallow('{prefix}') failed: {e}", file=sys.stderr)
        return []


def walk_keys(s3, prefix: str, include_dir_markers: bool = False) -> list[tuple[str, int]]:
    """
    Recursively collect all (key, size) pairs under prefix/ by walking the
    tree with shallow listings only.  Never triggers pagination.

    include_dir_markers: also append the virtual-directory marker objects
    (keys ending in '/') that S3 stores to represent empty directories.
    These appear as CommonPrefixes, not Contents, so list_files_shallow misses
    them.  Pass True when building a deletion plan so markers are removed too.
    delete_object on a non-existent key is a safe no-op on S3-compatible stores.
    """
    results: list[tuple[str, int]] = []
    results.extend(list_files_shallow(s3, prefix))
    for sub in list_subdirs(s3, prefix):
        sub_prefix = f"{prefix.rstrip('/')}/{sub}"
        results.extend(walk_keys(s3, sub_prefix, include_dir_markers=include_dir_markers))
        if include_dir_markers:
            results.append((sub_prefix + "/", 0))
    return results


def key_exists(s3, key: str) -> int | None:
    """Return file size if key exists in the bucket, else None."""
    try:
        resp = s3.head_object(Bucket=_bucket(), Key=key)
        return resp["ContentLength"]
    except Exception:
        return None


# ── Batch delete ──────────────────────────────────────────────────────────────

def delete_keys(s3, keys: list[str], label: str = "") -> int:
    """
    Delete keys one at a time via delete_object (singular DELETE requests).

    RunPod's S3 returns 307 Temporary Redirect on DeleteObjects (the batch
    POST endpoint), so we avoid it entirely and fall back to individual calls.
    This is slightly slower but universally compatible with S3-compatible stores.
    """
    total = 0
    for key in keys:
        try:
            s3.delete_object(Bucket=_bucket(), Key=key)
            total += 1
        except Exception as e:
            print(f"  Delete error: {key}: {e}", file=sys.stderr)
    tag = f" from {label}" if label else ""
    print(f"  Deleted {total} objects{tag}.")
    return total


# ── Formatting ────────────────────────────────────────────────────────────────

def human_size(b: int) -> str:
    for unit, threshold in [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if b >= threshold:
            return f"{b / threshold:.1f} {unit}"
    return f"{b} B"


# ══════════════════════════════════════════════════════════════════════════════
# list
# ══════════════════════════════════════════════════════════════════════════════

def cmd_list(s3, _args) -> None:
    print(f"\nBucket:   s3://{_bucket()}")
    print(f"Endpoint: {os.environ['RUNPOD_S3_ENDPOINT']}\n")
    print(f"{'Prefix':<58}  {'Size':>10}  {'Files':>7}")
    print("─" * 80)

    # data/audio_* hold 30 K+ flat FLAC files — listing times out on RunPod S3.
    # They are immutable training data; just show a static note.
    print(f"{'data/audio_{0,1,2}  (training data — not listed)':<58}  {'~95K files':>10}")

    managed_prefixes = [
        ("childs_speech_recog_chall/checkpoints", "checkpoints"),
        (".cache/huggingface", ".cache/huggingface (HF models)"),
        ("logs", "logs"),
        ("childs_speech_recog_chall/.git", "repo .git"),
    ]

    results: dict[str, tuple[int, int]] = {}

    def fetch(prefix: str, _label: str) -> None:
        objs = walk_keys(s3, prefix)
        results[prefix] = (sum(sz for _, sz in objs), len(objs))

    threads = [threading.Thread(target=fetch, args=(p, l)) for p, l in managed_prefixes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for prefix, label in managed_prefixes:
        b, n = results.get(prefix, (0, 0))
        if n > 0:
            print(f"{label:<58}  {human_size(b):>10}  {n:>7}")

    print("\nCheckpoints detail:")
    runs = list_subdirs(s3, "childs_speech_recog_chall/checkpoints")
    if not runs:
        print("  (none)")
    for run in runs:
        subs = list_subdirs(s3, f"childs_speech_recog_chall/checkpoints/{run}")
        for sub in subs:
            objs = walk_keys(s3, f"childs_speech_recog_chall/checkpoints/{run}/{sub}")
            b = sum(sz for _, sz in objs)
            print(f"  {run}/{sub:<52}  {human_size(b):>10}  {len(objs):>7}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# get
# ══════════════════════════════════════════════════════════════════════════════

def cmd_get(s3, args) -> None:
    print(f"\nBucket:   s3://{_bucket()}")
    print(f"Endpoint: {os.environ['RUNPOD_S3_ENDPOINT']}\n")

    do_checkpoints = not args.logs_only
    do_logs = not args.checkpoints_only

    # ── Checkpoints ──────────────────────────────────────────────────────────
    if do_checkpoints:
        ckpt_root = "childs_speech_recog_chall/checkpoints"
        runs = [args.run] if args.run else list_subdirs(s3, ckpt_root)
        if not runs:
            print("No checkpoint runs found in the network volume.")

        for run in runs:
            local_run_dir = REPO_ROOT / "checkpoints" / run
            print(f">>> Downloading checkpoints/{run}/ ...")
            objs = walk_keys(s3, f"{ckpt_root}/{run}")
            if not objs:
                print("  (empty — nothing to download)")
                continue
            count = 0
            strip = len(f"{ckpt_root}/{run}/")
            for key, _ in objs:
                rel = key[strip:]
                dest = local_run_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(_bucket(), key, str(dest))
                count += 1
                print(f"\r  {count}/{len(objs)} files", end="", flush=True)
            print(f"\r  {count} files → {local_run_dir}")

    # ── Logs ─────────────────────────────────────────────────────────────────
    if do_logs:
        print(">>> Downloading logs/ ...")
        local_log_dir = REPO_ROOT / "logs"
        local_log_dir.mkdir(parents=True, exist_ok=True)

        # walk_keys uses shallow Delimiter listing — works fine for a small logs/ dir.
        # Fallback to head_object for known keys if listing returns nothing.
        objs = walk_keys(s3, "logs")
        if not objs:
            for candidate in ["logs/current.log", "logs/run_train.sh"]:
                size = key_exists(s3, candidate)
                if size is not None:
                    objs.append((candidate, size))

        if not objs:
            print("  No log files found.")
        else:
            count = 0
            for key, size in objs:
                filename = key[len("logs/"):]
                if not filename:
                    continue
                dest = local_log_dir / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"  {key}  ({human_size(size)})", end=" ... ", flush=True)
                s3.download_file(_bucket(), key, str(dest))
                print("done")
                count += 1
            print(f"  {count} file(s) → {local_log_dir}")

    print("\nDone.")


# ══════════════════════════════════════════════════════════════════════════════
# clean
# ══════════════════════════════════════════════════════════════════════════════

def cmd_clean(s3, args) -> None:
    # Plan: collect all (keys, label) BEFORE any confirmation or deletion.
    plan: list[tuple[list[str], str]] = []

    # ── Checkpoints ──────────────────────────────────────────────────────────
    if args.old_checkpoints or args.all_checkpoints:
        print("\n── Checkpoints ──────────────────────────────────────────────────")
        ckpt_root = "childs_speech_recog_chall/checkpoints"
        runs = [args.run] if args.run else list_subdirs(s3, ckpt_root)
        if not runs:
            print("  No checkpoint runs found.")

        for run in runs:
            print(f"\n  Run: {run}")
            subs = list_subdirs(s3, f"{ckpt_root}/{run}")

            if args.all_checkpoints:
                objs = walk_keys(s3, f"{ckpt_root}/{run}", include_dir_markers=True)
                total_b = sum(sz for _, sz in objs)
                keys = [k for k, _ in objs]
                # Also include the run-level marker itself
                keys.append(f"{ckpt_root}/{run}/")
                real_count = sum(1 for k in keys if not k.endswith("/"))
                print(f"  Will delete: {run}/  ({human_size(total_b)}, {real_count} files + markers)")
                if keys:
                    plan.append((keys, f"{run}/"))
            else:
                ckpt_subs = sorted(
                    [s for s in subs if s.startswith("checkpoint-")],
                    key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0,
                )
                latest = ckpt_subs[-1] if ckpt_subs else None

                for sub in subs:
                    if sub in ("final_model", "runs"):
                        print(f"  Keeping:  {run}/{sub}/  (protected)")
                        continue
                    if sub == latest:
                        print(f"  Keeping:  {run}/{sub}/  (latest checkpoint, used for resume)")
                        continue
                    objs = walk_keys(s3, f"{ckpt_root}/{run}/{sub}", include_dir_markers=True)
                    total_b = sum(sz for _, sz in objs)
                    keys = [k for k, _ in objs]
                    keys.append(f"{ckpt_root}/{run}/{sub}/")  # sub-level marker
                    real_count = sum(1 for k in keys if not k.endswith("/"))
                    print(f"  Will delete: {run}/{sub}/  ({human_size(total_b)}, {real_count} files + markers)")
                    if keys:
                        plan.append((keys, f"{run}/{sub}/"))

    # ── HuggingFace cache ─────────────────────────────────────────────────────
    if args.hf_cache:
        print("\n── HuggingFace cache ────────────────────────────────────────────")
        objs = walk_keys(s3, ".cache/huggingface", include_dir_markers=True)
        if not objs:
            print("  .cache/huggingface/ is empty — nothing to delete.")
        else:
            total_b = sum(sz for _, sz in objs)
            keys = [k for k, _ in objs]
            print(f"  Will delete: .cache/huggingface/  ({human_size(total_b)}, {len(keys)} objects)")
            print("  Models will be re-downloaded from HuggingFace Hub on next run.")
            plan.append((keys, ".cache/huggingface/"))

    # ── Python cache ──────────────────────────────────────────────────────────
    if args.pycache:
        print("\n── Python cache ─────────────────────────────────────────────────")
        objs = walk_keys(s3, "childs_speech_recog_chall", include_dir_markers=False)
        py_keys = [k for k, _ in objs if "__pycache__" in k or k.endswith(".pyc")]
        if not py_keys:
            print("  No Python cache files found.")
        else:
            print(f"  Will delete: {len(py_keys)} .pyc / __pycache__ objects")
            plan.append((py_keys, "Python cache"))

    # ── Logs ─────────────────────────────────────────────────────────────────
    if args.logs:
        print("\n── Logs ─────────────────────────────────────────────────────────")
        objs = walk_keys(s3, "logs")
        log_keys = [k for k, _ in objs if k != "logs/current.log"]
        if not log_keys:
            print("  Only logs/current.log found — nothing else to delete.")
        else:
            print(f"  Will delete: {len(log_keys)} object(s)  (keeping logs/current.log)")
            for k in log_keys:
                print(f"    {k}")
            plan.append((log_keys, "logs (excl. current.log)"))

    # ── Nothing to do ─────────────────────────────────────────────────────────
    if not plan:
        print("\nNothing to delete.")
        return

    # ── Dry-run ───────────────────────────────────────────────────────────────
    if not args.yes:
        print("\nDry-run complete — nothing deleted. Re-run with --yes to execute.")
        return

    # ── Summary + confirm ─────────────────────────────────────────────────────
    print("\n┌─ Deletion summary " + "─" * 59)
    for keys, label in plan:
        print(f"│  {label:<60}  {len(keys):>5} objects")
    print("└" + "─" * 78)
    print()
    answer = input("Permanently delete all of the above from S3? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted — nothing deleted.")
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    print()
    for keys, label in plan:
        print(f"  Deleting {label} ...")
        delete_keys(s3, keys, label)

    print("\nAll deletions complete.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pod_storage.py",
        description="Manage RunPod network volume storage via S3 API (no Pod required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show storage usage by prefix")

    get_p = sub.add_parser("get", help="Download results from the network volume")
    get_g = get_p.add_mutually_exclusive_group()
    get_g.add_argument("--logs-only",        action="store_true", help="Download only logs")
    get_g.add_argument("--checkpoints-only", action="store_true", help="Download only checkpoints")
    get_p.add_argument("--run", metavar="NAME", help="Download a specific run only")

    clean_p = sub.add_parser("clean", help="Delete obsolete data from the network volume")
    what = clean_p.add_mutually_exclusive_group(required=True)
    what.add_argument("--old-checkpoints", action="store_true",
                      help="Delete intermediate checkpoint-N/ dirs; keep final_model + latest")
    what.add_argument("--all-checkpoints", action="store_true",
                      help="Delete entire checkpoint run(s)")
    what.add_argument("--hf-cache",        action="store_true",
                      help="Delete .cache/huggingface/  (~15 GB; re-downloaded automatically)")
    what.add_argument("--pycache",         action="store_true",
                      help="Delete __pycache__ and .pyc files in the repo clone")
    what.add_argument("--logs",            action="store_true",
                      help="Delete log files (keeps logs/current.log)")
    clean_p.add_argument("--run", metavar="NAME",
                         help="Scope --old-checkpoints / --all-checkpoints to one run")
    clean_p.add_argument("--yes", action="store_true",
                         help="Execute deletions (still confirms); without --yes = safe dry-run")

    args = parser.parse_args()
    load_env()
    s3 = make_client()

    if args.command == "list":
        cmd_list(s3, args)
    elif args.command == "get":
        cmd_get(s3, args)
    elif args.command == "clean":
        cmd_clean(s3, args)


if __name__ == "__main__":
    main()
