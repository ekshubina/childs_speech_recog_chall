#!/usr/bin/env python3
"""
Training script for fine-tuning Whisper on children's speech data.

This script loads configuration, prepares datasets, initializes the model,
and executes the training loop with validation and checkpointing.

Usage:
    python scripts/train.py --config configs/baseline_whisper_medium.yaml
    python scripts/train.py --config configs/baseline_whisper_medium.yaml \
        --resume checkpoints/whisper-finetuned/checkpoint-5000
    python scripts/train.py --config configs/baseline_whisper_medium.yaml \
        --debug  # Train on small subset
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Suppress noisy progress bars from transformers/accelerate during model loading
import transformers

transformers.utils.logging.set_verbosity_info()
logging.getLogger("accelerate").setLevel(logging.INFO)

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import WhisperProcessor

from src.data.dataset import ChildSpeechDataset, create_train_val_split
from src.models.model_factory import ModelFactory
from src.models.whisper_model import prepare_model_for_finetuning
from src.training.trainer import WhisperTrainer
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger


def get_git_branch() -> str:
    """Return the current git branch name, sanitized for use in a directory path.

    Replaces '/' with '-' so that branch names like 'feature/foo' become 'feature-foo'.
    Falls back to 'unknown-branch' if git is unavailable or the repo has no commits.
    """
    import subprocess

    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent.parent,
            )
            .decode()
            .strip()
        )
        # Sanitize: replace path separators with dashes
        return branch.replace("/", "-") if branch else "unknown-branch"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown-branch"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model on children's speech data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with small subset of data")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file (default: logs/train_{timestamp}.log)")

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_file = args.log_file
    if log_file is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/train_{timestamp}.log"

    logger = setup_logger("train", log_file)
    logger.info("=" * 80)
    logger.info("Starting Whisper Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Validation ratio: {args.val_ratio}")
    logger.info(f"Random seed: {args.seed}")

    # Set random seed for reproducibility
    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)

        # Append git branch name to output / logging directories so that checkpoints
        # from different branches never collide and are always traceable to their origin.
        git_branch = get_git_branch()
        logger.info(f"Git branch: {git_branch}")
        base_output_dir = config["training"].get("output_dir", "checkpoints/whisper-finetuned")
        config["training"]["output_dir"] = f"{base_output_dir}_{git_branch}"
        if "logging_dir" in config["training"]:
            config["training"]["logging_dir"] = f"{config['training']['logging_dir']}_{git_branch}"
        logger.info(f"Checkpoint directory: {config['training']['output_dir']}")

        logger.info(f"Model: {config['model']['name']} - {config['model']['variant']}")
        logger.info(f"Training epochs: {config['training']['num_epochs']}")
        logger.info(f"Batch size: {config['training']['batch_size']}")
        logger.info(f"Learning rate: {config['training']['learning_rate']}")

        # Check device
        if torch.cuda.is_available():
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon GPU)")
            logger.info("Batch size will be reduced and gradient checkpointing enabled to fit MPS memory.")
        else:
            logger.info("Using CPU (training will be slow)")

        # Load processor first (needed for dataset)
        logger.info("Loading Whisper processor...")
        processor = WhisperProcessor.from_pretrained(
            config["model"].get("pretrained", f"openai/whisper-{config['model']['variant']}"),
            language="english",
            task="transcribe",
        )
        logger.info("Processor loaded successfully")

        # Create train/val split
        logger.info("Creating train/validation split...")
        train_manifest_path = config["data"]["train_manifest"]

        if not Path(train_manifest_path).exists():
            logger.error(f"Training manifest not found: {train_manifest_path}")
            sys.exit(1)

        train_manifest, val_manifest = create_train_val_split(
            train_manifest_path, val_ratio=args.val_ratio, random_seed=args.seed, stratify_by="age_bucket"
        )

        logger.info(f"Training samples: {len(train_manifest):,}")
        logger.info(f"Validation samples: {len(val_manifest):,}")

        # Debug mode: use small subset
        if args.debug:
            logger.warning("DEBUG MODE: Using only 100 training and 20 validation samples")
            train_manifest = train_manifest[:100]
            val_manifest = val_manifest[:20]

        # Create datasets
        logger.info("Creating datasets...")
        audio_dirs = config["data"]["audio_dirs"]

        train_dataset = ChildSpeechDataset(
            samples=train_manifest, audio_dirs=audio_dirs, processor=processor, sample_rate=16000
        )

        val_dataset = ChildSpeechDataset(samples=val_manifest, audio_dirs=audio_dirs, processor=processor, sample_rate=16000)

        logger.info(f"Training dataset size: {len(train_dataset):,}")
        logger.info(f"Validation dataset size: {len(val_dataset):,}")

        # Load model
        logger.info("Creating model...")
        model = ModelFactory.create_model(config)

        # Load pretrained weights
        pretrained_path = config["model"].get("pretrained", f"openai/whisper-{config['model']['variant']}")
        logger.info(f"Loading pretrained weights from {pretrained_path}...")
        model.load(pretrained_path)
        logger.info("Model loaded successfully")

        # Prepare model for fine-tuning
        logger.info("Preparing model for fine-tuning...")
        model_obj = model.model  # Get the actual model from WhisperModel wrapper

        freeze_encoder = config["model"].get("freeze_encoder", False)
        dropout = config["model"].get("dropout", 0.1)

        model_obj = prepare_model_for_finetuning(
            model_obj, freeze_encoder=freeze_encoder, dropout=dropout, language="english", task="transcribe"
        )

        logger.info(f"Model prepared (freeze_encoder={freeze_encoder}, dropout={dropout})")

        # Get model info
        model_info = model.get_model_info()
        logger.info(f"Total parameters: {model_info['parameters']:,}")
        logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")

        # Create trainer
        logger.info("Creating trainer...")
        trainer = WhisperTrainer.create_trainer(
            model=model_obj,
            processor=processor,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            config=config,
            resume_from_checkpoint=args.resume,
        )

        logger.info("Trainer created successfully")

        # Train
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)

        # Re-enable tqdm progress bar — it was suppressed during model loading to avoid
        # noisy loading bars, but we want it back for the training loop.
        transformers.utils.logging.enable_progress_bar()

        # Reset peak VRAM counters so the summary reflects training only (not model load)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_result = trainer.train(resume_from_checkpoint=args.resume)

        # Log training results
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        logger.info(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
        logger.info(f"Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")

        # Save final model
        final_model_path = Path(config["training"]["output_dir"]) / "final_model"
        logger.info(f"Saving final model to: {final_model_path}")

        model_obj.save_pretrained(str(final_model_path))
        processor.save_pretrained(str(final_model_path))

        logger.info("Final model saved successfully")

        # Evaluate on validation set
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()

        logger.info("=" * 80)
        logger.info("Final Evaluation Results")
        logger.info("=" * 80)
        for key, value in eval_results.items():
            logger.info(f"{key}: {value}")

        logger.info("=" * 80)
        logger.info("Training script completed successfully!")
        logger.info("=" * 80)

        # ── System Resource Summary (for hyperparameter tuning) ────────────────
        logger.info("=" * 80)
        logger.info("System Resource Summary")
        logger.info("=" * 80)

        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated(0)
            peak_reserved = torch.cuda.max_memory_reserved(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU:                    {torch.cuda.get_device_name(0)}")
            logger.info(f"  Total VRAM:           {total_mem / 1e9:.2f} GB")
            logger.info(f"  Peak allocated:       {peak_alloc / 1e9:.2f} GB  ({100 * peak_alloc / total_mem:.1f}% of VRAM)")
            logger.info(
                f"  Peak reserved:        {peak_reserved / 1e9:.2f} GB  ({100 * peak_reserved / total_mem:.1f}% of VRAM)"
            )
            logger.info(f"  Free headroom:        {(total_mem - peak_reserved) / 1e9:.2f} GB")
            if (total_mem - peak_reserved) / 1e9 > 2.0:
                logger.info("  Tip: >2 GB free — consider increasing batch_size or reducing gradient_accumulation_steps")
            elif peak_reserved / total_mem > 0.95:
                logger.info("  Tip: VRAM >95% used — consider enabling gradient_checkpointing or reducing batch_size")
        else:
            logger.info("GPU: not available (CPU training)")

        t_metrics = train_result.metrics
        logger.info(f"Throughput:             {t_metrics.get('train_samples_per_second', 0):.2f} samples/sec")
        total_secs = t_metrics.get("train_runtime", 0)
        logger.info(f"Total training time:    {total_secs:.0f} s  ({total_secs / 60:.1f} min)")
        total_steps = int(round(t_metrics.get("train_steps_per_second", 0) * total_secs))
        if total_steps:
            logger.info(f"Total steps:            {total_steps:,}")

        per_device_bs = config["training"]["batch_size"]
        grad_accum = config["training"].get("gradient_accumulation_steps", 1)
        effective_bs = per_device_bs * grad_accum
        logger.info(f"Effective batch size:   {effective_bs}  (per_device={per_device_bs} × grad_accum={grad_accum})")
        logger.info(f"Learning rate:          {config['training'].get('learning_rate', 'N/A')}")
        logger.info(f"FP16:                   {config['training'].get('fp16', False)}")
        logger.info(f"Gradient checkpointing: {config['model'].get('gradient_checkpointing', False)}")
        logger.info(f"Freeze encoder:         {config['model'].get('freeze_encoder', False)}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("Training failed with error:")
        logger.error("=" * 80)
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
