"""
Training infrastructure for Whisper fine-tuning.

This module provides a wrapper around Hugging Face's Seq2SeqTrainer
with configurations specific to Whisper fine-tuning on children's speech data.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor
from transformers.trainer_utils import get_last_checkpoint

from src.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class WhisperTrainingConfig:
    """
    Configuration for Whisper fine-tuning.

    Encapsulates all training hyperparameters and settings.

    Attributes:
        output_dir: Directory to save checkpoints and logs
        num_epochs: Number of training epochs
        batch_size: Training batch size per device
        eval_batch_size: Evaluation batch size per device
        learning_rate: Initial learning rate
        warmup_steps: Number of warmup steps for learning rate scheduler
        gradient_accumulation_steps: Number of steps to accumulate gradients
        fp16: Whether to use mixed precision training (requires CUDA)
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log metrics every N steps
        save_total_limit: Maximum number of checkpoints to keep
        load_best_model_at_end: Whether to load best model after training
        metric_for_best_model: Metric to use for selecting best model
        greater_is_better: Whether higher metric value is better
        generation_max_length: Maximum length for generated sequences
        predict_with_generate: Whether to use generation for evaluation
    """

    output_dir: str = "checkpoints/whisper-finetuned"
    num_epochs: int = 10
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    gradient_checkpointing: bool = False
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False  # Lower WER is better
    generation_max_length: int = 225
    predict_with_generate: bool = True


class WhisperTrainer(Seq2SeqTrainer):
    """
    Custom trainer for Whisper fine-tuning.

    Extends Hugging Face's Seq2SeqTrainer with Whisper-specific
    configurations and logging.

    Attributes:
        model: Whisper model to train
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Whisper tokenizer/processor
        data_collator: Data collator for batching
        compute_metrics: Function to compute evaluation metrics

    Example:
        >>> from transformers import WhisperForConditionalGeneration, WhisperProcessor
        >>> from src.data.dataset import ChildSpeechDataset
        >>>
        >>> model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')
        >>> processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
        >>> train_ds = ChildSpeechDataset('train.jsonl', processor)
        >>> eval_ds = ChildSpeechDataset('val.jsonl', processor)
        >>>
        >>> trainer = WhisperTrainer.create_trainer(
        ...     model=model,
        ...     processor=processor,
        ...     train_dataset=train_ds,
        ...     eval_dataset=eval_ds,
        ...     config={'training': {'num_epochs': 5, 'batch_size': 16}}
        ... )
        >>> trainer.train()
    """

    @staticmethod
    def create_trainer(
        model,
        processor: WhisperProcessor,
        train_dataset,
        eval_dataset,
        config: Dict[str, Any],
        data_collator=None,
        resume_from_checkpoint: Optional[Union[str, Path]] = None,
    ) -> "WhisperTrainer":
        """
        Factory method to create configured WhisperTrainer.

        Args:
            model: Whisper model to train
            processor: WhisperProcessor for tokenization
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            config: Configuration dictionary with 'training' section
            data_collator: Optional custom data collator
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Configured WhisperTrainer instance

        Raises:
            ValueError: If config is missing required fields
        """
        if "training" not in config:
            raise ValueError("Config must contain 'training' section")

        training_config = config["training"]
        model_config = config.get("model", {})

        # Detect device and adjust settings accordingly
        use_cuda = torch.cuda.is_available()
        use_mps = torch.backends.mps.is_available() and not use_cuda
        use_cpu = not use_cuda and not use_mps

        original_batch_size = training_config.get("batch_size", 8)
        batch_size = original_batch_size

        if use_mps and original_batch_size > 2:
            batch_size = 2  # MPS has limited memory - keep batch small, rely on grad accumulation
            logger.warning(
                f"MPS device detected. Reducing batch_size from {original_batch_size} to {batch_size} "
                f"to fit within MPS memory limits. Gradient accumulation compensates for smaller batch."
            )
        elif use_cpu and original_batch_size > 4:
            batch_size = 4  # Reduce batch size for CPU training
            logger.warning(
                f"CPU training detected. Reducing batch_size from {original_batch_size} to {batch_size} "
                f"to accommodate CPU memory constraints."
            )

        # Read gradient_checkpointing from model config (saves ~40% activation memory)
        gradient_checkpointing = model_config.get("gradient_checkpointing", use_mps)

        # Create training config
        train_cfg = WhisperTrainingConfig(
            output_dir=training_config.get("output_dir", "checkpoints/whisper-finetuned"),
            num_epochs=training_config.get("num_epochs", 10),
            batch_size=batch_size,
            eval_batch_size=training_config.get("eval_batch_size", 8),
            learning_rate=training_config.get("learning_rate", 1e-5),
            warmup_steps=training_config.get("warmup_steps", 500),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            fp16=training_config.get("fp16", True) and use_cuda,  # Only enable fp16 with CUDA (MPS uses bfloat16/fp32)
            gradient_checkpointing=gradient_checkpointing,
            save_steps=training_config.get("save_steps", 1000),
            eval_steps=training_config.get("eval_steps", 1000),
            logging_steps=training_config.get("logging_steps", 100),
            save_total_limit=training_config.get("save_total_limit", 3),
        )

        # Log warning if fp16 requested but not on CUDA
        if training_config.get("fp16", False) and not use_cuda:
            if use_mps:
                logger.info("fp16=true in config but running on MPS - using fp32 instead.")
            else:
                logger.warning("fp16=true in config but CUDA not available. Disabling fp16. Training will use fp32.")

        if gradient_checkpointing:
            logger.info("Gradient checkpointing enabled (reduces memory ~40%, slightly slower)")

        # Create output directory
        output_dir = Path(train_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Training configuration:")
        logger.info(f"  Output dir: {train_cfg.output_dir}")
        logger.info(f"  Num epochs: {train_cfg.num_epochs}")
        logger.info(f"  Batch size: {train_cfg.batch_size}")
        logger.info(f"  Gradient accumulation: {train_cfg.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {train_cfg.batch_size * train_cfg.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {train_cfg.learning_rate}")
        logger.info(f"  FP16: {train_cfg.fp16}")
        logger.info(f"  Gradient checkpointing: {gradient_checkpointing}")
        logger.info(f"  Device: {'CUDA' if use_cuda else 'MPS' if use_mps else 'CPU'}")

        # Data loading settings — critical for GPU utilisation
        # Default HuggingFace Trainer uses 0 workers (serial), which starves the GPU
        dataloader_num_workers = training_config.get("dataloader_num_workers", 4)
        dataloader_pin_memory = training_config.get("dataloader_pin_memory", True) and use_cuda
        # prefetch_factor requires num_workers > 0
        dataloader_prefetch_factor = (
            training_config.get("dataloader_prefetch_factor", 4) if dataloader_num_workers > 0 else None
        )

        logger.info(f"  Dataloader workers: {dataloader_num_workers}")
        logger.info(f"  Dataloader pin_memory: {dataloader_pin_memory}")
        logger.info(f"  Dataloader prefetch_factor: {dataloader_prefetch_factor}")

        # Create training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=train_cfg.output_dir,
            num_train_epochs=train_cfg.num_epochs,
            per_device_train_batch_size=train_cfg.batch_size,
            per_device_eval_batch_size=train_cfg.eval_batch_size,
            learning_rate=train_cfg.learning_rate,
            warmup_steps=train_cfg.warmup_steps,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            fp16=train_cfg.fp16,
            gradient_checkpointing=train_cfg.gradient_checkpointing,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=train_cfg.save_steps,
            eval_steps=train_cfg.eval_steps,
            logging_steps=train_cfg.logging_steps,
            logging_first_step=True,
            save_total_limit=train_cfg.save_total_limit,
            load_best_model_at_end=train_cfg.load_best_model_at_end,
            metric_for_best_model=train_cfg.metric_for_best_model,
            greater_is_better=train_cfg.greater_is_better,
            generation_max_length=train_cfg.generation_max_length,
            predict_with_generate=train_cfg.predict_with_generate,
            remove_unused_columns=False,  # Keep all columns for custom processing
            label_names=["labels"],  # Specify label column name
            push_to_hub=False,
            report_to=["tensorboard"],
            # Data loading — parallel workers eliminate serial CPU bottleneck
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=dataloader_pin_memory,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
        )

        # Use default data collator if not provided
        if data_collator is None:
            from src.data.dataset import WhisperDataCollator

            data_collator = WhisperDataCollator(
                processor=processor,
                padding="longest",
            )

        # Create compute_metrics wrapper with tokenizer
        def compute_metrics_with_tokenizer(pred):
            return compute_metrics(pred, tokenizer=processor.tokenizer)

        # Handle checkpoint resumption
        checkpoint = None
        if resume_from_checkpoint is not None:
            checkpoint = str(resume_from_checkpoint)
            logger.info(f"Resuming from checkpoint: {checkpoint}")
        else:
            # Check for last checkpoint in output_dir
            last_checkpoint = get_last_checkpoint(train_cfg.output_dir)
            if last_checkpoint is not None:
                logger.info(f"Found existing checkpoint: {last_checkpoint}")
                logger.info("Use resume_from_checkpoint parameter to resume training")

        # Create trainer
        trainer = WhisperTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_with_tokenizer,
        )

        logger.info("WhisperTrainer created successfully")
        logger.info(f"Training samples: {len(train_dataset):,}")
        logger.info(f"Validation samples: {len(eval_dataset):,}")

        return trainer

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log method to add custom logging.

        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for computing elapsed time (added in newer transformers)
        """
        # Add custom logging here if needed
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

        # Log WER prominently if present
        if "eval_wer" in logs:
            logger.info(f">>> Validation WER: {logs['eval_wer']:.4f}")

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Override checkpoint saving to add custom logging.

        Args:
            model: Model to save
            trial: Trial object (for hyperparameter search)
            metrics: Metrics to save with checkpoint
        """
        # Call parent without metrics argument for compatibility
        checkpoint_folder = super()._save_checkpoint(model, trial)

        if checkpoint_folder is not None:
            logger.info(f"Checkpoint saved to: {checkpoint_folder}")
            if metrics is not None and "eval_wer" in metrics:
                logger.info(f"Checkpoint WER: {metrics['eval_wer']:.4f}")

        return checkpoint_folder
