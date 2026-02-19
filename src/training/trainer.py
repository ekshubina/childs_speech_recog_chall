"""
Training infrastructure for Whisper fine-tuning.

This module provides a wrapper around Hugging Face's Seq2SeqTrainer
with configurations specific to Whisper fine-tuning on children's speech data.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperTokenizer,
    WhisperProcessor
)
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
    output_dir: str = 'checkpoints/whisper-finetuned'
    num_epochs: int = 10
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = 'wer'
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
        resume_from_checkpoint: Optional[Union[str, Path]] = None
    ) -> 'WhisperTrainer':
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
        if 'training' not in config:
            raise ValueError("Config must contain 'training' section")
        
        training_config = config['training']
        
        # Detect if we're on CPU and adjust batch size if needed
        is_cpu = not torch.cuda.is_available()
        original_batch_size = training_config.get('batch_size', 8)
        batch_size = original_batch_size
        
        if is_cpu and original_batch_size > 4:
            batch_size = 4  # Reduce batch size for CPU training
            logger.warning(
                f"CPU training detected. Reducing batch_size from {original_batch_size} to {batch_size} "
                f"to accommodate CPU memory constraints."
            )
        
        # Create training config
        train_cfg = WhisperTrainingConfig(
            output_dir=training_config.get('output_dir', 'checkpoints/whisper-finetuned'),
            num_epochs=training_config.get('num_epochs', 10),
            batch_size=batch_size,
            eval_batch_size=training_config.get('eval_batch_size', 8),
            learning_rate=training_config.get('learning_rate', 1e-5),
            warmup_steps=training_config.get('warmup_steps', 500),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
            fp16=training_config.get('fp16', True) and torch.cuda.is_available(),  # Only enable fp16 with CUDA
            save_steps=training_config.get('save_steps', 1000),
            eval_steps=training_config.get('eval_steps', 1000),
            logging_steps=training_config.get('logging_steps', 100),
            save_total_limit=training_config.get('save_total_limit', 3),
        )
        
        # Log warning if fp16 requested but CUDA not available
        if training_config.get('fp16', False) and not torch.cuda.is_available():
            logger.warning(
                "fp16=true in config but CUDA not available. "
                "Disabling fp16 (only supported on CUDA). Training will use fp32."
            )
        
        # Create output directory
        output_dir = Path(train_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training configuration:")
        logger.info(f"  Output dir: {train_cfg.output_dir}")
        logger.info(f"  Num epochs: {train_cfg.num_epochs}")
        logger.info(f"  Batch size: {train_cfg.batch_size}")
        logger.info(f"  Gradient accumulation: {train_cfg.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {train_cfg.batch_size * train_cfg.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {train_cfg.learning_rate}")
        logger.info(f"  FP16: {train_cfg.fp16}")
        
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
            eval_strategy='steps',
            save_strategy='steps',
            save_steps=train_cfg.save_steps,
            eval_steps=train_cfg.eval_steps,
            logging_steps=train_cfg.logging_steps,
            save_total_limit=train_cfg.save_total_limit,
            load_best_model_at_end=train_cfg.load_best_model_at_end,
            metric_for_best_model=train_cfg.metric_for_best_model,
            greater_is_better=train_cfg.greater_is_better,
            generation_max_length=train_cfg.generation_max_length,
            predict_with_generate=train_cfg.predict_with_generate,
            remove_unused_columns=False,  # Keep all columns for custom processing
            label_names=['labels'],  # Specify label column name
            push_to_hub=False,
            report_to=['tensorboard'],
        )
        
        # Use default data collator if not provided
        if data_collator is None:
            from src.data.dataset import WhisperDataCollator
            data_collator = WhisperDataCollator(
                processor=processor,
                padding='longest',
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
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Override log method to add custom logging.
        
        Args:
            logs: Dictionary of metrics to log
        """
        # Add custom logging here if needed
        super().log(logs)
        
        # Log WER prominently if present
        if 'eval_wer' in logs:
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
            if metrics is not None and 'eval_wer' in metrics:
                logger.info(f"Checkpoint WER: {metrics['eval_wer']:.4f}")
        
        return checkpoint_folder
