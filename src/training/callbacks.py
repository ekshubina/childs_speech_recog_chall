"""
Custom HuggingFace Trainer callbacks for Whisper fine-tuning.
"""

import logging

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class EncoderUnfreezeCallback(TrainerCallback):
    """
    Freezes the Whisper encoder at the start of training and unfreezes it
    once ``state.global_step`` reaches ``freeze_steps``.

    This gives the decoder time to adapt to the pre-trained encoder
    representations before the encoder parameters are disturbed.

    Args:
        freeze_steps: Global step at which to unfreeze the encoder.
                      Set to 0 to disable (encoder is never frozen by this callback).

    Example:
        >>> callback = EncoderUnfreezeCallback(freeze_steps=1000)
    """

    def __init__(self, freeze_steps: int) -> None:
        self.freeze_steps = freeze_steps
        self._unfrozen = False  # Guard so we only log once

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Freeze the encoder at the very start of training."""
        if model is None or self.freeze_steps <= 0:
            return

        for param in model.model.encoder.parameters():
            param.requires_grad = False

        encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
        logger.info(
            f"[EncoderUnfreezeCallback] Encoder frozen at step 0 "
            f"({encoder_params:,} params). Will unfreeze at step {self.freeze_steps}."
        )

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Unfreeze the encoder once the target step is reached."""
        if model is None or self._unfrozen or self.freeze_steps <= 0:
            return

        if state.global_step >= self.freeze_steps:
            for param in model.model.encoder.parameters():
                param.requires_grad = True

            self._unfrozen = True
            encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
            logger.info(
                f"[EncoderUnfreezeCallback] Encoder unfrozen at step {state.global_step} "
                f"({encoder_params:,} params now trainable)."
            )
