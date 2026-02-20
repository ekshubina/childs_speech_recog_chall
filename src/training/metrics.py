"""
WER metric computation for ASR evaluation.

This module provides Word Error Rate (WER) computation using jiwer library
with Whisper's text normalization to match competition scoring.
"""

import logging
from typing import Dict, List, Optional, Union

import jiwer
from transformers import WhisperTokenizer

logger = logging.getLogger(__name__)


class WERMetric:
    """
    Word Error Rate metric with text normalization.

    Uses Whisper's EnglishTextNormalizer for consistent preprocessing
    of predictions and references before computing WER with jiwer.

    This ensures WER computation matches the competition's scoring methodology.

    Attributes:
        normalizer: Whisper's EnglishTextNormalizer instance

    Example:
        >>> metric = WERMetric()
        >>> predictions = ["Hello world", "How are you"]
        >>> references = ["hello world", "how are you"]
        >>> wer = metric.compute(predictions, references)
        >>> print(f"WER: {wer:.2%}")
    """

    def __init__(self):
        """
        Initialize WER metric with text normalizer.

        Loads Whisper's EnglishTextNormalizer for preprocessing text
        before WER computation.
        """
        try:
            # Import Whisper normalizer
            from whisper.normalizers import EnglishTextNormalizer

            self.normalizer = EnglishTextNormalizer()
            logger.info("Initialized WERMetric with EnglishTextNormalizer")
        except ImportError:
            logger.warning("Failed to import whisper normalizer. " "Install with: pip install openai-whisper")
            # Fallback to basic normalization
            self.normalizer = None

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text using Whisper's normalizer.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text
        """
        if self.normalizer is not None:
            return self.normalizer(text)
        else:
            # Basic fallback normalization
            return text.lower().strip()

    def compute(
        self, predictions: List[str], references: List[str], return_details: bool = False
    ) -> Union[float, Dict[str, Union[float, int]]]:
        """
        Compute Word Error Rate between predictions and references.

        Applies text normalization to both predictions and references
        before computing WER using jiwer library.

        Args:
            predictions: List of predicted transcriptions
            references: List of ground truth transcriptions
            return_details: If True, return dict with WER, insertions,
                          deletions, substitutions

        Returns:
            WER as float (0.0 = perfect, 1.0 = completely wrong).
            If return_details=True, returns dict with detailed metrics.

        Raises:
            ValueError: If predictions and references have different lengths

        Example:
            >>> metric = WERMetric()
            >>> preds = ["the cat sat on the mat", "hello world"]
            >>> refs = ["the cat sits on the mat", "hello world"]
            >>> wer = metric.compute(preds, refs)
            >>> print(f"WER: {wer:.2%}")
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) " f"must have the same length"
            )

        if len(predictions) == 0:
            logger.warning("Empty predictions and references provided")
            return 0.0 if not return_details else {"wer": 0.0, "insertions": 0, "deletions": 0, "substitutions": 0, "hits": 0}

        # Normalize all texts
        normalized_predictions = [self._normalize_text(pred) for pred in predictions]
        normalized_references = [self._normalize_text(ref) for ref in references]

        # Compute WER using jiwer
        try:
            if return_details:
                # Get detailed measures using process_words
                result = jiwer.process_words(normalized_references, normalized_predictions)
                return {
                    "wer": result.wer,
                    "insertions": result.insertions,
                    "deletions": result.deletions,
                    "substitutions": result.substitutions,
                    "hits": result.hits,
                }
            else:
                # Just compute WER
                wer = jiwer.wer(normalized_references, normalized_predictions)
                return wer

        except Exception as e:
            logger.error(f"Failed to compute WER: {e}")
            raise RuntimeError(f"WER computation failed: {e}") from e


def compute_metrics(pred, tokenizer: Optional[WhisperTokenizer] = None) -> Dict[str, float]:
    """
    Compute metrics callback for Hugging Face Seq2SeqTrainer.

    Decodes predictions and labels, then computes WER using WERMetric.
    This function is designed to be passed as compute_metrics to Trainer.

    Args:
        pred: EvalPrediction object containing predictions and label_ids
        tokenizer: WhisperTokenizer for decoding. If None, creates a new one.

    Returns:
        Dictionary with 'wer' metric

    Example:
        >>> from functools import partial
        >>> from transformers import WhisperTokenizer
        >>>
        >>> tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-medium')
        >>> compute_metrics_fn = partial(compute_metrics, tokenizer=tokenizer)
        >>>
        >>> trainer = Seq2SeqTrainer(
        ...     model=model,
        ...     compute_metrics=compute_metrics_fn,
        ...     ...
        ... )
    """
    # Extract predictions and labels
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 in labels (used for padding) with tokenizer.pad_token_id
    if tokenizer is None:
        logger.warning("No tokenizer provided, creating default WhisperTokenizer")
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium")

    # Replace -100 (padding) with pad_token_id for decoding
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer_metric = WERMetric()
    wer = wer_metric.compute(pred_str, label_str)

    logger.info(f"Validation WER: {wer:.4f}")

    return {"wer": wer}
