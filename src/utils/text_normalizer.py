"""Text normalization utilities for consistent preprocessing.

This module provides a wrapper around Whisper's EnglishTextNormalizer to ensure
consistent text preprocessing for WER computation. The normalization matches the
competition scoring methodology, handling punctuation, contractions, and numbers
in a standardized way.
"""

from typing import List, Union

from whisper.normalizers import EnglishTextNormalizer as WhisperNormalizer


class TextNormalizer:
    """Wrapper for Whisper's EnglishTextNormalizer with convenience methods.

    This class wraps the EnglishTextNormalizer from the openai-whisper package
    to provide a consistent interface for text normalization across the project.
    It's critical for ensuring local validation WER matches competition scoring.

    The normalizer handles:
    - Lowercasing
    - Removing punctuation
    - Expanding contractions (e.g., "don't" -> "do not")
    - Normalizing numbers and symbols
    - Removing extra whitespace

    Example:
        >>> normalizer = TextNormalizer()
        >>> normalizer.normalize("Hello, world!")
        'hello world'
        >>> normalizer.normalize_batch(["I don't know.", "It's great!"])
        ['i do not know', 'it is great']
    """

    def __init__(self):
        """Initialize the text normalizer using Whisper's EnglishTextNormalizer."""
        self._normalizer = WhisperNormalizer()

    def normalize(self, text: str) -> str:
        """Normalize a single text string.

        Applies Whisper's English text normalization rules to ensure consistency
        with competition scoring. This includes lowercasing, removing punctuation,
        expanding contractions, and normalizing whitespace.

        Args:
            text: Input text string to normalize

        Returns:
            Normalized text string

        Example:
            >>> normalizer = TextNormalizer()
            >>> normalizer.normalize("I don't know!")
            'i do not know'
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected string, got {type(text).__name__}")

        return self._normalizer(text)

    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize a batch of text strings.

        Applies normalization to multiple texts efficiently. This is useful
        for processing predictions and references before WER computation.

        Args:
            texts: List of text strings to normalize

        Returns:
            List of normalized text strings in the same order

        Raises:
            TypeError: If input is not a list or contains non-string elements

        Example:
            >>> normalizer = TextNormalizer()
            >>> texts = ["Hello, world!", "I don't know."]
            >>> normalizer.normalize_batch(texts)
            ['hello world', 'i do not know']
        """
        if not isinstance(texts, list):
            raise TypeError(f"Expected list, got {type(texts).__name__}")

        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All elements in the list must be strings")

        return [self._normalizer(text) for text in texts]

    def normalize_pair(self, prediction: str, reference: str) -> tuple[str, str]:
        """Normalize a prediction-reference pair.

        Convenience method for normalizing both prediction and reference texts
        before WER computation. Ensures both are processed identically.

        Args:
            prediction: Predicted transcription text
            reference: Ground truth reference text

        Returns:
            Tuple of (normalized_prediction, normalized_reference)

        Example:
            >>> normalizer = TextNormalizer()
            >>> pred, ref = normalizer.normalize_pair("I don't know!", "I don't know.")
            >>> pred
            'i do not know'
            >>> ref
            'i do not know'
        """
        return self.normalize(prediction), self.normalize(reference)

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Make the normalizer callable for convenient usage.

        Automatically handles both single strings and lists of strings,
        dispatching to the appropriate normalization method.

        Args:
            text: Either a single string or list of strings to normalize

        Returns:
            Normalized text (same type as input)

        Example:
            >>> normalizer = TextNormalizer()
            >>> normalizer("Hello!")
            'hello'
            >>> normalizer(["Hello!", "World!"])
            ['hello', 'world']
        """
        if isinstance(text, str):
            return self.normalize(text)
        elif isinstance(text, list):
            return self.normalize_batch(text)
        else:
            raise TypeError(f"Expected str or List[str], got {type(text).__name__}")


def create_normalizer() -> TextNormalizer:
    """Factory function to create a TextNormalizer instance.

    Provides a simple way to instantiate the normalizer with default settings.
    This is useful for consistent initialization across different modules.

    Returns:
        New TextNormalizer instance

    Example:
        >>> normalizer = create_normalizer()
        >>> normalizer.normalize("Hello, world!")
        'hello world'
    """
    return TextNormalizer()
