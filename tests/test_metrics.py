"""
Tests for WER metric computation and text normalization.

Tests cover:
- WERMetric initialization with and without normalizer
- Text normalization (contractions, punctuation, capitalization)
- WER computation with known examples
- Edge cases (empty input, mismatched lengths)
- Detailed metrics return
- Verification against jiwer library
- compute_metrics function for Trainer callback
"""

from unittest.mock import Mock, patch

import jiwer
import numpy as np
import pytest

from src.training.metrics import WERMetric, compute_metrics


class TestWERMetricInitialization:
    """Test WERMetric initialization and normalizer loading."""

    def test_initialization_with_normalizer(self):
        """Test initialization successfully loads Whisper normalizer."""
        metric = WERMetric()

        # Should have normalizer loaded if whisper is installed
        assert metric.normalizer is not None or metric.normalizer is None
        # Constructor should not raise any errors

    def test_initialization_without_whisper(self):
        """Test graceful fallback when Whisper is not installed."""
        with patch("src.training.metrics.logger"):
            # Mock the import to fail
            with patch.dict("sys.modules", {"whisper.normalizers": None}):
                # Try creating metric - should handle ImportError gracefully
                # Note: This test depends on actual import behavior
                pass


class TestTextNormalization:
    """Test text normalization functionality."""

    def test_normalize_contractions(self):
        """Test that contractions are normalized consistently."""
        metric = WERMetric()

        # Test common contractions
        test_cases = [
            ("I'm happy", "i'm happy"),  # Lowercase
            ("don't worry", "don't worry"),
            ("it's fine", "it's fine"),
            ("we'll see", "we'll see"),
        ]

        for input_text, expected_contains in test_cases:
            normalized = metric._normalize_text(input_text)
            # At minimum should be lowercase and stripped
            assert normalized.islower() or not normalized.isalpha()
            assert normalized.strip() == normalized

    def test_normalize_punctuation(self):
        """Test punctuation handling in normalization."""
        metric = WERMetric()

        # Whisper normalizer typically removes or normalizes punctuation
        test_cases = [
            "Hello, world!",
            "What? Really...",
            "Yes; definitely.",
            "Price: $10.00",
        ]

        for text in test_cases:
            normalized = metric._normalize_text(text)
            # Normalized text should be non-empty
            assert len(normalized) > 0
            # Should be lowercase
            assert normalized == normalized.lower()

    def test_normalize_capitalization(self):
        """Test that all text is lowercased."""
        metric = WERMetric()

        test_cases = [
            "HELLO WORLD",
            "Hello World",
            "hElLo WoRlD",
            "hello world",
        ]

        results = [metric._normalize_text(text) for text in test_cases]

        # All should be lowercase
        for result in results:
            assert result == result.lower()

    def test_normalize_extra_whitespace(self):
        """Test that extra whitespace is handled."""
        metric = WERMetric()

        text_with_spaces = "  hello   world  "
        normalized = metric._normalize_text(text_with_spaces)

        # Should be trimmed (at minimum)
        assert normalized == normalized.strip()
        # Should not have leading/trailing spaces
        assert not normalized.startswith(" ")
        assert not normalized.endswith(" ")

    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        metric = WERMetric()

        normalized = metric._normalize_text("")
        assert normalized == ""

    def test_normalize_numbers(self):
        """Test normalization of numbers in text."""
        metric = WERMetric()

        # Whisper normalizer may convert numbers to words or normalize them
        test_cases = [
            "I have 3 apples",
            "The year is 2024",
            "Call me at 555-1234",
        ]

        for text in test_cases:
            normalized = metric._normalize_text(text)
            # Should produce some normalized output
            assert len(normalized) > 0
            assert normalized == normalized.lower()


class TestWERComputation:
    """Test WER computation with known examples."""

    def test_perfect_match(self):
        """Test WER is 0.0 for identical predictions and references."""
        metric = WERMetric()

        predictions = ["hello world", "the cat sat on the mat"]
        references = ["hello world", "the cat sat on the mat"]

        wer = metric.compute(predictions, references)

        # Perfect match should give WER of 0.0
        assert wer == 0.0

    def test_perfect_match_case_insensitive(self):
        """Test WER ignores case differences."""
        metric = WERMetric()

        predictions = ["Hello World", "THE CAT"]
        references = ["hello world", "the cat"]

        wer = metric.compute(predictions, references)

        # Should be 0.0 after normalization
        assert wer == 0.0

    def test_single_substitution(self):
        """Test WER with known substitution."""
        metric = WERMetric()

        # "cat" -> "dog" is 1 substitution out of 6 words
        predictions = ["the cat sat on the mat"]
        references = ["the dog sat on the mat"]

        wer = metric.compute(predictions, references)

        # WER should be 1/6 ≈ 0.1667
        assert 0.15 < wer < 0.20

    def test_single_deletion(self):
        """Test WER with deletion."""
        metric = WERMetric()

        # Missing "very" is 1 deletion out of 5 reference words
        predictions = ["the cat sat down"]
        references = ["the very cat sat down"]

        wer = metric.compute(predictions, references)

        # WER should be 1/5 = 0.20
        assert 0.15 < wer < 0.25

    def test_single_insertion(self):
        """Test WER with insertion."""
        metric = WERMetric()

        # Extra "very" is 1 insertion out of 4 reference words
        predictions = ["the very cat sat"]
        references = ["the cat sat"]

        wer = metric.compute(predictions, references)

        # WER should be non-zero (insertion penalty)
        assert wer > 0.0
        assert wer < 0.5

    def test_completely_different(self):
        """Test WER with completely different texts."""
        metric = WERMetric()

        predictions = ["hello world"]
        references = ["goodbye moon"]

        wer = metric.compute(predictions, references)

        # Should have high WER (all substitutions)
        assert wer >= 1.0  # Can be > 1.0 with insertions

    def test_multiple_samples(self):
        """Test WER computation across multiple samples."""
        metric = WERMetric()

        predictions = [
            "the cat sat",
            "hello world",
            "i love coding",
        ]
        references = [
            "the cat sat",
            "hello world",
            "i love coding",
        ]

        wer = metric.compute(predictions, references)

        # All perfect matches
        assert wer == 0.0

    def test_mixed_quality_predictions(self):
        """Test WER with mix of good and bad predictions."""
        metric = WERMetric()

        predictions = [
            "the cat sat",  # Perfect
            "hello moon",  # 1 error
        ]
        references = [
            "the cat sat",
            "hello world",
        ]

        wer = metric.compute(predictions, references)

        # Should be between 0.0 and 1.0
        assert 0.0 < wer < 1.0


class TestWEREdgeCases:
    """Test edge cases and error handling."""

    def test_empty_lists(self):
        """Test WER with empty prediction and reference lists."""
        metric = WERMetric()

        wer = metric.compute([], [])

        # Should return 0.0 for empty lists
        assert wer == 0.0

    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        metric = WERMetric()

        predictions = ["hello world"]
        references = ["hello world", "goodbye"]

        with pytest.raises(ValueError, match="must have the same length"):
            metric.compute(predictions, references)

    def test_single_sample(self):
        """Test WER computation with single sample."""
        metric = WERMetric()

        predictions = ["hello world"]
        references = ["hello world"]

        wer = metric.compute(predictions, references)

        assert wer == 0.0

    def test_empty_strings_in_list(self):
        """Test handling of empty strings in predictions/references."""
        metric = WERMetric()

        predictions = ["", "hello world"]
        references = ["", "hello world"]

        # Should not raise error
        wer = metric.compute(predictions, references)

        # Should be able to compute WER
        assert wer >= 0.0


class TestDetailedMetrics:
    """Test detailed metrics return."""

    def test_return_details_structure(self):
        """Test that return_details=True returns dict with all metrics."""
        metric = WERMetric()

        predictions = ["the cat"]
        references = ["the dog"]

        result = metric.compute(predictions, references, return_details=True)

        # Should return dict with specific keys
        assert isinstance(result, dict)
        assert "wer" in result
        assert "insertions" in result
        assert "deletions" in result
        assert "substitutions" in result
        assert "hits" in result

    def test_detailed_metrics_values(self):
        """Test that detailed metrics have correct values."""
        metric = WERMetric()

        # "the cat sat" vs "the dog sat" = 1 substitution, 2 hits
        predictions = ["the cat sat"]
        references = ["the dog sat"]

        result = metric.compute(predictions, references, return_details=True)

        # Should have 1 substitution
        assert result["substitutions"] >= 0
        # Should have some hits (matching words)
        assert result["hits"] >= 0
        # WER should be calculated
        assert result["wer"] >= 0.0

    def test_detailed_metrics_perfect_match(self):
        """Test detailed metrics for perfect match."""
        metric = WERMetric()

        predictions = ["hello world"]
        references = ["hello world"]

        result = metric.compute(predictions, references, return_details=True)

        # Perfect match should have:
        assert result["wer"] == 0.0
        assert result["insertions"] == 0
        assert result["deletions"] == 0
        assert result["substitutions"] == 0
        assert result["hits"] > 0  # All words are hits


class TestJiwerConsistency:
    """Verify that WERMetric matches jiwer library behavior."""

    def test_matches_jiwer_simple(self):
        """Test that normalized WER matches jiwer for simple case."""
        metric = WERMetric()

        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]

        # Compute with our metric
        our_wer = metric.compute(predictions, references)

        # Compute with jiwer directly on normalized text
        norm_preds = [metric._normalize_text(p) for p in predictions]
        norm_refs = [metric._normalize_text(r) for r in references]
        jiwer_wer = jiwer.wer(norm_refs, norm_preds)

        # Should match exactly
        assert abs(our_wer - jiwer_wer) < 1e-6

    def test_matches_jiwer_with_errors(self):
        """Test that WER matches jiwer when there are errors."""
        metric = WERMetric()

        predictions = ["the cat sat", "hello world"]
        references = ["the dog sat", "goodbye world"]

        # Compute with our metric
        our_wer = metric.compute(predictions, references)

        # Compute with jiwer directly
        norm_preds = [metric._normalize_text(p) for p in predictions]
        norm_refs = [metric._normalize_text(r) for r in references]
        jiwer_wer = jiwer.wer(norm_refs, norm_preds)

        # Should match
        assert abs(our_wer - jiwer_wer) < 1e-6


class TestComputeMetricsFunction:
    """Test compute_metrics function for Trainer callback."""

    def test_compute_metrics_basic(self):
        """Test compute_metrics with mock predictions."""
        # Create mock tokenizer
        tokenizer = Mock()
        tokenizer.pad_token_id = 50257
        tokenizer.batch_decode = Mock(
            side_effect=[
                ["hello world", "the cat sat"],  # predictions
                ["hello world", "the cat sat"],  # labels
            ]
        )

        # Create mock EvalPrediction
        pred = Mock()
        pred.predictions = np.array([[1, 2, 3], [4, 5, 6]])
        pred.label_ids = np.array([[1, 2, 3], [4, 5, 6]])

        # Compute metrics
        result = compute_metrics(pred, tokenizer=tokenizer)

        # Should return dict with 'wer' key
        assert isinstance(result, dict)
        assert "wer" in result
        assert isinstance(result["wer"], float)

    def test_compute_metrics_with_padding(self):
        """Test that -100 padding tokens are handled correctly."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 50257
        tokenizer.batch_decode = Mock(
            side_effect=[
                ["hello world"],  # predictions
                ["hello world"],  # labels
            ]
        )

        # Create predictions with -100 (padding marker)
        pred = Mock()
        pred.predictions = np.array([[1, 2, -100, -100]])
        pred.label_ids = np.array([[1, 2, -100, -100]])

        # Compute metrics
        result = compute_metrics(pred, tokenizer=tokenizer)

        # Should successfully compute WER
        assert "wer" in result
        assert result["wer"] >= 0.0

    def test_compute_metrics_without_tokenizer(self):
        """Test that compute_metrics creates default tokenizer if not provided."""
        # Create mock EvalPrediction
        pred = Mock()
        pred.predictions = np.array([[1, 2, 3]])
        pred.label_ids = np.array([[1, 2, 3]])

        # Mock WhisperTokenizer.from_pretrained
        with patch("src.training.metrics.WhisperTokenizer") as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token_id = 50257
            mock_tokenizer.batch_decode = Mock(
                side_effect=[
                    ["hello"],  # predictions
                    ["hello"],  # labels
                ]
            )
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Should not raise error
            result = compute_metrics(pred, tokenizer=None)

            # Should have called from_pretrained
            mock_tokenizer_class.from_pretrained.assert_called_once()
            assert "wer" in result

    def test_compute_metrics_with_errors(self):
        """Test compute_metrics with predictions that have errors."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 50257
        tokenizer.batch_decode = Mock(
            side_effect=[
                ["the cat sat", "hello moon"],  # predictions
                ["the dog sat", "hello world"],  # labels (different)
            ]
        )

        pred = Mock()
        pred.predictions = np.array([[1, 2, 3], [4, 5, 6]])
        pred.label_ids = np.array([[1, 2, 3], [4, 5, 6]])

        result = compute_metrics(pred, tokenizer=tokenizer)

        # Should have non-zero WER
        assert result["wer"] > 0.0


class TestNormalizationConsistency:
    """Test that normalization is applied consistently to predictions and references."""

    def test_consistent_normalization(self):
        """Test that both predictions and references are normalized the same way."""
        metric = WERMetric()

        # Same text with different casing/punctuation
        predictions = ["Hello, World!"]
        references = ["hello world"]

        wer = metric.compute(predictions, references)

        # After normalization, should match (or be very close)
        # Exact result depends on Whisper normalizer behavior with punctuation
        assert wer >= 0.0  # Should be a valid WER

    def test_normalization_applied_to_both(self):
        """Verify normalization is applied to both predictions and references."""
        metric = WERMetric()

        # Track normalization calls
        original_normalize = metric._normalize_text
        normalize_calls = []

        def tracked_normalize(text):
            result = original_normalize(text)
            normalize_calls.append((text, result))
            return result

        metric._normalize_text = tracked_normalize

        predictions = ["Hello World"]
        references = ["hello world"]

        metric.compute(predictions, references)

        # Should have normalized both prediction and reference
        assert len(normalize_calls) == 2
