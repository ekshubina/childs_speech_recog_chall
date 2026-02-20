#!/usr/bin/env python3
"""
Prediction script for generating transcriptions.

This script loads a trained model and performs batch inference on audio files,
writing predictions to JSONL format for submission or evaluation.

Usage:
    python scripts/predict.py --model-path checkpoints/finetuned_model \
        --input-jsonl data/test.jsonl --output-jsonl predictions.jsonl
    python scripts/predict.py --model-path checkpoints/finetuned_model \
        --input-jsonl data/test.jsonl --output-jsonl predictions.jsonl --batch-size 32
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import Predictor
from src.utils.logging_utils import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate predictions using trained Whisper model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input-jsonl", type=str, required=True, help="Path to input JSONL manifest file")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Path to output JSONL predictions file")
    parser.add_argument(
        "--audio-dirs",
        type=str,
        nargs="+",
        default=["data/audio_0", "data/audio_1", "data/audio_2"],
        help="Directories to search for audio files (default: data/audio_0 data/audio_1 data/audio_2)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference (default: 16)")
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu", "mps"], help="Device to run on (default: auto-detect)"
    )
    parser.add_argument("--language", type=str, default="english", help="Language for transcription (default: english)")
    parser.add_argument(
        "--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Task type (default: transcribe)"
    )
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file (default: logs/predict_{timestamp}.log)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")

    return parser.parse_args()


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_file = args.log_file
    if log_file is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/predict_{timestamp}.log"

    logger = setup_logger("predict", log_file)
    logger.info("=" * 80)
    logger.info("Starting Prediction")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Input: {args.input_jsonl}")
    logger.info(f"Output: {args.output_jsonl}")
    logger.info(f"Audio directories: {args.audio_dirs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device or 'auto-detect'}")

    try:
        # Verify input files exist
        input_path = Path(args.input_jsonl)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)

        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model path not found: {model_path}")
            sys.exit(1)

        # Verify audio directories exist
        audio_dirs = [Path(d) for d in args.audio_dirs]
        missing_dirs = [d for d in audio_dirs if not d.exists()]
        if missing_dirs:
            logger.warning(f"Some audio directories not found: {missing_dirs}")

        # Create output directory if needed
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize predictor
        logger.info("Initializing predictor...")
        predictor = Predictor(
            model_path=args.model_path, device=args.device, batch_size=args.batch_size, language=args.language, task=args.task
        )

        # Log model info
        model_info = predictor.get_model_info()
        logger.info(f"Model loaded: {model_info['parameters']:,} parameters")
        logger.info(f"Device: {model_info['device']}")

        # Generate predictions
        logger.info("=" * 80)
        logger.info("Generating predictions...")
        logger.info("=" * 80)

        results = predictor.predict_from_manifest(
            manifest_path=args.input_jsonl,
            audio_dirs=args.audio_dirs,
            output_path=args.output_jsonl,
            show_progress=not args.no_progress,
        )

        # Log results
        logger.info("=" * 80)
        logger.info("Prediction completed!")
        logger.info("=" * 80)
        logger.info(f"Total predictions: {len(results):,}")
        logger.info(f"Output written to: {args.output_jsonl}")

        # Show sample predictions
        if results:
            logger.info("Sample predictions:")
            for i, result in enumerate(results[:3]):
                logger.info(f"  [{i + 1}] {result['utterance_id']}: {result['orthographic_text'][:100]}...")

        logger.info("=" * 80)
        logger.info("Prediction script completed successfully!")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("Prediction failed with error:")
        logger.error("=" * 80)
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
