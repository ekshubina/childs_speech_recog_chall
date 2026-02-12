#!/usr/bin/env python3
"""
Evaluation script for computing WER metrics.

This script evaluates a trained model on a validation set, computing
WER overall and broken down by age groups, with detailed error analysis.

Usage:
    python scripts/evaluate.py --model-path checkpoints/finetuned_model --val-jsonl data/val.jsonl
    python scripts/evaluate.py --model-path checkpoints/finetuned_model --val-jsonl data/val.jsonl --output-json results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.inference.predictor import Predictor
from src.training.metrics import WERMetric


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained Whisper model on validation set'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--val-jsonl',
        type=str,
        required=True,
        help='Path to validation JSONL manifest file'
    )
    parser.add_argument(
        '--audio-dirs',
        type=str,
        nargs='+',
        default=['data/audio_0', 'data/audio_1', 'data/audio_2'],
        help='Directories to search for audio files (default: data/audio_0 data/audio_1 data/audio_2)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save evaluation results as JSON'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for inference (default: 16)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu', 'mps'],
        help='Device to run on (default: auto-detect)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: logs/evaluate_{timestamp}.log)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )
    
    return parser.parse_args()


def load_validation_data(val_jsonl: Path) -> List[Dict[str, Any]]:
    """
    Load validation data from JSONL file.
    
    Args:
        val_jsonl: Path to validation JSONL file
    
    Returns:
        List of validation entries
    """
    data = []
    with open(val_jsonl, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def compute_metrics_by_age(
    predictions: List[str],
    references: List[str],
    age_buckets: List[str],
    wer_metric: WERMetric
) -> Dict[str, Any]:
    """
    Compute WER metrics overall and by age bucket.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of ground truth transcriptions
        age_buckets: List of age bucket labels
        wer_metric: WERMetric instance
    
    Returns:
        Dictionary with overall and per-age-bucket metrics
    """
    # Compute overall WER
    overall_wer = wer_metric.compute(predictions, references, return_details=True)
    
    # Group by age bucket
    age_groups = defaultdict(lambda: {'predictions': [], 'references': []})
    for pred, ref, age in zip(predictions, references, age_buckets):
        age_groups[age]['predictions'].append(pred)
        age_groups[age]['references'].append(ref)
    
    # Compute WER per age bucket
    age_metrics = {}
    for age, data in sorted(age_groups.items()):
        if data['predictions']:  # Skip empty groups
            age_wer = wer_metric.compute(
                data['predictions'],
                data['references'],
                return_details=True
            )
            age_metrics[age] = {
                'wer': age_wer['wer'],
                'samples': len(data['predictions']),
                'insertions': age_wer['insertions'],
                'deletions': age_wer['deletions'],
                'substitutions': age_wer['substitutions'],
                'hits': age_wer['hits']
            }
    
    return {
        'overall': {
            'wer': overall_wer['wer'],
            'samples': len(predictions),
            'insertions': overall_wer['insertions'],
            'deletions': overall_wer['deletions'],
            'substitutions': overall_wer['substitutions'],
            'hits': overall_wer['hits']
        },
        'by_age': age_metrics
    }


def print_evaluation_results(metrics: Dict[str, Any], logger: logging.Logger):
    """
    Print formatted evaluation results.
    
    Args:
        metrics: Evaluation metrics dictionary
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    
    # Overall metrics
    overall = metrics['overall']
    logger.info(f"\nOverall Performance:")
    logger.info(f"  Samples:       {overall['samples']:,}")
    logger.info(f"  WER:           {overall['wer']:.4f} ({overall['wer']*100:.2f}%)")
    logger.info(f"  Hits:          {overall['hits']:,}")
    logger.info(f"  Substitutions: {overall['substitutions']:,}")
    logger.info(f"  Deletions:     {overall['deletions']:,}")
    logger.info(f"  Insertions:    {overall['insertions']:,}")
    
    # Per-age metrics
    logger.info(f"\nPerformance by Age Group:")
    logger.info(f"  {'Age Group':<15} {'Samples':<10} {'WER':<10} {'WER %':<10}")
    logger.info(f"  {'-'*45}")
    
    age_metrics = metrics['by_age']
    for age in sorted(age_metrics.keys()):
        age_data = age_metrics[age]
        wer_pct = age_data['wer'] * 100
        logger.info(
            f"  {age:<15} {age_data['samples']:<10,} "
            f"{age_data['wer']:<10.4f} {wer_pct:<10.2f}"
        )
    
    logger.info("=" * 80)


def save_results(metrics: Dict[str, Any], output_path: Path, logger: logging.Logger):
    """
    Save evaluation results to JSON file.
    
    Args:
        metrics: Evaluation metrics dictionary
        output_path: Path to output JSON file
        logger: Logger instance
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_file = args.log_file
    if log_file is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/evaluate_{timestamp}.log'
    
    logger = setup_logger('evaluate', log_file)
    logger.info("=" * 80)
    logger.info("Starting Evaluation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Validation data: {args.val_jsonl}")
    logger.info(f"Audio directories: {args.audio_dirs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device or 'auto-detect'}")
    
    try:
        # Verify input files exist
        val_path = Path(args.val_jsonl)
        if not val_path.exists():
            logger.error(f"Validation file not found: {val_path}")
            sys.exit(1)
        
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model path not found: {model_path}")
            sys.exit(1)
        
        # Load validation data
        logger.info("Loading validation data...")
        val_data = load_validation_data(val_path)
        logger.info(f"Loaded {len(val_data):,} validation samples")
        
        # Extract references and age buckets
        references = [entry['normalized_text'] for entry in val_data]
        age_buckets = [entry.get('age_bucket', 'unknown') for entry in val_data]
        
        # Initialize predictor
        logger.info("Initializing predictor...")
        predictor = Predictor(
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            language='english',
            task='transcribe'
        )
        
        model_info = predictor.get_model_info()
        logger.info(f"Model loaded: {model_info['parameters']:,} parameters")
        logger.info(f"Device: {model_info['device']}")
        
        # Generate predictions
        logger.info("=" * 80)
        logger.info("Generating predictions...")
        logger.info("=" * 80)
        
        # Get predictions from manifest (without writing output)
        results = predictor.predict_from_manifest(
            manifest_path=args.val_jsonl,
            audio_dirs=args.audio_dirs,
            output_path=None,
            show_progress=not args.no_progress
        )
        
        predictions = [result['orthographic_text'] for result in results]
        
        logger.info(f"Generated {len(predictions):,} predictions")
        
        # Compute metrics
        logger.info("Computing metrics...")
        wer_metric = WERMetric()
        metrics = compute_metrics_by_age(
            predictions,
            references,
            age_buckets,
            wer_metric
        )
        
        # Print results
        print_evaluation_results(metrics, logger)
        
        # Save results if output path specified
        if args.output_json:
            output_path = Path(args.output_json)
            save_results(metrics, output_path, logger)
        
        logger.info("=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("Evaluation failed with error:")
        logger.error("=" * 80)
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
