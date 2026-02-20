#!/usr/bin/env python3
"""
Submission preparation and validation script.

This script validates submission JSONL format, checks for required fields,
ensures all test utterances are covered, and reports any formatting issues.

Usage:
    python scripts/prepare_submission.py --submission predictions.jsonl \
        --format data/submission_format_aqPHQ8m.jsonl
    python scripts/prepare_submission.py --submission predictions.jsonl \
        --format data/submission_format_aqPHQ8m.jsonl --output validated_submission.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate and prepare submission JSONL file")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission JSONL file to validate")
    parser.add_argument(
        "--format", type=str, default=None, help="Path to submission format JSONL file (with expected utterance_ids)"
    )
    parser.add_argument("--output", type=str, default=None, help="Path to write validated/corrected submission (optional)")
    parser.add_argument(
        "--log-file", type=str, default=None, help="Path to log file (default: logs/prepare_submission_{timestamp}.log)"
    )
    parser.add_argument("--strict", action="store_true", help="Fail validation if any issues found")

    return parser.parse_args()


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load JSONL file into list of dictionaries.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON objects

    Raises:
        ValueError: If file contains invalid JSON
    """
    data = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
    return data


def validate_submission_format(submission_data: List[Dict[str, Any]], logger: logging.Logger) -> Tuple[bool, List[str]]:
    """
    Validate submission JSONL format and required fields.

    Args:
        submission_data: List of submission entries
        logger: Logger instance

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check if empty
    if not submission_data:
        issues.append("Submission file is empty")
        return False, issues

    logger.info(f"Validating {len(submission_data)} entries...")

    # Required fields
    required_fields = {"utterance_id", "orthographic_text"}

    # Track seen utterance IDs
    seen_ids = set()

    # Validate each entry
    for idx, entry in enumerate(submission_data, 1):
        # Check it's a dictionary
        if not isinstance(entry, dict):
            issues.append(f"Entry {idx}: Not a JSON object")
            continue

        # Check required fields
        missing_fields = required_fields - set(entry.keys())
        if missing_fields:
            issues.append(f"Entry {idx}: Missing required fields: {missing_fields}")

        # Check for unexpected fields
        extra_fields = set(entry.keys()) - required_fields
        if extra_fields:
            issues.append(f"Entry {idx}: Contains extra fields: {extra_fields}")

        # Validate field types
        if "utterance_id" in entry:
            if not isinstance(entry["utterance_id"], str):
                issues.append(f"Entry {idx}: utterance_id must be a string")
            elif not entry["utterance_id"].strip():
                issues.append(f"Entry {idx}: utterance_id is empty")
            else:
                # Check for duplicates
                uid = entry["utterance_id"]
                if uid in seen_ids:
                    issues.append(f"Entry {idx}: Duplicate utterance_id: {uid}")
                seen_ids.add(uid)

        if "orthographic_text" in entry:
            if not isinstance(entry["orthographic_text"], str):
                issues.append(f"Entry {idx}: orthographic_text must be a string")
            # Empty transcription is allowed (though might indicate an issue)
            if not entry["orthographic_text"].strip():
                logger.warning(f"Entry {idx}: orthographic_text is empty (utterance_id: {entry.get('utterance_id', 'N/A')})")

    is_valid = len(issues) == 0

    if is_valid:
        logger.info("✓ All entries have valid format")
    else:
        logger.warning(f"✗ Found {len(issues)} format issues")

    return is_valid, issues


def validate_coverage(
    submission_data: List[Dict[str, Any]], format_data: List[Dict[str, Any]], logger: logging.Logger
) -> Tuple[bool, List[str]]:
    """
    Validate that submission covers all expected utterance IDs.

    Args:
        submission_data: List of submission entries
        format_data: List of expected format entries
        logger: Logger instance

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Extract utterance IDs
    submission_ids = {entry["utterance_id"] for entry in submission_data if "utterance_id" in entry}
    expected_ids = {entry["utterance_id"] for entry in format_data if "utterance_id" in entry}

    logger.info(f"Submission contains {len(submission_ids)} unique utterance IDs")
    logger.info(f"Expected format contains {len(expected_ids)} unique utterance IDs")

    # Check for missing IDs
    missing_ids = expected_ids - submission_ids
    if missing_ids:
        issues.append(f"Missing {len(missing_ids)} utterance IDs from expected set")
        if len(missing_ids) <= 10:
            issues.append(f"  Missing IDs: {sorted(missing_ids)}")
        else:
            issues.append(f"  First 10 missing IDs: {sorted(list(missing_ids))[:10]}")

    # Check for extra IDs
    extra_ids = submission_ids - expected_ids
    if extra_ids:
        issues.append(f"Submission contains {len(extra_ids)} unexpected utterance IDs")
        if len(extra_ids) <= 10:
            issues.append(f"  Extra IDs: {sorted(extra_ids)}")
        else:
            issues.append(f"  First 10 extra IDs: {sorted(list(extra_ids))[:10]}")

    is_valid = len(missing_ids) == 0 and len(extra_ids) == 0

    if is_valid:
        logger.info("✓ Submission covers all expected utterance IDs")
    else:
        logger.warning("✗ Coverage validation failed")

    return is_valid, issues


def write_submission(submission_data: List[Dict[str, Any]], output_path: Path, logger: logging.Logger):
    """
    Write validated submission to JSONL file.

    Args:
        submission_data: List of submission entries
        output_path: Path to output file
        logger: Logger instance
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for entry in submission_data:
            # Only include required fields
            clean_entry = {"utterance_id": entry["utterance_id"], "orthographic_text": entry["orthographic_text"]}
            f.write(json.dumps(clean_entry) + "\n")

    logger.info(f"Wrote validated submission to: {output_path}")


def main():
    """Main validation function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_file = args.log_file
    if log_file is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/prepare_submission_{timestamp}.log"

    logger = setup_logger("prepare_submission", log_file)
    logger.info("=" * 80)
    logger.info("Submission Validation")
    logger.info("=" * 80)
    logger.info(f"Submission file: {args.submission}")
    logger.info(f"Format file: {args.format}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Strict mode: {args.strict}")

    try:
        # Load submission file
        submission_path = Path(args.submission)
        if not submission_path.exists():
            logger.error(f"Submission file not found: {submission_path}")
            sys.exit(1)

        logger.info("Loading submission file...")
        try:
            submission_data = load_jsonl(submission_path)
        except ValueError as e:
            logger.error(f"Failed to parse submission file: {e}")
            sys.exit(1)

        logger.info(f"Loaded {len(submission_data)} entries")

        # Validate format
        logger.info("\n" + "=" * 80)
        logger.info("Validating Format")
        logger.info("=" * 80)

        format_valid, format_issues = validate_submission_format(submission_data, logger)

        if format_issues:
            for issue in format_issues:
                logger.error(f"  ✗ {issue}")

        # Validate coverage if format file provided
        coverage_valid = True
        coverage_issues = []

        if args.format:
            format_path = Path(args.format)
            if not format_path.exists():
                logger.warning(f"Format file not found: {format_path}")
            else:
                logger.info("\n" + "=" * 80)
                logger.info("Validating Coverage")
                logger.info("=" * 80)

                try:
                    format_data = load_jsonl(format_path)
                    logger.info(f"Loaded {len(format_data)} expected entries")

                    coverage_valid, coverage_issues = validate_coverage(submission_data, format_data, logger)

                    if coverage_issues:
                        for issue in coverage_issues:
                            logger.error(f"  ✗ {issue}")

                except ValueError as e:
                    logger.warning(f"Could not parse format file: {e}")

        # Overall validation result
        logger.info("\n" + "=" * 80)
        logger.info("Validation Summary")
        logger.info("=" * 80)

        all_valid = format_valid and coverage_valid

        if all_valid:
            logger.info("✓ Submission is VALID")
            logger.info(f"  Total entries: {len(submission_data)}")
            logger.info(f"  Unique utterance IDs: {len({e['utterance_id'] for e in submission_data if 'utterance_id' in e})}")
        else:
            logger.warning("✗ Submission has ISSUES")
            logger.warning(f"  Format issues: {len(format_issues)}")
            logger.warning(f"  Coverage issues: {len(coverage_issues)}")

        # Write output if specified
        if args.output:
            output_path = Path(args.output)
            logger.info(f"\nWriting validated submission to: {output_path}")
            write_submission(submission_data, output_path, logger)

        logger.info("=" * 80)

        # Exit with error code if validation failed and strict mode enabled
        if args.strict and not all_valid:
            logger.error("Validation failed in strict mode")
            sys.exit(1)

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("Validation failed with error:")
        logger.error("=" * 80)
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
