#!/usr/bin/env python3
"""
Create judge-filtered MAmmoTH-v2 target lookup.

Reads mammoth_style_target_lookup_v2_gemma.json and removes entries
where judge_rejected == True.  The resulting file can be used as
--mammoth_targets in train.py so the training script
never sees rejected expansions (it falls back to human descriptors).

Usage:
    python scripts/create_judge_filtered_targets.py \
        --input mammoth_style_target_lookup_v2_gemma.json \
        --output mammoth_style_target_lookup_v2_gemma_judge_filtered.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter judge-rejected entries from MAmmoTH-v2 targets")
    parser.add_argument(
        "--input", type=str,
        default="mammoth_style_target_lookup_v2_gemma.json",
        help="Input target lookup JSON (default: mammoth_style_target_lookup_v2_gemma.json)",
    )
    parser.add_argument(
        "--output", type=str,
        default="mammoth_style_target_lookup_v2_gemma_judge_filtered.json",
        help="Output filtered JSON (default: mammoth_style_target_lookup_v2_gemma_judge_filtered.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if output_path.exists():
        print(f"ERROR: Output file already exists: {output_path}", file=sys.stderr)
        print("  Delete it first or choose a different --output path.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading targets from {input_path} ...")
    with open(input_path) as f:
        lookup = json.load(f)

    total = len(lookup)
    rejected_keys = [k for k, v in lookup.items() if v.get("judge_rejected")]
    passed_keys = [k for k, v in lookup.items() if not v.get("judge_rejected")]

    # Build filtered lookup (remove rejected entries entirely)
    filtered = {k: v for k, v in lookup.items() if not v.get("judge_rejected")}

    print(f"\n{'='*50}")
    print(f"  Total images:       {total}")
    print(f"  Judge passed:       {len(passed_keys)} ({100*len(passed_keys)/total:.1f}%)")
    print(f"  Judge rejected:     {len(rejected_keys)} ({100*len(rejected_keys)/total:.1f}%)")
    print(f"  Filtered output:    {len(filtered)} entries")
    print(f"{'='*50}")

    # Show a few rejected image names for verification
    if rejected_keys:
        print(f"\n  Sample rejected images (first 5):")
        for k in rejected_keys[:5]:
            print(f"    - {k}")

    print(f"\nSaving filtered targets to {output_path} ...")
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)

    # Verify round-trip
    with open(output_path) as f:
        verify = json.load(f)
    assert len(verify) == len(filtered), "Round-trip verification failed!"
    print(f"Verified: {len(verify)} entries written successfully.")


if __name__ == "__main__":
    main()
