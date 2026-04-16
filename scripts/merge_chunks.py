#!/usr/bin/env python3
"""Merge chunked MAmmoTH-v2 JSON outputs into a single file."""
import argparse
import json
import glob
import sys


def main():
    parser = argparse.ArgumentParser(description="Merge chunked MAmmoTH-v2 JSON outputs")
    parser.add_argument("--pattern", type=str, default="mammoth_v2_chunk_*_gemma.json",
                        help="Glob pattern for chunk files")
    parser.add_argument("--output", type=str, default="mammoth_style_target_lookup_v2_gemma.json",
                        help="Merged output file")
    args = parser.parse_args()

    chunk_files = sorted(glob.glob(args.pattern))
    if not chunk_files:
        print(f"ERROR: No files matching '{args.pattern}'")
        sys.exit(1)

    merged = {}
    for path in chunk_files:
        with open(path) as f:
            data = json.load(f)
        print(f"  {path}: {len(data)} images")
        merged.update(data)

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged {len(chunk_files)} chunks -> {args.output} ({len(merged)} images)")


if __name__ == "__main__":
    main()
