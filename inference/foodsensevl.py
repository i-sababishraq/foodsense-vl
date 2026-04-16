"""
Quick demo: run inference on 5 sample images with the FoodSense-VL model.

Usage:
  python inference/foodsensevl.py
  python inference/foodsensevl.py --adapter_dir checkpoints/foodsense-vl_chkpt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import run_gemma_inference, compute_metrics
from dataset import load_human_sensory_data
from config.prompts import USER_PROMPT

IMAGE_DIR = "data/Images"
HUMAN_CSV = "data/FINAL_DATASET_COMPLETE_with_rescaling.csv"

# 5 sample images for quick demo
IMAGE_PATHS = [
    "0001_01lamiW2bWW0rXlllNHYMA.jpg",
    "0002_01zZeZBIFZ82S5XmA4GYJg.jpg",
    "0005_08Eu2m3RTrpssX9GIKtHtg.jpg",
    "0010_0dHJ9fque7joEy7J0UrHmA.jpg",
    "0015_0g2pruxDhqhh2E-cEoYOLA.jpg",
]

DEFAULT_ADAPTER = "checkpoints/foodsense-vl_chkpt"
BASE_MODEL = "google/gemma-3-27b-it"


def _load_ground_truth(image_names: list) -> list:
    """Load ground-truth ratings for the given image names (per-image mean)."""
    df = load_human_sensory_data(HUMAN_CSV, IMAGE_DIR, require_all_caninfer=True)
    df = df.copy()
    df["_img"] = df["saved_path"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x)
    )
    gt_df = df.groupby("_img")[
        ["sensory_taste", "sensory_smell", "sensory_texture", "sensory_sound"]
    ].mean()
    gt_map = {idx: row.to_dict() for idx, row in gt_df.iterrows()}

    targets = []
    for img in image_names:
        row = gt_map.get(img, {})
        targets.append({
            "taste": row.get("sensory_taste"),
            "smell": row.get("sensory_smell"),
            "texture": row.get("sensory_texture"),
            "sound": row.get("sensory_sound"),
        })
    return targets


def main():
    parser = argparse.ArgumentParser(description="FoodSense-VL quick demo")
    parser.add_argument(
        "--adapter_dir", type=str, default=DEFAULT_ADAPTER,
        help="Path to QLoRA adapter directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="demo_outputs",
        help="Output directory for predictions",
    )
    args = parser.parse_args()

    prompt = USER_PROMPT
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Adapter: {args.adapter_dir}")
    print(f"Prompt:  {prompt[:80]}...")
    print(f"Images:  {len(IMAGE_PATHS)}\n")

    # Load ground truth
    print("Loading ground truth...")
    targets = _load_ground_truth(IMAGE_PATHS)

    # Run inference
    print("Running inference...")
    results = run_gemma_inference(
        image_paths=IMAGE_PATHS,
        adapter_dir=args.adapter_dir,
        base_model=BASE_MODEL,
        base_only=False,
        prompt=prompt,
        max_new_tokens=1024,
        image_dir=IMAGE_DIR,
    )

    # Save predictions
    jsonl_file = out_path / "predictions.jsonl"
    with open(jsonl_file, "w") as f:
        for r in results:
            f.write(json.dumps({
                "image": r["image"],
                "text": r["text"],
                "ratings": r["ratings"],
            }) + "\n")

    txt_file = out_path / "predictions.txt"
    with open(txt_file, "w") as f:
        f.write(f"=== FoodSense-VL Demo ===\n")
        f.write(f"Adapter: {args.adapter_dir}\n")
        f.write(f"Prompt: {prompt}\n\n")
        for r in results:
            f.write(f"\n--- {r['image']} ---\n")
            f.write(f"Ratings: {r['ratings']}\n")
            f.write(r["text"] + "\n")

    # Compute metrics
    pred_ratings = [r["ratings"] for r in results]
    metrics = compute_metrics(pred_ratings, targets)

    metrics_file = out_path / "eval_results.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "n_images": len(IMAGE_PATHS),
            "adapter_dir": args.adapter_dir,
            "metrics": metrics,
        }, f, indent=2)

    SENSES = ["taste", "smell", "texture", "sound"]
    rows = [
        {
            "sense": s,
            "mae": metrics[s]["mae"],
            "rmse": metrics[s]["rmse"],
            "pearson": metrics[s]["pearson"],
            "n": metrics[s]["n"],
        }
        for s in SENSES
    ]
    rows.append({
        "sense": "overall",
        "mae": metrics["overall_mae"],
        "rmse": "",
        "pearson": metrics["overall_pearson"],
        "n": len(IMAGE_PATHS),
    })
    csv_file = out_path / "eval_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)

    print(f"\nOverall MAE: {metrics['overall_mae']:.4f}  "
          f"Pearson: {metrics['overall_pearson']:.4f}")
    print(f"Saved: {jsonl_file}, {txt_file}, {metrics_file}, {csv_file}")


if __name__ == "__main__":
    main()
