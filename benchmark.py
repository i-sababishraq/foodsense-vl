#!/usr/bin/env python3
"""
Comprehensive Sensory Evaluation Benchmark (Tier 1 + Tier 4)

Computes advanced metrics from existing predictions WITHOUT new inference:
  - Raw MAE / RMSE / Pearson (baseline, same as original)
  - Calibrated MAE (linear regression fit on val-like portion)
  - Spearman ρ (rank correlation, scale-invariant)
  - Lin's CCC (concordance correlation coefficient)
  - Ordinal Accuracy (Low/Med/High 3-class)
  - Within-1 Accuracy (predictions within 1.0 of ground truth)
  - Parse Success Rate
  - Human inter-annotator agreement (Tier 4)

Usage:
  python benchmark.py
  python benchmark.py --output_dir eval_outputs --human_csv data/FINAL_DATASET_COMPLETE_with_rescaling.csv
"""

import argparse
import json
import sys
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from dataset import create_image_level_splits, load_human_sensory_data

SENSES = ["taste", "smell", "texture", "sound"]
ORDINAL_BINS = [(1.0, 2.33, "Low"), (2.33, 3.67, "Med"), (3.67, 5.0, "High")]


# ─────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────

def load_partial_shards(eval_dir: Path, model_name: str) -> Optional[Tuple[List[str], List[Dict], List[Dict]]]:
    """Load and merge partial shard files for a model."""
    shard_files = sorted(eval_dir.glob(f"partial/{model_name}_shard*.json"))
    if not shard_files:
        return None
    
    all_images, all_preds, all_targets = [], [], []
    for sf in shard_files:
        with open(sf) as f:
            data = json.load(f)
        all_images.extend(data.get("images", []))
        all_preds.extend(data.get("preds", []))
        all_targets.extend(data.get("targets", []))
    
    return all_images, all_preds, all_targets


def load_internvl_predictions(eval_dir: Path) -> Optional[Tuple[List[str], List[Dict]]]:
    """Load InternVL predictions from JSONL."""
    jsonl = eval_dir / "internvl" / "internvl_predictions.jsonl"
    if not jsonl.exists():
        return None
    
    images, preds = [], []
    with open(jsonl) as f:
        for line in f:
            entry = json.loads(line.strip())
            images.append(entry["image"])
            preds.append(entry.get("ratings", {}))
    return images, preds


def _load_jsonl_predictions(eval_dir: Path, model_name: str, gt_map: Dict) -> Dict:
    """Load predictions from canonical JSONL path; fail if missing."""
    jsonl_path = eval_dir / model_name / f"{model_name}_predictions.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Missing predictions for '{model_name}': expected {jsonl_path}. "
            "Run the corresponding evaluation job first."
        )
    
    images, preds = [], []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            images.append(entry["image"])
            preds.append(entry.get("ratings", {}))

    if not images:
        raise ValueError(f"Predictions file for '{model_name}' is empty: {jsonl_path}")
    
    targets = []
    for img in images:
        gt = gt_map.get(img, {})
        targets.append({
            "taste": gt.get("sensory_taste"),
            "smell": gt.get("sensory_smell"),
            "texture": gt.get("sensory_texture"),
            "sound": gt.get("sensory_sound"),
        })
    return {"images": images, "preds": preds, "targets": targets}


def load_all_model_predictions(eval_dir: Path, gt_map: Dict, model_names: List[str]) -> Dict[str, Dict]:
    """Load predictions for requested models from canonical paths; fail-fast on missing files."""
    models = {}
    for model_name in model_names:
        models[model_name] = _load_jsonl_predictions(eval_dir, model_name, gt_map)
    return models


# ─────────────────────────────────────────────────────────
# Metric Functions
# ─────────────────────────────────────────────────────────

def _get_valid_pairs(preds, targets, sense):
    """Extract valid (pred, target) pairs for a given sense."""
    p_vals, t_vals = [], []
    for pred, tgt in zip(preds, targets):
        pv = pred.get(sense)
        tv = tgt.get(sense)
        if pv is not None and tv is not None and not np.isnan(pv) and not np.isnan(tv):
            p_vals.append(float(pv))
            t_vals.append(float(tv))
    return np.array(p_vals), np.array(t_vals)


def raw_mae(p, t):
    return float(np.mean(np.abs(p - t))) if len(p) > 0 else np.nan


def raw_rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2))) if len(p) > 0 else np.nan


def pearson_r(p, t):
    if len(p) < 3:
        return np.nan
    r, _ = stats.pearsonr(p, t)
    return float(r)


def spearman_rho(p, t):
    if len(p) < 3:
        return np.nan
    rho, _ = stats.spearmanr(p, t)
    return float(rho)


def calibrated_mae(p, t, n_splits=5, random_state=42):
    """Calibrated MAE: 5-fold CV linear regression calibration."""
    if len(p) < 10:
        return np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_errors = []
    for train_idx, test_idx in kf.split(p):
        reg = LinearRegression()
        reg.fit(p[train_idx].reshape(-1, 1), t[train_idx])
        p_cal = reg.predict(p[test_idx].reshape(-1, 1))
        all_errors.extend(np.abs(p_cal - t[test_idx]).tolist())
    return float(np.mean(all_errors))


def lins_ccc(p, t):
    """Lin's Concordance Correlation Coefficient (CCC).
    
    Measures agreement between predictions and targets, penalising
    both lack of correlation and deviation from the 45-degree line.
    CCC = 2*cov(p,t) / (var(p) + var(t) + (mean(p) - mean(t))^2)
    """
    if len(p) < 3:
        return np.nan
    p_mean, t_mean = np.mean(p), np.mean(t)
    p_var, t_var = np.var(p), np.var(t)
    covariance = np.mean((p - p_mean) * (t - t_mean))
    denom = p_var + t_var + (p_mean - t_mean) ** 2
    if denom < 1e-12:
        return np.nan
    return float(2.0 * covariance / denom)


def pairwise_ordering_accuracy(p, t):
    """Proportion of image pairs where model correctly orders them (Harrell's C)."""
    if len(p) < 2:
        return np.nan
    concordant, discordant, tied = 0, 0, 0
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            t_diff = t[j] - t[i]
            p_diff = p[j] - p[i]
            if abs(t_diff) < 1e-6:
                tied += 1
            elif t_diff * p_diff > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return float(concordant / total) if total > 0 else np.nan


def ordinal_accuracy(p, t):
    """3-class ordinal accuracy: Low (1-2.33), Med (2.33-3.67), High (3.67-5)."""
    def to_bin(val):
        if val <= 2.33:
            return 0
        elif val <= 3.67:
            return 1
        else:
            return 2
    
    if len(p) == 0:
        return np.nan
    correct = sum(1 for pv, tv in zip(p, t) if to_bin(pv) == to_bin(tv))
    return float(correct / len(p))


def within_k_accuracy(p, t, k=1.0):
    """Fraction of predictions within k of ground truth."""
    if len(p) == 0:
        return np.nan
    return float(np.mean(np.abs(p - t) <= k))


def parse_success_rate(preds):
    """Fraction of predictions where all 4 senses were parsed."""
    if len(preds) == 0:
        return 0.0
    success = sum(1 for p in preds if all(p.get(s) is not None for s in SENSES))
    return float(success / len(preds))


# ─────────────────────────────────────────────────────────
# Human Agreement (Tier 4)
# ─────────────────────────────────────────────────────────

def compute_human_agreement(df: pd.DataFrame, test_images: List[str]) -> Dict[str, Any]:
    """Compute inter-annotator statistics from raw human data."""
    sense_cols = {
        "taste": "sensory_taste",
        "smell": "sensory_smell",
        "texture": "sensory_texture",
        "sound": "sensory_sound",
    }
    
    def img_name(row):
        sp = row.get("saved_path")
        return sp[0] if isinstance(sp, list) and sp else str(sp) if sp else ""
    
    df = df.copy()
    df["_img"] = df.apply(img_name, axis=1)
    df = df[df["_img"].isin(test_images)]
    
    results = {}
    for sense, col in sense_cols.items():
        # For each image, compute pairwise MAE between raters
        per_image_mae = []
        per_image_loo = []
        per_image_std = []
        for img, group in df.groupby("_img"):
            vals = group[col].dropna().values
            if len(vals) < 2:
                continue
            per_image_std.append(float(np.std(vals)))
            # Pairwise MAE between raters
            pairwise_diffs = []
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    pairwise_diffs.append(abs(vals[i] - vals[j]))
            if pairwise_diffs:
                per_image_mae.append(float(np.mean(pairwise_diffs)))
            # LOO: |rater_i - mean(others)| — comparable to model-vs-mean MAE
            loo_diffs = []
            for i in range(len(vals)):
                others = np.delete(vals, i)
                loo_diffs.append(abs(vals[i] - np.mean(others)))
            if loo_diffs:
                per_image_loo.append(float(np.mean(loo_diffs)))
        
        results[sense] = {
            "inter_rater_mae": float(np.mean(per_image_mae)) if per_image_mae else np.nan,
            "loo_rater_mae": float(np.mean(per_image_loo)) if per_image_loo else np.nan,
            "avg_std": float(np.mean(per_image_std)) if per_image_std else np.nan,
            "n_images": len(per_image_mae),
        }
    
    # Overall
    overall_mae = np.mean([results[s]["inter_rater_mae"] for s in SENSES 
                           if not np.isnan(results[s]["inter_rater_mae"])])
    results["overall_inter_rater_mae"] = float(overall_mae)
    overall_loo = np.mean([results[s]["loo_rater_mae"] for s in SENSES 
                           if not np.isnan(results[s]["loo_rater_mae"])])
    results["overall_loo_rater_mae"] = float(overall_loo)
    return results


# ─────────────────────────────────────────────────────────
# Main Benchmark Runner
# ─────────────────────────────────────────────────────────

def run_benchmark(models_data: Dict, human_agreement: Dict) -> pd.DataFrame:
    """Run all Tier 1 metrics on all models. Returns a DataFrame."""
    rows = []
    
    for model_name, data in models_data.items():
        preds = data["preds"]
        targets = data["targets"]
        psr = parse_success_rate(preds)
        
        for sense in SENSES:
            p, t = _get_valid_pairs(preds, targets, sense)
            n = len(p)
            
            loo_rater = human_agreement.get(sense, {}).get("loo_rater_mae", np.nan)
            model_mae = raw_mae(p, t)
            agreement_ratio = model_mae / loo_rater if loo_rater > 0 else np.nan
            
            rows.append({
                "model": model_name,
                "sense": sense,
                "n": n,
                "mae": model_mae,
                "calibrated_mae": calibrated_mae(p, t),
                "rmse": raw_rmse(p, t),
                "pearson": pearson_r(p, t),
                "spearman": spearman_rho(p, t),
                "concordance": lins_ccc(p, t),
                "ordinal_acc": ordinal_accuracy(p, t),
                "within_1": within_k_accuracy(p, t, k=1.0),
                "within_05": within_k_accuracy(p, t, k=0.5),
                "pred_std": float(np.std(p)) if len(p) > 0 else np.nan,
                "parse_rate": psr,
                "loo_rater_mae": loo_rater,
                "agreement_ratio": agreement_ratio,
            })
        
        # Overall row
        sense_metrics = [r for r in rows if r["model"] == model_name and r["sense"] in SENSES]
        avg = lambda key: float(np.nanmean([r[key] for r in sense_metrics[-4:]]))
        
        rows.append({
            "model": model_name,
            "sense": "overall",
            "n": int(np.mean([r["n"] for r in sense_metrics[-4:]])),
            "mae": avg("mae"),
            "calibrated_mae": avg("calibrated_mae"),
            "rmse": avg("rmse"),
            "pearson": avg("pearson"),
            "spearman": avg("spearman"),
            "concordance": avg("concordance"),
            "ordinal_acc": avg("ordinal_acc"),
            "within_1": avg("within_1"),
            "within_05": avg("within_05"),
            "pred_std": avg("pred_std"),
            "parse_rate": psr,
            "loo_rater_mae": human_agreement.get("overall_loo_rater_mae", np.nan),
            "agreement_ratio": avg("agreement_ratio"),
        })
    
    return pd.DataFrame(rows)


def print_comparison_table(df: pd.DataFrame, human_agreement: Dict):
    """Print a nicely formatted comparison table."""
    overall = df[df["sense"] == "overall"].copy()
    overall = overall.sort_values("mae")
    
    print("\n" + "=" * 120)
    print("COMPREHENSIVE SENSORY EVALUATION BENCHMARK")
    print("=" * 120)
    
    # Main comparison table
    print(f"\n{'Model':<16} {'MAE↓':>7} {'CalMAE↓':>8} {'Pearson↑':>9} {'Spearman↑':>10} {'CCC↑':>7} "
          f"{'σ_pred':>7} {'Ord.Acc↑':>9} {'W/in 1↑':>8} {'W/in.5↑':>8} {'Parse%':>7} {'Agr.Rat↓':>9}")
    print("-" * 130)
    
    for _, row in overall.iterrows():
        print(f"{row['model']:<16} {row['mae']:>7.4f} {row['calibrated_mae']:>8.4f} {row['pearson']:>9.4f} "
              f"{row['spearman']:>10.4f} {row['concordance']:>7.4f} "
              f"{row.get('pred_std', np.nan):>7.4f} {row['ordinal_acc']:>9.4f} "
              f"{row['within_1']:>8.4f} {row['within_05']:>8.4f} {row['parse_rate']*100:>6.1f}% "
              f"{row['agreement_ratio']:>9.3f}")
    
    # Per-sense breakdown for each model
    for model_name in overall["model"].values:
        print(f"\n--- {model_name} (per-sense) ---")
        model_df = df[(df["model"] == model_name) & (df["sense"] != "overall")]
        print(f"  {'Sense':<10} {'MAE':>7} {'CalMAE':>8} {'Pearson':>8} {'Spearman':>9} {'CCC':>7} "
              f"{'σ_pred':>7} {'OrdAcc':>7} {'W/in 1':>7}")
        for _, row in model_df.iterrows():
            print(f"  {row['sense']:<10} {row['mae']:>7.4f} {row['calibrated_mae']:>8.4f} {row['pearson']:>8.4f} "
                  f"{row['spearman']:>9.4f} {row['concordance']:>7.4f} {row.get('pred_std', np.nan):>7.4f} "
                  f"{row['ordinal_acc']:>7.4f} {row['within_1']:>7.4f}")
    
    # Human agreement baseline
    print(f"\n{'=' * 80}")
    print("HUMAN INTER-ANNOTATOR AGREEMENT (performance ceiling)")
    print(f"{'=' * 80}")
    print(f"  {'Sense':<10} {'Pairwise MAE':>14} {'LOO-vs-Mean':>13} {'Avg Std':>10} {'N images':>10}")
    for sense in SENSES:
        ha = human_agreement.get(sense, {})
        print(f"  {sense:<10} {ha.get('inter_rater_mae', np.nan):>14.4f} {ha.get('loo_rater_mae', np.nan):>13.4f} "
              f"{ha.get('avg_std', np.nan):>10.4f} {ha.get('n_images', 0):>10d}")
    print(f"  {'overall':<10} {human_agreement.get('overall_inter_rater_mae', np.nan):>14.4f} "
          f"{human_agreement.get('overall_loo_rater_mae', np.nan):>13.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Sensory Evaluation Benchmark")
    parser.add_argument("--human_csv", type=str, default="data/FINAL_DATASET_COMPLETE_with_rescaling.csv")
    parser.add_argument("--image_dir", type=str, default="data/Images")
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument(
        "--models",
        type=str,
        default="ours,base,food_llama,internvl,qwen2_vl,llava,v2_s2,abl_single_lr2e6",
        help="Comma-separated models to benchmark. Each must have canonical predictions at output_dir/<model>/<model>_predictions.jsonl",
    )
    args = parser.parse_args()
    
    eval_dir = Path(args.output_dir)
    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if not model_names:
        raise ValueError("No models specified for benchmarking.")
    
    # Load human data for ground truth and agreement analysis
    print("Loading human sensory data...")
    df = load_human_sensory_data(args.human_csv, args.image_dir, require_all_caninfer=True)
    train_df, val_df, test_df = create_image_level_splits(df, test_size=0.15, val_size=0.10)
    
    def img_name(row):
        sp = row.get("saved_path")
        return sp[0] if isinstance(sp, list) and sp else str(sp) if sp else ""
    
    test_df = test_df.copy()
    test_df["_img"] = test_df.apply(img_name, axis=1)
    test_images = test_df["_img"].unique().tolist()
    
    # Build ground truth map
    gt_df = test_df.groupby("_img")[["sensory_taste", "sensory_smell", "sensory_texture", "sensory_sound"]].mean()
    gt_map = {idx: row.to_dict() for idx, row in gt_df.iterrows()}
    
    # Load model predictions
    print("Loading model predictions...")
    print(f"  Requested models: {model_names}")
    models_data = load_all_model_predictions(eval_dir, gt_map, model_names)
    print(f"  Loaded predictions for: {list(models_data.keys())}")
    for name, data in models_data.items():
        print(f"    {name}: {len(data['preds'])} images, parse rate: {parse_success_rate(data['preds']):.1%}")
    
    # Build constant-mean baseline from training set (Fix #5)
    train_df_copy = train_df.copy()
    train_df_copy["_img"] = train_df_copy.apply(img_name, axis=1)
    sense_means = {
        "taste": float(train_df_copy["sensory_taste"].mean()),
        "smell": float(train_df_copy["sensory_smell"].mean()),
        "texture": float(train_df_copy["sensory_texture"].mean()),
        "sound": float(train_df_copy["sensory_sound"].mean()),
    }
    # Create synthetic predictions: every image gets the training mean
    first_model_data = next(iter(models_data.values()))
    const_preds = [dict(sense_means) for _ in first_model_data["images"]]
    models_data["const_mean"] = {
        "images": first_model_data["images"],
        "preds": const_preds,
        "targets": first_model_data["targets"],
    }
    print(f"  Added 'const_mean' baseline (training means: {sense_means})")
    
    # Compute human agreement (Tier 4)
    print("\nComputing human inter-annotator agreement...")
    human_agreement = compute_human_agreement(df, test_images)
    
    # Run benchmark
    print("Running benchmark metrics...")
    results_df = run_benchmark(models_data, human_agreement)
    
    # Print results
    print_comparison_table(results_df, human_agreement)
    
    # Save results
    out_csv = eval_dir / "benchmark_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    
    out_json = eval_dir / "benchmark_results.json"
    results_df.to_json(out_json, orient="records", indent=2)
    print(f"Results saved to {out_json}")
    
    # Save human agreement
    ha_json = eval_dir / "human_agreement.json"
    with open(ha_json, "w") as f:
        json.dump(human_agreement, f, indent=2)
    print(f"Human agreement saved to {ha_json}")


if __name__ == "__main__":
    main()
