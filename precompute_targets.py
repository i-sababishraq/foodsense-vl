"""
MAmmoTH-v2 prose expansion for sensory training targets.

Improvements over v1:
  - Redesigned prompt: 3-4 sentences/sense, varied openers, NO embedded ratings
  - bf16 precision (not 4-bit) for higher quality generation
  - Supports both Gemma and Qwen as teacher models
  - Stricter quality filters: min chars, no rating leakage, no templated openers
  - Retry logic (up to 2 retries) for failed quality checks
  - Higher max_new_tokens (768 default)
  - AdaptLLM food-Llama judge pass to filter hallucinated expansions

Output: mammoth_style_target_lookup_v2.json
  { "image_name": { "taste": "expansion prose", "smell": "...", ... } }

Usage:
  python precompute_targets.py \
    --human_csv data/FINAL_DATASET_COMPLETE_with_rescaling.csv \
    --image_dir data/Images \
    --output mammoth_style_target_lookup_v2.json \
    --model_name google/gemma-3-27b-it \
    [--max_images 50] [--resume] [--retries 2]
"""

import argparse
import gc
import json
import os
import re
import time

# Disable safetensors mmap — critical for Lustre/network filesystems (PSC Bridges-2).
# mmap causes random I/O that stalls on shared storage; sequential reads are much faster.
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ.setdefault("HF_SAFETENSORS_NO_AVX", "1")
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from dataset import load_human_sensory_data


# ---------------------------------------------------------------------------
# Prompts — redesigned for richer, more varied sensory-detail prose
# ---------------------------------------------------------------------------

EXPANSION_SYSTEM = (
    "You are a sensory analysis expert specializing in cross-modal food perception. "
    "Given a food image and human sensory ratings with descriptors, write a detailed "
    "visual justification for each sensory dimension.\n\n"
    "Rules:\n"
    "- Write 3-4 sentences per sense grounded in what you see in the image.\n"
    "- Cover specific visual cues: colors, textures, surface details, char marks, "
    "steam, garnishes, plating, portion size, ingredient visibility.\n"
    "- For Taste: use vocabulary like sweet, savory, salty, spicy, bland, fishy, "
    "fresh, meaty, cheesy, rich, flavorful, tangy, umami, buttery, sour, earthy, "
    "bitter, juicy, tart, zesty, peppery, herbal, nutty, smoky, charred, roasted, "
    "grilled, briny, citrusy, robust, mellow, sharp, acidic, piquant, creamy, "
    "delicious. Describe visual cues that indicate these flavors.\n"
    "- For Smell: use vocabulary like sweet, fishy, savory, fresh, spicy, meaty, "
    "cheesy, fried, aromatic, fruity, rich, earthy, warm, smoky, buttery, oily, "
    "pungent, fragrant, mild, floral, herbaceous, nutty, toasted, charred, briny, "
    "citrusy, yeasty, fermented, woody, musky, strong. Describe what visual "
    "elements suggest these aromas.\n"
    "- For Texture: use vocabulary like soft, crunchy, chewy, smooth, crispy, "
    "mushy, creamy, rough, hard, tender, firm, moist, gooey, juicy, flaky, "
    "grainy, thick, dry, squishy, silky, velvety, crumbly, brittle, springy, "
    "dense, airy, sticky, gelatinous, fibrous, pillowy, succulent, pasty, "
    "rubbery, soggy. Describe visible surface and structure cues.\n"
    "- For Sound: use vocabulary like crunchy, quiet, soft, crunch, squishy, "
    "chewy, slurp, crispy, loud, sizzling, moist, chewing, snap, pop, fizz, "
    "crackle, squelch, hiss, bubble, crack, rustle, silent. Describe what visual "
    "textures and structures would produce these sounds when eaten.\n"
    "- Use varied sentence structures. Do NOT start every sense with the same phrase.\n"
    "- Use diverse, natural vocabulary. AVOID overusing technical jargon like "
    "'Maillard reaction', 'caramelization', 'sheen', 'crispness', 'golden-brown', "
    "or 'glistening' — prefer everyday sensory words.\n"
    "- Do NOT include any numeric ratings (like '4.2/5.0') in the prose body.\n"
    "- Plain text only, no markdown, no bullet points."
)

EXPANSION_USER_TEMPLATE = """Analyze this food image. Human sensory assessments:
Taste: {taste_r:.1f}/5.0 ({taste_desc})
Smell: {smell_r:.1f}/5.0 ({smell_desc})
Texture: {texture_r:.1f}/5.0 ({texture_desc})
Sound: {sound_r:.1f}/5.0 ({sound_desc})

For each sense, write 3-4 sentences of detailed visual justification explaining which specific visual cues in the image support the rating and descriptor. Use this exact format:

Taste: [3-4 sentences here]
Smell: [3-4 sentences here]
Texture: [3-4 sentences here]
Sound: [3-4 sentences here]

Output only these four sections, nothing else."""


# ---------------------------------------------------------------------------
# Judge (AdaptLLM food-Llama) — filters hallucinated expansions
# ---------------------------------------------------------------------------

JUDGE_MODEL_ID = "AdaptLLM/food-Llama-3.2-11B-Vision-Instruct"

JUDGE_PROMPT = """Here is a sensory assessment of this food image:

{assessment_block}

Does this description logically match what is shown in the image? Does it avoid hallucinating visual details that are not present? Answer with only: YES or NO"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SENSE_ORDER = ["taste", "smell", "texture", "sound"]
SENSE_LABELS = ["Taste", "Smell", "Texture", "Sound"]

# Templated openers to reject
BAD_OPENERS = [
    "the visual justification",
    "the visual cues that support",
    "the visual evidence for",
    "the visual cues for",
    "based on the visual cues",
]

# Rating patterns to strip from prose
RATING_PATTERN = re.compile(r"\b\d+\.?\d*/5\.0\b")
RATING_PHRASE_PATTERN = re.compile(
    r"(The visual (justification|evidence|cues) for the \w+ rating of [0-9.]+/5\.0"
    r"(?: for (?:the )?\w+)? (?:is|are)\s*)",
    re.IGNORECASE,
)
RATING_SUPPORT_PATTERN = re.compile(
    r"(The visual cues that support the \w+ rating of [0-9.]+/5\.0"
    r" (?:for this image )?(?:are|is)\s*)",
    re.IGNORECASE,
)


def _format_desc(desc: str, default: str = "pleasing") -> str:
    """Use descriptor or placeholder if empty."""
    s = (desc or "").strip().lower()
    if s in {"", "nan", "not sure", "none", "n/a", "idk", "unsure"}:
        return default
    return desc.strip()


def _sanitize_prose(text: str) -> str:
    """Strip rating references and templated phrases from prose body."""
    if not text:
        return text
    text = RATING_PHRASE_PATTERN.sub("", text)
    text = RATING_SUPPORT_PATTERN.sub("", text)
    text = RATING_PATTERN.sub("", text)
    # Clean up double spaces and leading punctuation
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"^[,;:\s]+", "", text).strip()
    return text


# Terms that Gemma tends to overuse — trigger retry if too many appear
OVERUSED_TERMS = [
    "maillard", "caramelization", "caramelized", "golden-brown",
    "sheen", "glistening", "crispness", "brittleness", "rendered fat",
]
MAX_OVERUSED_PER_BLOCK = 2  # allow up to 2 per sense block; reject if more


def _check_quality(prose: str, min_chars: int = 100) -> Tuple[bool, str]:
    """Check if prose meets quality requirements. Returns (pass, reason)."""
    if not prose or len(prose.strip()) < min_chars:
        return False, f"too_short ({len(prose.strip()) if prose else 0} < {min_chars})"
    lower = prose.lower().strip()
    for opener in BAD_OPENERS:
        if lower.startswith(opener):
            return False, f"templated_opener: {opener}"
    if RATING_PATTERN.search(prose):
        return False, "rating_leakage"
    # Must have at least 2 sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', prose) if s.strip()]
    if len(sentences) < 2:
        return False, f"too_few_sentences ({len(sentences)})"
    # Check overused terms
    overused_count = sum(1 for term in OVERUSED_TERMS if term in lower)
    if overused_count > MAX_OVERUSED_PER_BLOCK:
        return False, f"overused_terms ({overused_count} > {MAX_OVERUSED_PER_BLOCK})"
    return True, "ok"


def _parse_v2_output(text: str) -> Dict[str, str]:
    """
    Parse model output to extract per-sense expansion prose.
    v2 format: "Taste: [prose]\nSmell: [prose]\n..."
    Also handles v1 format: "Taste (X.X/5.0): desc. [prose]"
    """
    result = {}
    block = text.strip()

    for i, (sense, label) in enumerate(zip(SENSE_ORDER, SENSE_LABELS)):
        # Try v2 format first: "Taste: prose..."
        # Then v1 format: "Taste (4.2/5.0): desc. prose..."
        patterns = [
            # v2: "Taste: prose" (no rating in header)
            rf"{label}\s*:\s*(.+?)(?=\n(?:Taste|Smell|Texture|Sound)\s*[:(]|\Z)",
            # v1: "Taste (X.X/5.0): desc. prose"
            rf"{label}\s*\([0-9.]+/5\.0\)\s*:\s*(.+?)(?=\n(?:Taste|Smell|Texture|Sound)\s*[:(]|\Z)",
        ]
        matched = False
        for pat in patterns:
            m = re.search(pat, block, re.IGNORECASE | re.DOTALL)
            if m:
                content = m.group(1).strip()
                # If v1 format, strip leading "descriptor. " (1-4 words before first ". ")
                if "5.0)" in pat:
                    desc_match = re.match(r"^([^.]{1,40})\.\s+(.+)$", content, re.DOTALL)
                    if desc_match:
                        content = desc_match.group(2).strip()
                # Sanitize
                content = _sanitize_prose(content)
                if content and content[-1] not in ".!?":
                    content += "."
                result[sense] = content
                matched = True
                break
        if not matched:
            result[sense] = ""

    return result


def _first_nonempty(series: pd.Series) -> str:
    for v in series:
        s = str(v or "").strip().lower()
        if s and s not in {"nan", "not sure", "none", "n/a"}:
            return str(v).strip()
    return ""


def get_image_level_seed(df: pd.DataFrame) -> pd.DataFrame:
    """One row per image: mean ratings, first non-empty descs."""
    df = df.copy()
    df["_img"] = df["saved_path"].apply(
        lambda x: x[0] if isinstance(x, list) and x else ""
    )
    agg = df.groupby("_img").agg(
        sensory_taste=("sensory_taste", "mean"),
        sensory_smell=("sensory_smell", "mean"),
        sensory_texture=("sensory_texture", "mean"),
        sensory_sound=("sensory_sound", "mean"),
        taste_desc=("taste_desc", lambda s: _first_nonempty(s)),
        smell_desc=("smell_desc", lambda s: _first_nonempty(s)),
        texture_desc=("texture_desc", lambda s: _first_nonempty(s)),
        sound_desc=("sound_desc", lambda s: _first_nonempty(s)),
    ).reset_index()
    agg = agg.rename(columns={"_img": "image_name"})
    return agg


def _build_assessment_block_v2(
    taste_r: float, smell_r: float, texture_r: float, sound_r: float,
    taste_d: str, smell_d: str, texture_d: str, sound_d: str,
    expansions: Dict[str, str],
) -> str:
    """Build full sensory assessment text for judge evaluation."""
    ratings = [taste_r, smell_r, texture_r, sound_r]
    descs = [taste_d, smell_d, texture_d, sound_d]
    lines = []
    for i, (sense, label) in enumerate(zip(SENSE_ORDER, SENSE_LABELS)):
        exp = (expansions.get(sense) or "").strip()
        body = f"{descs[i]}. {exp}" if exp else descs[i]
        if body and body[-1] not in ".!?":
            body += "."
        lines.append(f"{label} ({ratings[i]:.1f}/5.0): {body}")
    return "Sensory Assessment:\n" + "\n".join(lines)


def _judge_expansion(
    judge_model,
    judge_processor,
    image: Image.Image,
    assessment_block: str,
    max_new_tokens: int = 512,
    lenient: bool = True,
) -> Tuple[bool, str]:
    """Run food-Llama judge; return (True if passes, raw response text)."""
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": JUDGE_PROMPT.format(assessment_block=assessment_block)},
        ]}
    ]
    try:
        input_text = judge_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = judge_processor(image, input_text, add_special_tokens=False, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(judge_model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.no_grad():
            out = judge_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = judge_processor.decode(out[0][prompt_len:], skip_special_tokens=True)
        raw = text.strip()
        lower = raw.lower()
        first_word = (raw.split() or [""])[0].lower()
        if first_word.startswith("yes"):
            return True, raw
        if first_word.startswith("no"):
            return False, raw
        if lenient:
            tail = lower[-200:] if len(lower) > 200 else lower
            # Check for explicit "The answer is: YES/NO"
            ans_match = re.search(r'\bthe answer is:\s*(yes|no)\b', tail)
            if ans_match:
                return ans_match.group(1) == "yes", raw
            # Check for YES/NO as standalone words in tail
            if re.search(r'\byes\b', tail):
                return True, raw
            # food-Llama outputs long reasoning then concludes after "Therefore".
            # The CONCLUSION (last sentence) is the verdict — check it first.
            conclusion = tail
            therefore_pos = tail.rfind("therefore")
            if therefore_pos >= 0:
                conclusion = tail[therefore_pos:]
            affirmative_phrases = [
                r'logically matches',
                r'matches what is shown',
                r'grounded in the visual',
                r'accurate .{0,30}sensory assessment',
                r'consistent with what is shown',
                r'does not hallucinate',
                r'avoids hallucinating',
                r'without .{0,30}hallucinating',
                r'comprehensive .{0,30}sensory (assessment|profile)',
                r'logical and (reasonable|grounded)',
                r'a logical .{0,30}interpretation',
            ]
            negative_phrases = [
                r'not entirely logical',
                r'does not .{0,20}match',
                r'not .{0,15}logically match',
                r'includes .{0,20}details that are not',
            ]
            # Check conclusion for verdict
            for pat in negative_phrases:
                if re.search(pat, conclusion):
                    return False, raw
            for pat in affirmative_phrases:
                if re.search(pat, conclusion):
                    return True, raw
            # Fallback: bare "no" in tail (only if no affirmative conclusion)
            if re.search(r'\bno\b', tail):
                return False, raw
            # Fallback: first 80 chars
            s = lower[:80]
            yes_pos, no_pos = s.find("yes"), s.find("no")
            if yes_pos >= 0 and (no_pos < 0 or yes_pos < no_pos):
                return True, raw
        return False, raw
    except Exception as e:
        return False, str(e)


def _load_model_and_processor(model_name: str, precision: str = "bf16"):
    """Load model and processor. Supports Gemma and Qwen architectures."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True
    )
    if getattr(processor, "tokenizer", None) and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    load_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    if precision == "bf16":
        pass  # default torch_dtype=bfloat16
    elif precision == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Try flash_attention_2 for Gemma; Qwen may not support it the same way
    is_qwen = "qwen" in model_name.lower()
    if not is_qwen:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    return model, processor


def _generate_expansion(
    model, processor, image: Image.Image,
    taste_r: float, smell_r: float, texture_r: float, sound_r: float,
    taste_d: str, smell_d: str, texture_d: str, sound_d: str,
    max_new_tokens: int = 768,
    is_qwen: bool = False,
) -> str:
    """Generate expansion prose for one image."""
    user_text = EXPANSION_USER_TEMPLATE.format(
        taste_r=taste_r, smell_r=smell_r, texture_r=texture_r, sound_r=sound_r,
        taste_desc=taste_d, smell_desc=smell_d, texture_desc=texture_d, sound_desc=sound_d,
    )

    if is_qwen:
        # Qwen uses a different message format for images
        messages = [
            {"role": "system", "content": EXPANSION_SYSTEM},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
        ]
        text_input = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=[text_input], images=[image],
            return_tensors="pt", padding=True
        )
    else:
        # Gemma format
        messages = [
            {"role": "system", "content": [{"type": "text", "text": EXPANSION_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    text = processor.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return text


def main():
    parser = argparse.ArgumentParser(description="MAmmoTH-v2 prose expansion")
    parser.add_argument("--human_csv", type=str, default="data/FINAL_DATASET_COMPLETE_with_rescaling.csv")
    parser.add_argument("--image_dir", type=str, default="data/Images", help="Root directory containing all images.")
    parser.add_argument("--output", type=str, default="mammoth_style_target_lookup_v2.json")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "4bit"])
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images (for pilot)")
    parser.add_argument("--resume", action="store_true", help="Skip images already in output")
    parser.add_argument("--reverse", action="store_true", help="Process from end to start")
    parser.add_argument("--stop_at", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--retries", type=int, default=2, help="Max retries per image on quality failure")
    parser.add_argument("--min_chars_per_sense", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--pilot_compare", action="store_true",
                        help="Print detailed quality stats for pilot comparison")
    parser.add_argument("--judge_model", type=str, default=JUDGE_MODEL_ID,
                        help="Model-as-judge for filtering hallucinated expansions")
    parser.add_argument("--no_judge", action="store_true",
                        help="Skip model-as-judge filtering")
    parser.add_argument("--judge_lenient", action="store_true", default=True,
                        help="Accept 'yes' anywhere in response (food-Llama outputs long reasoning)")
    parser.add_argument("--judge_debug", action="store_true",
                        help="Print first 20 raw judge responses for debugging")
    parser.add_argument("--chunk_idx", type=int, default=None,
                        help="0-based chunk index for parallel array jobs")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Total number of chunks for parallel array jobs")
    args = parser.parse_args()

    is_qwen = "qwen" in args.model_name.lower()

    # Load human data
    df = load_human_sensory_data(args.human_csv, args.image_dir, require_all_caninfer=True)
    seed_df = get_image_level_seed(df)
    image_names = seed_df["image_name"].tolist()

    if args.max_images:
        image_names = image_names[:args.max_images]
    if args.stop_at is not None:
        if args.reverse:
            image_names = image_names[args.stop_at:][::-1]
        else:
            image_names = image_names[:args.stop_at]
    elif args.reverse:
        image_names = list(reversed(image_names))

    # Chunking for parallel array jobs
    if args.chunk_idx is not None and args.num_chunks is not None:
        total = len(image_names)
        chunk_size = (total + args.num_chunks - 1) // args.num_chunks
        start = args.chunk_idx * chunk_size
        end = min(start + chunk_size, total)
        image_names = image_names[start:end]
        print(f"Chunk {args.chunk_idx}/{args.num_chunks}: images [{start}:{end}] ({len(image_names)} images)")

    print(f"MAmmoTH-v2: {len(image_names):,} images, teacher={args.model_name}, precision={args.precision}")

    # Resumable output
    out_path = Path(args.output)
    lookup = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            lookup = json.load(f)
        print(f"Resuming: {len(lookup):,} images already in {out_path}")

    # Load model
    print(f"Loading model: {args.model_name} ({args.precision})...")
    model, processor = _load_model_and_processor(args.model_name, args.precision)
    print("Model loaded.")

    image_dir = Path(args.image_dir)
    seed_idx = seed_df.set_index("image_name")

    # Quality tracking
    stats = {"total": 0, "pass": 0, "retry_pass": 0, "fail": 0,
             "chars_per_sense": [], "sentences_per_sense": []}

    t0 = time.time()
    for i, img_name in enumerate(image_names):
        if img_name in lookup and lookup[img_name]:
            # Check if existing entry has all 4 senses with content
            existing = lookup[img_name]
            if all(existing.get(s) for s in SENSE_ORDER):
                continue

        img_path = image_dir / img_name
        if not img_path.exists():
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        row = seed_idx.loc[img_name]
        taste_r = float(row["sensory_taste"])
        smell_r = float(row["sensory_smell"])
        texture_r = float(row["sensory_texture"])
        sound_r = float(row["sensory_sound"])
        taste_d = _format_desc(str(row["taste_desc"]), "pleasing")
        smell_d = _format_desc(str(row["smell_desc"]), "aromatic")
        texture_d = _format_desc(str(row["texture_desc"]), "smooth")
        sound_d = _format_desc(str(row["sound_desc"]), "quiet")

        best_parsed = None
        best_pass_count = -1

        for attempt in range(1 + args.retries):
            raw_text = _generate_expansion(
                model, processor, image,
                taste_r, smell_r, texture_r, sound_r,
                taste_d, smell_d, texture_d, sound_d,
                max_new_tokens=args.max_new_tokens,
                is_qwen=is_qwen,
            )
            parsed = _parse_v2_output(raw_text)

            # Check quality for each sense
            pass_count = 0
            for sense in SENSE_ORDER:
                ok, reason = _check_quality(parsed.get(sense, ""), args.min_chars_per_sense)
                if ok:
                    pass_count += 1

            if pass_count > best_pass_count:
                best_pass_count = pass_count
                best_parsed = parsed

            if pass_count == 4:
                if attempt > 0:
                    stats["retry_pass"] += 1
                break

        stats["total"] += 1
        if best_pass_count == 4:
            stats["pass"] += 1
        else:
            stats["fail"] += 1

        # Track per-sense stats
        for sense in SENSE_ORDER:
            prose = best_parsed.get(sense, "")
            stats["chars_per_sense"].append(len(prose))
            sentences = [s.strip() for s in re.split(r'[.!?]+', prose) if s.strip()]
            stats["sentences_per_sense"].append(len(sentences))

        lookup[img_name] = best_parsed

        # Progress + save
        if (i + 1) % args.save_every == 0 or i == len(image_names) - 1:
            with open(out_path, "w") as f:
                json.dump(lookup, f, indent=2)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(image_names) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:>5}/{len(image_names)}] saved={len(lookup):,} "
                f"pass={stats['pass']}/{stats['total']} "
                f"retry_pass={stats['retry_pass']} fail={stats['fail']} "
                f"rate={rate:.2f} img/s  ETA={eta/60:.0f}m"
            )

    # Final save
    with open(out_path, "w") as f:
        json.dump(lookup, f, indent=2)

    # Print quality summary
    print(f"\n{'='*60}")
    print(f"MAmmoTH-v2 Generation Complete")
    print(f"{'='*60}")
    print(f"Teacher: {args.model_name} ({args.precision})")
    print(f"Images processed: {stats['total']}")
    print(f"Pass (4/4 senses): {stats['pass']} ({100*stats['pass']/max(1,stats['total']):.1f}%)")
    print(f"Retry recoveries: {stats['retry_pass']}")
    print(f"Fail (<4 senses): {stats['fail']} ({100*stats['fail']/max(1,stats['total']):.1f}%)")

    if stats["chars_per_sense"]:
        chars = np.array(stats["chars_per_sense"])
        sents = np.array(stats["sentences_per_sense"])
        print(f"Chars/sense: mean={chars.mean():.0f}, median={np.median(chars):.0f}, "
              f"min={chars.min()}, max={chars.max()}")
        print(f"Sentences/sense: mean={sents.mean():.1f}, median={np.median(sents):.0f}")

    if args.pilot_compare:
        print(f"\n--- Pilot Quality Details ---")
        for img_name in list(lookup.keys())[:10]:
            entry = lookup[img_name]
            print(f"\n  Image: {img_name}")
            for sense in SENSE_ORDER:
                prose = entry.get(sense, "")
                ok, reason = _check_quality(prose, args.min_chars_per_sense)
                status = "PASS" if ok else f"FAIL({reason})"
                print(f"    {sense}: [{status}] {len(prose)} chars | {prose[:120]}...")

    # ---- Judge pass (model-as-judge, food-Llama) ----
    if not args.no_judge and lookup:
        print(f"\nRunning judge ({args.judge_model}) to filter hallucinations...")

        # Free teacher model memory BEFORE loading judge to avoid OOM
        try:
            del model, processor
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        from transformers import AutoProcessor as AP2
        try:
            from transformers import MllamaForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForCausalLM as MllamaForConditionalGeneration
        judge_model = MllamaForConditionalGeneration.from_pretrained(
            args.judge_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        judge_processor = AP2.from_pretrained(args.judge_model, trust_remote_code=True, local_files_only=True)
        judge_model.eval()

        rejected = 0
        judge_total = 0
        for j, img_name in enumerate(list(lookup.keys())):
            entry = lookup[img_name]
            if not entry or not any(entry.get(s) for s in SENSE_ORDER):
                continue
            img_path = image_dir / img_name
            if not img_path.exists():
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            row = seed_idx.loc[img_name]
            block = _build_assessment_block_v2(
                float(row["sensory_taste"]), float(row["sensory_smell"]),
                float(row["sensory_texture"]), float(row["sensory_sound"]),
                _format_desc(str(row["taste_desc"]), "pleasing"),
                _format_desc(str(row["smell_desc"]), "aromatic"),
                _format_desc(str(row["texture_desc"]), "smooth"),
                _format_desc(str(row["sound_desc"]), "quiet"),
                entry,
            )
            judge_retries = 2
            passed, raw = False, ""
            for _jt in range(judge_retries):
                passed, raw = _judge_expansion(judge_model, judge_processor, image, block, lenient=args.judge_lenient)
                if passed or not raw.startswith("Traceback"):
                    break  # got a real answer (pass or fail), stop retrying
            if args.judge_debug and j < 20:
                print(f"  [Judge #{j+1}] raw={repr(raw[:80])} passed={passed}")
            judge_total += 1
            if not passed:
                lookup[img_name]["judge_rejected"] = True
                lookup[img_name]["judge_reason"] = raw
                rejected += 1
            else:
                lookup[img_name]["judge_passed"] = True
            if (j + 1) % 50 == 0:
                print(f"  Judge: {j + 1:,} checked, {rejected} rejected")
        print(f"  Judge complete: {rejected}/{judge_total} expansions rejected")

        # Final save after judge
        with open(out_path, "w") as f:
            json.dump(lookup, f, indent=2)

    print(f"\nOutput: {out_path} ({len(lookup):,} images)")
    elapsed_total = time.time() - t0
    print(f"Total time: {elapsed_total/60:.1f} min ({elapsed_total/3600:.1f} h)")


if __name__ == "__main__":
    main()
