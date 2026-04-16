"""
Sensory Evaluation Pipeline: Compare fine-tuned model vs baselines on test set.

Loads human-annotated test split, runs inference with Gemma (fine-tuned + base)
and optionally API models (GPT-4o, Gemini, Claude), parses sensory ratings,
computes MAE/RMSE/Pearson per sense, and saves results.

Usage:
  python evaluate.py --adapter_dir checkpoints/.../checkpoint-200
  python evaluate.py --adapter_dir ... --models qwen2_vl,llava,internvl --max_images 50
  python evaluate.py --adapter_dir ... --models ours,base --split val  # validation set
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.prompts import SYSTEM_PROMPT, USER_PROMPT
from dataset import create_image_level_splits, load_human_sensory_data


SENSES = ["taste", "smell", "texture", "sound"]
SENSE_DIMS = ["Taste", "Smell", "Texture", "Sound"]


def parse_ratings(text: str) -> Dict[str, float]:
    """Extract Taste, Smell, Texture, Sound ratings from model output."""
    ratings = {}
    # Strip markdown formatting: bold (**), headers (#), dashes used as bullets
    clean = text.replace("*", "").replace("#", "")
    _NUM = r'([0-9]+(?:\.[0-9]+)?)'
    for dim in SENSE_DIMS:
        # Try several formats models use:
        #   Taste (3.8/5.0): ...    Taste (3.85/5.0): ...
        m = re.search(rf"{dim}\s*\({_NUM}/5(?:\.0)?\)", clean)
        #   Taste: 3.8/5.0  or  Taste: 3.8
        if not m:
            m = re.search(rf"{dim}\s*[:–\-]\s*{_NUM}(?:/5(?:\.0)?)?", clean)
        #   Taste = 3.8
        if not m:
            m = re.search(rf"{dim}\s*=\s*{_NUM}", clean)
        if m:
            val = float(m.group(1))
            if val <= 5.0:  # sanity check
                ratings[dim.lower()] = val
    return ratings


def _safe_float(v) -> float:
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def compute_metrics(
    preds: List[Dict[str, float]],
    targets: List[Dict[str, float]],
) -> Dict[str, Any]:
    """Compute MAE, RMSE, Pearson per sense and overall."""
    out = {}
    for sense in SENSES:
        p = np.array([_safe_float(x.get(sense)) for x in preds])
        t = np.array([_safe_float(x.get(sense)) for x in targets])
        valid = ~np.isnan(p) & ~np.isnan(t)
        p, t = p[valid], t[valid]
        if len(p) == 0:
            out[sense] = {"mae": np.nan, "rmse": np.nan, "pearson": np.nan, "n": 0}
            continue
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        r = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 else np.nan
        out[sense] = {"mae": mae, "rmse": rmse, "pearson": r, "n": len(p)}
    mae_list = [out[s]["mae"] for s in SENSES if not np.isnan(out[s]["mae"])]
    out["overall_mae"] = float(np.mean(mae_list)) if mae_list else np.nan
    out["overall_pearson"] = float(np.nanmean([out[s]["pearson"] for s in SENSES]))
    return out


def _require_flash_attention_2() -> str:
    """Require FlashAttention2 and fail fast if unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError("FlashAttention2 requires CUDA, but CUDA is not available.")
    try:
        from transformers.utils import is_flash_attn_2_available
    except Exception as e:
        raise RuntimeError("Could not check FlashAttention2 availability.") from e
    if not is_flash_attn_2_available():
        raise RuntimeError(
            "FlashAttention2 is not available in this environment. "
            "Install/enable flash_attn for this runtime."
        )
    return "flash_attention_2"


def _load_model_with_attention(model_cls, model_id: str, attn_impl: str, **kwargs):
    """
    Load model with a requested attention backend.
    Tries both kwargs used across transformer model classes.
    """
    try:
        return model_cls.from_pretrained(model_id, attn_implementation=attn_impl, **kwargs)
    except TypeError:
        try:
            return model_cls.from_pretrained(model_id, _attn_implementation=attn_impl, **kwargs)
        except TypeError:
            raise RuntimeError(
                f"{model_cls.__name__} does not accept attention implementation override; "
                "cannot enforce FlashAttention2."
            )


def _report_attention_impl(model_name: str, model: Any, requested_impl: str) -> None:
    """Print requested vs resolved attention implementation for debugging."""
    cfg = getattr(model, "config", None)
    resolved_impl = None
    if cfg is not None:
        resolved_impl = getattr(cfg, "_attn_implementation", None) or getattr(cfg, "attn_implementation", None)
    print(f"[{model_name}] attention_impl requested={requested_impl} resolved={resolved_impl}")


def run_gemma_inference(
    image_paths: List[str],
    adapter_dir: str,
    base_model: str,
    base_only: bool,
    prompt: str,
    max_new_tokens: int,
    image_dir: str,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    local_files_only: bool = False,
) -> List[Dict[str, Any]]:
    """Run Gemma inference and return list of {image, text, ratings}."""
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = {"": 0} if device.type == "cuda" else "cpu"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    attn_impl = _require_flash_attention_2()

    proc_path = adapter_dir
    if adapter_dir is None or not any((Path(adapter_dir) / n).exists() for n in ["preprocessor_config.json", "processor_config.json", "tokenizer_config.json"]):
        proc_path = base_model
    processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True, local_files_only=local_files_only)
    if getattr(processor, "tokenizer", None) and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        local_files_only=local_files_only,
    )
    if not base_only:
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    else:
        model = base
    model.eval()
    _report_attention_impl("gemma", model, attn_impl)

    results = []
    img_dir = Path(image_dir)

    for img_path in tqdm(image_paths, desc="Gemma inference"):
        if isinstance(img_path, (list, tuple)):
            img_name = img_path[0] if img_path else ""
        else:
            img_name = str(img_path)
        full_path = img_dir / img_name if not Path(img_name).is_absolute() else Path(img_name)
        if not full_path.exists():
            results.append({"image": img_name, "text": "", "ratings": {}})
            continue
        img = Image.open(full_path).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v) and k != "token_type_ids"}
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        text = processor.tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True).strip()
        ratings = parse_ratings(text)
        results.append({"image": img_name, "text": text, "ratings": ratings})

    return results


def run_gpt4o_vision(image_path: str, prompt: str, model: str = "gpt-4o") -> str:
    """Run OpenAI GPT-4o vision on one image."""
    import base64
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def run_gemini_vision(image_path: str, prompt: str) -> str:
    """Run Google Gemini vision on one image."""
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
    img = genai.upload_file(image_path)
    response = model.generate_content([img, prompt])
    return response.text or ""


def run_claude_vision(image_path: str, prompt: str) -> str:
    """Run Anthropic Claude vision on one image."""
    import base64
    import anthropic
    client = anthropic.Anthropic()
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return msg.content[0].text if msg.content else ""


# Open-source VLMs (HuggingFace) - ~27B parameter range
FOOD_LLAMA_ID = "AdaptLLM/food-Llama-3.2-11B-Vision-Instruct"
QWEN2_VL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"  # 32B
LLAVA_ID = "llava-hf/llava-v1.6-34b-hf"  # 34B
INTERNVL_ID = "OpenGVLab/InternVL2_5-26B"  # 26B
PHI4_ID = "microsoft/Phi-4-multimodal-instruct"


def run_food_llama_inference(
    image_paths: List[str],
    prompt: str,
    max_new_tokens: int,
    image_dir: str,
    model_id: str = FOOD_LLAMA_ID,
    local_files_only: bool = False,
) -> List[Dict[str, Any]]:
    """Run AdaptLLM food-Llama (Llama 3.2 11B Vision) on images."""
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
    model.eval()

    results = []
    img_dir = Path(image_dir)
    for img_path in tqdm(image_paths, desc="food_llama inference"):
        img_name = img_path[0] if isinstance(img_path, (list, tuple)) else str(img_path)
        full_path = img_dir / img_name if not Path(img_name).is_absolute() else Path(img_name)
        if not full_path.exists():
            results.append({"image": img_name, "text": "", "ratings": {}})
            continue
        img = Image.open(full_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(img, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        ratings = parse_ratings(text)
        results.append({"image": img_name, "text": text, "ratings": ratings})
    return results


def run_qwen2vl_inference(
    image_paths: List[str],
    prompt: str,
    max_new_tokens: int,
    image_dir: str,
    model_id: str = QWEN2_VL_ID,
    do_sample: bool = False,
    temperature: float = 0.0,
    local_files_only: bool = False,
) -> List[Dict[str, Any]]:
    """Run Qwen2.5-VL-32B on images."""
    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_impl = _require_flash_attention_2()
    model = _load_model_with_attention(
        Qwen2_5_VLForConditionalGeneration,
        model_id,
        attn_impl,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
    model.eval()
    _report_attention_impl("qwen2_vl", model, attn_impl)

    results = []
    img_dir = Path(image_dir)
    for img_path in tqdm(image_paths, desc="qwen2_vl inference"):
        img_name = img_path[0] if isinstance(img_path, (list, tuple)) else str(img_path)
        full_path = img_dir / img_name if not Path(img_name).is_absolute() else Path(img_name)
        if not full_path.exists():
            results.append({"image": img_name, "text": "", "ratings": {}})
            continue
        img = Image.open(full_path).convert("RGB")
        messages = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        text = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        ratings = parse_ratings(text)
        results.append({"image": img_name, "text": text, "ratings": ratings})
    return results


def run_llava_inference(
    image_paths: List[str],
    prompt: str,
    max_new_tokens: int,
    image_dir: str,
    model_id: str = LLAVA_ID,
    do_sample: bool = False,
    temperature: float = 0.0,
    local_files_only: bool = False,
) -> List[Dict[str, Any]]:
    """Run LLaVA-1.6-34B on images."""
    from transformers import LlavaNextForConditionalGeneration, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_impl = _require_flash_attention_2()
    model = _load_model_with_attention(
        LlavaNextForConditionalGeneration,
        model_id,
        attn_impl,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
    model.eval()
    _report_attention_impl("llava", model, attn_impl)

    results = []
    img_dir = Path(image_dir)
    for img_path in tqdm(image_paths, desc="llava inference"):
        img_name = img_path[0] if isinstance(img_path, (list, tuple)) else str(img_path)
        full_path = img_dir / img_name if not Path(img_name).is_absolute() else Path(img_name)
        if not full_path.exists():
            results.append({"image": img_name, "text": "", "ratings": {}})
            continue
        img = Image.open(full_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text_prompt, images=img, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        text = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        ratings = parse_ratings(text)
        results.append({"image": img_name, "text": text, "ratings": ratings})
    return results


def _internvl_load_image(image: Image.Image, image_size: int = 448, max_num: int = 12) -> torch.Tensor:
    """Preprocess image for InternVL chat (dynamic tiling)."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(sz: int):
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((sz, sz), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def find_closest_aspect_ratio(ar, ratios, w, h, sz):
        best_diff = float("inf")
        best_r = (1, 1)
        for r in ratios:
            rar = r[0] / r[1]
            diff = abs(ar - rar)
            if diff < best_diff or (diff == best_diff and w * h > 0.5 * sz * sz * r[0] * r[1]):
                best_diff = diff
                best_r = r
        return best_r

    def dynamic_preprocess(img, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        w, h = img.size
        ar = w / h
        target_ratios = sorted(
            {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
             if i * j <= max_num and i * j >= min_num},
            key=lambda x: x[0] * x[1],
        )
        r0, r1 = find_closest_aspect_ratio(ar, target_ratios, w, h, image_size)
        tw, th = image_size * r0, image_size * r1
        resized = img.resize((tw, th))
        blocks = r0 * r1
        imgs = []
        for i in range(blocks):
            x0 = (i % r0) * image_size
            y0 = (i // r0) * image_size
            box = (x0, y0, x0 + image_size, y0 + image_size)
            imgs.append(resized.crop(box))
        if use_thumbnail and blocks != 1:
            imgs.append(img.resize((image_size, image_size)))
        return imgs

    transform = build_transform(image_size)
    tiles = dynamic_preprocess(image, image_size=image_size, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values.to(torch.bfloat16)


def _ensure_internvl_generation_compat(model: Any) -> None:
    """
    InternVL remote code calls language_model.generate() inside model.chat().
    In newer transformers, some remote-code language models no longer inherit
    GenerationMixin, so generate() disappears. Patch it back at runtime.
    """
    language_model = getattr(model, "language_model", None)
    if language_model is None or hasattr(language_model, "generate"):
        return

    try:
        from transformers import GenerationMixin
    except ImportError:
        try:
            from transformers.generation.utils import GenerationMixin
        except ImportError:
            from transformers.generation import GenerationMixin

    lm_cls = language_model.__class__
    if not issubclass(lm_cls, GenerationMixin):
        patched_cls = type(f"{lm_cls.__name__}WithGenerationMixin", (lm_cls, GenerationMixin), {})
        language_model.__class__ = patched_cls

    if not hasattr(language_model, "generate"):
        raise RuntimeError(
            "InternVL language_model.generate is unavailable after compatibility patch."
        )

    # transformers >=4.50 requires language_model.generation_config to be a
    # GenerationConfig instance (not None). Patch it if missing.
    from transformers import GenerationConfig as _GC
    if getattr(language_model, "generation_config", None) is None:
        language_model.generation_config = _GC()


def run_internvl_inference(
    image_paths: List[str],
    prompt: str,
    max_new_tokens: int,
    image_dir: str,
    model_id: str = INTERNVL_ID,
    local_files_only: bool = False,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Run InternVL2.5-26B on images. Uses pixel_values + dict for generation (InternVL mutates it)."""
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=local_files_only,
        low_cpu_mem_usage=False
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False, local_files_only=local_files_only)
    _ensure_internvl_generation_compat(model)
    model.eval()

    # InternVL expects <image>\n prefix. Use dict (not GenerationConfig): chat() mutates it with eos_token_id.
    question = f"<image>\n{prompt}"
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
    }

    results = []
    img_dir = Path(image_dir)
    for img_path in tqdm(image_paths, desc="internvl inference"):
        img_name = img_path[0] if isinstance(img_path, (list, tuple)) else str(img_path)
        full_path = img_dir / img_name if not Path(img_name).is_absolute() else Path(img_name)
        if not full_path.exists():
            results.append({"image": img_name, "text": "", "ratings": {}})
            continue
        img = Image.open(full_path).convert("RGB")
        with torch.no_grad():
            try:
                dev = next(model.parameters()).device
                pixel_values = _internvl_load_image(img).to(dev)
                text = model.chat(tokenizer, pixel_values, question, generation_config)
            except Exception as e:
                if strict:
                    raise RuntimeError(f"InternVL inference failed on image '{img_name}': {e}") from e
                text = f"[InternVL error: {e}]"
        text = str(text).strip()
        ratings = parse_ratings(text)
        results.append({"image": img_name, "text": text, "ratings": ratings})
    return results


def run_phi4_inference(
    image_paths: List[str],
    prompt: str,
    max_new_tokens: int,
    image_dir: str,
    model_id: str = PHI4_ID,
    local_files_only: bool = False,
):
    """
    Run Microsoft Phi-4-multimodal-instruct on images.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor
    print(f"\n[INFO] Loading Phi-4: {model_id}")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=local_files_only,
        _attn_implementation="flash_attention_2"
    ).eval()
    
    results = []
    for img_path in tqdm(image_paths, desc="Phi-4 Inference"):
        try:
            full_path = Path(image_dir) / img_path
            image = Image.open(full_path).convert("RGB")
            
            # Use Phi-4 specific chat template syntax
            messages = [
                {"role": "user", "content": f"<|image_1|>{prompt}"}
            ]
            prompt_str = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = processor(
                text=prompt_str,
                images=image,
                return_tensors="pt"
            ).to("cuda:0")

            generation_args = {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "do_sample": False,
            }

            generate_ids = model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                **generation_args
            )

            # Generate IDs, slice off the prompt
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

            text_out = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            ratings = parse_ratings(text_out)
            results.append({
                "image": Path(img_path).name,
                "text": text_out,
                "ratings": ratings
            })
            
        except Exception as e:
            print(f"Error on {img_path}: {e}")
            results.append({
                "image": Path(img_path).name,
                "text": f"[Phi-4 error: {e}]",
                "ratings": {}
            })
            
    # Free memory
    del model
    del processor
    torch.cuda.empty_cache()
    
    return results


def run_api_model(
    image_paths: List[str],
    model_name: str,
    prompt: str,
    image_dir: Path,
) -> List[Dict[str, Any]]:
    """Run API model on image list. Returns list of {image, text, ratings}."""
    results = []
    img_dir = Path(image_dir) if not isinstance(image_dir, Path) else image_dir
    for img_path in tqdm(image_paths, desc=f"{model_name} inference"):
        img_name = img_path[0] if isinstance(img_path, (list, tuple)) else str(img_path)
        full_path = img_dir / img_name if not Path(img_name).is_absolute() else Path(img_name)
        if not full_path.exists():
            results.append({"image": img_name, "text": "", "ratings": {}})
            continue
        try:
            if model_name == "gpt4o":
                text = run_gpt4o_vision(str(full_path), prompt)
            elif model_name == "gpt4o-mini":
                text = run_gpt4o_vision(str(full_path), prompt, model="gpt-4o-mini")
            elif model_name == "gemini":
                text = run_gemini_vision(str(full_path), prompt)
            elif model_name == "claude":
                text = run_claude_vision(str(full_path), prompt)
            else:
                text = ""
        except Exception as e:
            text = f"[API error: {e}]"
        ratings = parse_ratings(text)
        results.append({"image": img_name, "text": text, "ratings": ratings})
    return results


def main():
    parser = argparse.ArgumentParser(description="Sensory evaluation: compare models on test set")
    parser.add_argument("--human_csv", type=str, default="data/FINAL_DATASET_COMPLETE_with_rescaling.csv")
    parser.add_argument("--image_dir", type=str, default="data/Images")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Fine-tuned adapter (for ours, base)")
    parser.add_argument("--base_model", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--models", type=str, default="qwen2_vl,llava,internvl", help="Comma-sep: ours,food_llama,qwen2_vl,llava,internvl,phi4,gpt4o,gemini,claude")
    parser.add_argument("--model", type=str, default=None, help="Single model (overrides --models for parallel jobs)")
    parser.add_argument("--phi4", action="store_true", help="Run Microsoft Phi-4-multimodal-instruct")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--image_ids", type=str, default=None, help="Comma-separated image filenames (e.g. 0002_01zZe....jpg,0005_08Eu....jpg) to run on specific images across any split")
    parser.add_argument("--max_images", type=int, default=None, help="Limit test images (debug)")
    parser.add_argument("--start_idx", type=int, default=None, help="Start image index (for sharded/parallel runs)")
    parser.add_argument("--end_idx", type=int, default=None, help="End image index (exclusive)")
    parser.add_argument("--partial_output", type=str, default=None, help="Save partial results for merging (path)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # Decoding config (default: deterministic/greedy for stable MAE comparisons)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling decoding (non-deterministic).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument("--prompt", type=str, default=None, help="Custom evaluation prompt.")
    parser.add_argument("--strict", action="store_true", help="Fail immediately if any requested model run errors.")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if args.model:
        models = [args.model.strip().lower()]
    if args.phi4:
        models.append("phi4")
    if "ours" in models:
        if not args.adapter_dir:
            parser.error("--adapter_dir required when using ours")
        if not Path(args.adapter_dir).exists():
            parser.error(f"Adapter dir not found: {args.adapter_dir}")

    prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else USER_PROMPT

    # Load data
    print("Loading human sensory data...")
    df = load_human_sensory_data(args.human_csv, args.image_dir, require_all_caninfer=True)
    train_df, val_df, test_df = create_image_level_splits(df, test_size=0.15, val_size=0.10)

    # Unique images (one row per image for inference)
    def img_name(row):
        sp = row.get("saved_path")
        return sp[0] if isinstance(sp, list) and sp else str(sp) if sp else ""

    if args.image_ids:
        # Specific images: use full df for ground truth (images may be in train/val/test)
        unique_images = [x.strip() for x in args.image_ids.split(",") if x.strip()]
        eval_df = df.copy()
    else:
        eval_df = val_df if args.split == "val" else test_df
        eval_df = eval_df.copy()
        eval_df["_img"] = eval_df.apply(img_name, axis=1)
        unique_images = eval_df["_img"].unique().tolist()
        if args.max_images:
            unique_images = unique_images[: args.max_images]

    eval_df = eval_df.copy()
    eval_df["_img"] = eval_df.apply(img_name, axis=1)
    n_total = len(unique_images)
    # Apply image range for sharded/parallel runs (only when not using image_ids)
    if not args.image_ids:
        start_idx = args.start_idx or 0
        end_idx = args.end_idx or n_total
        unique_images = unique_images[start_idx:end_idx]
    print(f"Test images: {len(unique_images)} (range {0}:{len(unique_images)} of {n_total})")

    # Ground truth: per-image mean
    gt_df = eval_df.groupby("_img")[["sensory_taste", "sensory_smell", "sensory_texture", "sensory_sound"]].mean()
    gt_map = {idx: row.to_dict() for idx, row in gt_df.iterrows()}
    targets = []
    for img in unique_images:
        row = gt_map.get(img, {})
        targets.append({
            "taste": row.get("sensory_taste"),
            "smell": row.get("sensory_smell"),
            "texture": row.get("sensory_texture"),
            "sound": row.get("sensory_sound"),
        })

    img_dir = Path(args.image_dir)
    full_paths = []
    for img in unique_images:
        p = img_dir / img if not (img_dir / img).is_absolute() else Path(img)
        full_paths.append(str(p) if p.exists() else img)

    all_results = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_partial = args.partial_output is not None

    def _resolve_model_out_dir(model_name: str) -> Path:
        """Use canonical per-model output dir without accidental double nesting."""
        return output_dir if output_dir.name == model_name else output_dir / model_name

    def _save_raw_preds(model_name: str, preds: List[Dict], out_dir: Path) -> None:
        """Save raw text predictions for debugging."""
        raw_path = out_dir / f"{model_name}_predictions.jsonl"
        with open(raw_path, "w") as f:
            for p in preds:
                f.write(json.dumps({"image": p["image"], "text": p.get("text", ""), "ratings": p.get("ratings", {})}) + "\n")
        print(f"  Raw predictions saved to {raw_path}")

    def _save_partial(model_name: str, pred_ratings: List[Dict], metrics: Dict[str, Any]) -> None:
        """Save partial results for later merging."""
        out_path = Path(args.partial_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": model_name,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "n_total": n_total,
            "images": unique_images,
            "preds": pred_ratings,
            "targets": targets,
            "metrics": metrics,
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Partial output saved to {out_path}")

    # Run Gemma models
    if "ours" in models:
        print("\n--- Running fine-tuned model (ours) ---")
        preds = run_gemma_inference(
            unique_images,
            args.adapter_dir,
            args.base_model,
            base_only=False,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            image_dir=args.image_dir,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            local_files_only=args.local_files_only,
        )
        if not is_partial:
            model_out_dir = _resolve_model_out_dir("ours")
            model_out_dir.mkdir(parents=True, exist_ok=True)
            _save_raw_preds("ours", preds, model_out_dir)
        pred_ratings = [r["ratings"] for r in preds]
        metrics = compute_metrics(pred_ratings, targets)
        all_results["ours"] = {"metrics": metrics, "preds": pred_ratings}
        print(f"  Overall MAE: {metrics['overall_mae']:.4f}  RMSE(avg): {np.nanmean([metrics[s]['rmse'] for s in SENSES]):.4f}  Pearson: {metrics['overall_pearson']:.4f}")
        if is_partial:
            _save_partial("ours", pred_ratings, metrics)
            return

    if "base" in models:
        print("\n--- Running base model (no adapter) ---")
        preds = run_gemma_inference(
            unique_images,
            args.adapter_dir or "",
            args.base_model,
            base_only=True,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            image_dir=args.image_dir,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            local_files_only=args.local_files_only,
        )
        if not is_partial:
            model_out_dir = _resolve_model_out_dir("base")
            model_out_dir.mkdir(parents=True, exist_ok=True)
            _save_raw_preds("base", preds, model_out_dir)
        pred_ratings = [r["ratings"] for r in preds]
        metrics = compute_metrics(pred_ratings, targets)
        all_results["base"] = {"metrics": metrics, "preds": pred_ratings}
        print(f"  Overall MAE: {metrics['overall_mae']:.4f}  RMSE(avg): {np.nanmean([metrics[s]['rmse'] for s in SENSES]):.4f}  Pearson: {metrics['overall_pearson']:.4f}")
        if is_partial:
            _save_partial("base", pred_ratings, metrics)
            return

    # Run open-source VLMs
    if "food_llama" in models:
        print("\n--- Running Food-Llama (AdaptLLM) ---")
        preds = run_food_llama_inference(
            unique_images,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            image_dir=args.image_dir,
            model_id=FOOD_LLAMA_ID,
            local_files_only=args.local_files_only,
        )
        if not is_partial:
            model_out_dir = _resolve_model_out_dir("food_llama")
            model_out_dir.mkdir(parents=True, exist_ok=True)
            _save_raw_preds("food_llama", preds, model_out_dir)
        pred_ratings = [r["ratings"] for r in preds]
        metrics = compute_metrics(pred_ratings, targets)
        all_results["food_llama"] = {"metrics": metrics, "preds": pred_ratings}
        print(f"  Overall MAE: {metrics['overall_mae']:.4f}  RMSE(avg): {np.nanmean([metrics[s]['rmse'] for s in SENSES]):.4f}  Pearson: {metrics['overall_pearson']:.4f}")
        if is_partial:
            _save_partial("food_llama", pred_ratings, metrics)
            return

    if "qwen2_vl" in models:
        print("\n--- Running Qwen2.5-VL-32B ---")
        try:
            preds = run_qwen2vl_inference(
                unique_images,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                image_dir=args.image_dir,
                model_id=QWEN2_VL_ID,
                do_sample=args.do_sample,
                temperature=args.temperature,
                local_files_only=args.local_files_only,
            )
            # Save raw predictions for benchmark
            model_out_dir = _resolve_model_out_dir("qwen2_vl")
            model_out_dir.mkdir(parents=True, exist_ok=True)
            _save_raw_preds("qwen2_vl", preds, model_out_dir)
            pred_ratings = [r["ratings"] for r in preds]
            metrics = compute_metrics(pred_ratings, targets)
            all_results["qwen2_vl"] = {"metrics": metrics, "preds": pred_ratings}
            print(f"  Overall MAE: {metrics['overall_mae']:.4f}  RMSE(avg): {np.nanmean([metrics[s]['rmse'] for s in SENSES]):.4f}  Pearson: {metrics['overall_pearson']:.4f}")
            if is_partial:
                _save_partial("qwen2_vl", pred_ratings, metrics)
                return
        except Exception as e:
            if args.strict:
                raise RuntimeError(f"qwen2_vl evaluation failed: {e}") from e
            print(f"  Skipped: {e}")

    if "llava" in models:
        print("\n--- Running LLaVA-v1.6-34B ---")
        try:
            preds = run_llava_inference(
                unique_images,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                image_dir=args.image_dir,
                model_id=LLAVA_ID,
                do_sample=args.do_sample,
                temperature=args.temperature,
                local_files_only=args.local_files_only,
            )
            # Save raw predictions for benchmark
            model_out_dir = _resolve_model_out_dir("llava")
            model_out_dir.mkdir(parents=True, exist_ok=True)
            _save_raw_preds("llava", preds, model_out_dir)
            pred_ratings = [r["ratings"] for r in preds]
            metrics = compute_metrics(pred_ratings, targets)
            all_results["llava"] = {"metrics": metrics, "preds": pred_ratings}
            print(f"  Overall MAE: {metrics['overall_mae']:.4f}  RMSE(avg): {np.nanmean([metrics[s]['rmse'] for s in SENSES]):.4f}  Pearson: {metrics['overall_pearson']:.4f}")
            if is_partial:
                _save_partial("llava", pred_ratings, metrics)
                return
        except Exception as e:
            if args.strict:
                raise RuntimeError(f"llava evaluation failed: {e}") from e
            print(f"  Skipped: {e}")

    if "internvl" in models:
        print("\n--- Running InternVL2.5-26B ---")
        try:
            preds = run_internvl_inference(
                unique_images,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                image_dir=args.image_dir,
                model_id=INTERNVL_ID,
                local_files_only=args.local_files_only,
                strict=args.strict,
            )
            # Save raw predictions for benchmark
            model_out_dir = _resolve_model_out_dir("internvl")
            model_out_dir.mkdir(parents=True, exist_ok=True)
            _save_raw_preds("internvl", preds, model_out_dir)
            pred_ratings = [r["ratings"] for r in preds]
            metrics = compute_metrics(pred_ratings, targets)
            all_results["internvl"] = {"metrics": metrics, "preds": pred_ratings}
            print(f"  Overall MAE: {metrics['overall_mae']:.4f}  RMSE(avg): {np.nanmean([metrics[s]['rmse'] for s in SENSES]):.4f}  Pearson: {metrics['overall_pearson']:.4f}")
            if is_partial:
                _save_partial("internvl", pred_ratings, metrics)
                return
        except Exception as e:
            if args.strict:
                raise RuntimeError(f"internvl evaluation failed: {e}") from e
            print(f"  Skipped: {e}")

    if "phi4" in models:
        print("\n--- Running Phi-4-Multimodal ---")
        try:
            preds = run_phi4_inference(
                unique_images,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                image_dir=args.image_dir,
                local_files_only=args.local_files_only,
            )
            # Save raw predictions for benchmark
            model_out_dir = _resolve_model_out_dir("phi4")
            model_out_dir.mkdir(parents=True, exist_ok=True)
            _save_raw_preds("phi4", preds, model_out_dir)
            pred_ratings = [r.get("ratings", {}) for r in preds]
            metrics = compute_metrics(pred_ratings, targets)
            all_results["phi4"] = {"metrics": metrics, "preds": pred_ratings}
            print(f"  Overall MAE: {metrics['overall_mae']:.4f}  RMSE(avg): {np.nanmean([metrics[s]['rmse'] for s in SENSES]):.4f}  Pearson: {metrics['overall_pearson']:.4f}")
            if is_partial:
                _save_partial("phi4", pred_ratings, metrics)
                return
        except Exception as e:
            if args.strict:
                raise RuntimeError(f"phi4 evaluation failed: {e}") from e
            print(f"  Skipped: {e}")

    # Run API models
    for api_name in ["gpt4o", "gpt4o-mini", "gemini", "claude"]:
        if api_name not in models:
            continue
        print(f"\n--- Running {api_name} ---")
        try:
            preds = run_api_model(unique_images, api_name, prompt, img_dir)
            pred_ratings = [r["ratings"] for r in preds]
            metrics = compute_metrics(pred_ratings, targets)
            all_results[api_name] = {"metrics": metrics, "preds": pred_ratings}
            print(f"  Overall MAE: {metrics['overall_mae']:.4f}  RMSE(avg): {np.nanmean([metrics[s]['rmse'] for s in SENSES]):.4f}  Pearson: {metrics['overall_pearson']:.4f}")
            if is_partial:
                _save_partial(api_name, pred_ratings, metrics)
                return
        except Exception as e:
            if args.strict:
                raise RuntimeError(f"{api_name} evaluation failed: {e}") from e
            print(f"  Skipped: {e}")

    # Summary table
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    header = f"{'Model':<16} {'Taste MAE':>10} {'Smell MAE':>10} {'Tex MAE':>10} {'Sound MAE':>10} {'Overall MAE':>12} {'Taste r':>10}"
    print(header)
    print("-" * 80)
    for name, data in all_results.items():
        m = data["metrics"]
        row = (
            f"{name:<16} "
            f"{m['taste']['mae']:>10.4f} "
            f"{m['smell']['mae']:>10.4f} "
            f"{m['texture']['mae']:>10.4f} "
            f"{m['sound']['mae']:>10.4f} "
            f"{m['overall_mae']:>12.4f} "
            f"{m['taste']['pearson']:>10.4f}"
        )
        print(row)

    # Save
    out_data = {
        "split": args.split,
        "n_images": len(unique_images),
        "models": {k: {"metrics": v["metrics"]} for k, v in all_results.items()},
    }
    out_json = output_dir / "eval_results.json"
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # CSV for paper
    rows = []
    for name, data in all_results.items():
        m = data["metrics"]
        for sense in SENSES:
            rows.append({
                "model": name,
                "sense": sense,
                "mae": m[sense]["mae"],
                "rmse": m[sense]["rmse"],
                "pearson": m[sense]["pearson"],
                "n": m[sense]["n"],
            })
        rows.append({"model": name, "sense": "overall", "mae": m["overall_mae"], "rmse": None, "pearson": m["overall_pearson"], "n": len(unique_images)})
    pd.DataFrame(rows).to_csv(output_dir / "eval_metrics.csv", index=False)
    print(f"Metrics CSV: {output_dir / 'eval_metrics.csv'}")


if __name__ == "__main__":
    main()
