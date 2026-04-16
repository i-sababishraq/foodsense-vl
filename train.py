"""
QLoRA Training Script for Gemma 3 27B IT on Yelp Food Data.
Uses 4-bit quantization with LoRA for efficient fine-tuning.

FIXED VERSION:
- Custom Trainer that uses compute_loss and prepare_inputs
- Proper image processing (PIL images for Gemma 3)
- Label generation for causal LM training
- Multi-task learning with rating prediction head
"""

import inspect
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path
import argparse
from PIL import Image

from dataset import create_image_level_splits, load_human_sensory_data
from config.prompts import SYSTEM_PROMPT, USER_PROMPT


# ============================================================================
# Custom Dataset for VLM Training
# ============================================================================

def _sanitize_mammoth_prose(text: str) -> str:
    """
    Strip templated phrases that embed ratings (e.g., 'The visual justification
    for the smell rating of 4.2/5.0 is...'). These contradict per-participant
    ratings in the header and cause template overfitting.
    """
    if not text or not isinstance(text, str):
        return text
    import re
    # Remove "The visual justification for the X rating of Y/5.0 [for the Z] is..."
    text = re.sub(
        r"The visual (justification|evidence|cues) for the \w+ rating of [0-9.]+/5\.0(?: for (?:the )?\w+)? is\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Remove "The visual cues that support the taste rating of 4.1/5.0 for this image are"
    text = re.sub(
        r"The visual cues that support the \w+ rating of [0-9.]+/5\.0 (?:for this image )?are\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


class GemmaVLMDataset(Dataset):
    """
    Dataset for Gemma 3 VLM training on human-annotated sensory data.
    Target = human ratings + sensory words; optionally + MAmmoTH-style expansion.
    Modes:
      human_only: Use only human ratings + descriptors (no mammoth). Best for
        learning the rating scale and avoiding template overfit.
      prefer_human: When human descriptor exists, use it only (no mammoth).
        Use mammoth only when human is empty. Mammoth is sanitized.
      full: Human + mammoth expansion (mammoth sanitized to remove embedded ratings).
    """
    
    def __init__(
        self,
        df,
        image_dir: str,
        processor,
        system_prompt: str,
        mammoth_target_lookup: Optional[Dict[str, Dict[str, str]]] = None,
        max_images_per_review: int = 5,
        max_length: int = 2048,
        human_only: bool = False,
        prefer_human: bool = False,
        user_prompt: str = USER_PROMPT,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.mammoth_target_lookup = mammoth_target_lookup or {}
        self.max_images = max_images_per_review
        self.max_length = max_length
        self.human_only = human_only
        self.prefer_human = prefer_human
        self.fixed_sense_order = False
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict:
        row = self.df.iloc[idx]
        
        # Get image paths (handle both list and single path formats). The
        # review-level DF we build stores paths in `saved_path`.
        image_paths = row.get('image_paths', row.get('saved_path', row.get('photo_id', [])))
        if isinstance(image_paths, str):
            import ast
            try:
                parsed = ast.literal_eval(image_paths)
                image_paths = parsed if isinstance(parsed, list) else [image_paths]
            except Exception:
                image_paths = [image_paths]
        elif not isinstance(image_paths, list):
            image_paths = []
        # Normalize list entries to strings
        image_paths = [str(p) for p in image_paths if p is not None]
        
        # Load first image as PIL (Gemma 3 processor handles PIL directly)
        image = None
        for img_path in image_paths[:1]:  # Use first image only for now
            full_path = self.image_dir / img_path
            if full_path.exists():
                try:
                    image = Image.open(full_path).convert('RGB')
                    break
                except Exception:
                    continue
        
        # Create placeholder if no valid image
        if image is None:
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Image name for base-model lookup (use basename for consistent keys)
        image_name = Path(image_paths[0]).name if image_paths else None
        
        # Get rating
        rating = float(row.get('review_rating', row.get('rating', 3)))
        
        # User prompt: match inference format (config.prompts)
        user_content = self.user_prompt

        # Build per-sense ratings and descriptions from human annotations
        sensory_ratings = {
            'taste': float(row.get('sensory_taste', rating)),
            'smell': float(row.get('sensory_smell', rating)),
            'texture': float(row.get('sensory_texture', rating)),
            'sound': float(row.get('sensory_sound', rating)),
        }
        sensory_descs = {
            'taste': str(row.get('taste_desc', '')),
            'smell': str(row.get('smell_desc', '')),
            'texture': str(row.get('texture_desc', '')),
            'sound': str(row.get('sound_desc', '')),
        }
        mammoth_prose = {}
        if image_name and self.mammoth_target_lookup:
            mammoth_prose = self.mammoth_target_lookup.get(image_name, {}) or {}
        target_response = self._create_target_response(
            rating, sensory_ratings=sensory_ratings, sensory_descs=sensory_descs,
            mammoth_prose=mammoth_prose
        )
        
        return {
            'image': image,
            'user_content': user_content,
            'target_response': target_response,
            'rating': rating,
            'sense_ratings': [
                float(sensory_ratings['taste']),
                float(sensory_ratings['smell']),
                float(sensory_ratings['texture']),
                float(sensory_ratings['sound']),
            ],
            'review_id': row.get('review_id', str(idx)),
        }
    
    def _create_target_response(
        self,
        rating: float,
        sensory_ratings: Dict[str, float],
        sensory_descs: Dict[str, str],
        mammoth_prose: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create target: human ratings + sensory descriptors; optionally + MAmmoTH expansion.
        human_only: use only human descriptors (no mammoth). Best for rating/desc grounding.
        prefer_human: use mammoth only when human descriptor is empty.
        full: human + mammoth (mammoth sanitized to remove embedded rating phrases).
        """
        mammoth_prose = mammoth_prose or {}
        ignore = {"", "nan", "not sure", "none", "n/a", "idk", "unsure"}
        sense_order = [("taste", "Taste"), ("smell", "Smell"), ("texture", "Texture"), ("sound", "Sound")]

        dimensions = []
        for sense, label in sense_order:
            r = max(1.0, min(5.0, float(sensory_ratings.get(sense, rating))))
            human = (sensory_descs.get(sense, "") or "").strip()
            if human.lower() in ignore:
                human = ""
            expansion_raw = (mammoth_prose.get(sense, "") or "").strip()
            expansion = _sanitize_mammoth_prose(expansion_raw) if expansion_raw else ""

            # Decide whether to use mammoth
            use_mammoth = (
                not self.human_only
                and expansion
                and (not self.prefer_human or not human)
            )

            # Body: human descriptor (primary) + optionally mammoth expansion
            parts = []
            if human:
                h = human if human.endswith((".", "!", "?")) else human + "."
                parts.append(h)
            if use_mammoth:
                parts.append(expansion)
            body = " ".join(parts).strip()
            if body and body[-1] not in ".!?":
                body += "."

            header = f"{label} ({r:.1f}/5.0):"
            dimensions.append(f"{header} {body}" if body else header)

        if not self.fixed_sense_order:
            random.shuffle(dimensions)
        return "Sensory Assessment:\n" + "\n".join(dimensions)



# ============================================================================
# Custom Trainer with Multi-Task Learning
# ============================================================================

class GemmaVLMTrainer(Trainer):
    """
    Custom Trainer that:
    1. Properly processes images through the Gemma 3 processor
    2. Generates labels for causal LM training
    3. Optionally adds rating prediction loss
    """
    
    def __init__(
        self,
        *args,
        processor=None,
        system_prompt: str = "",
        rating_head: nn.Module = None,
        rating_loss_weight: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.system_prompt = system_prompt
        self.rating_head = rating_head
        self.rating_loss_weight = rating_loss_weight
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation with multi-task learning.
        """
        # The inputs are already processed by our collator
        labels = inputs.pop("labels", None)
        ratings = inputs.pop("ratings", None)
        sense_ratings = inputs.pop("sense_ratings", None)
        prompt_lengths = inputs.pop("prompt_lengths", None)
        
        # Forward pass
        outputs = model(**inputs, output_hidden_states=True)
        
        # Language modeling loss
        if labels is not None:
            # Shift for causal LM
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            lm_loss = outputs.loss
        
        total_loss = lm_loss
        
        # Add rating prediction loss if rating head exists
        if self.rating_head is not None:
            hidden_states = outputs.hidden_states[-1]
            # Predict rating from the *prompt* representation (image+review),
            # not from the final token (which includes ground-truth assistant tokens).
            if prompt_lengths is None:
                # Fallback: use last non-padding token (best-effort)
                lengths = inputs.get("attention_mask", None)
                if lengths is not None:
                    lengths = lengths.sum(dim=1).to(hidden_states.device)
                else:
                    lengths = torch.full(
                        (hidden_states.size(0),),
                        hidden_states.size(1),
                        device=hidden_states.device,
                        dtype=torch.long,
                    )
            else:
                lengths = prompt_lengths.to(hidden_states.device)

            idx = (lengths - 1).clamp(min=0, max=hidden_states.size(1) - 1)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            prompt_hidden = hidden_states[batch_idx, idx, :].to(self.rating_head.weight.dtype)
            rating_pred = self.rating_head(prompt_hidden)
            if rating_pred.ndim == 1:
                rating_pred = rating_pred.unsqueeze(-1)

            rating_target = None
            if rating_pred.size(-1) == 1:
                if ratings is not None:
                    rating_target = ratings.to(rating_pred.dtype).unsqueeze(-1)
                elif sense_ratings is not None:
                    rating_target = sense_ratings.to(rating_pred.dtype).mean(dim=-1, keepdim=True)
            else:
                if sense_ratings is not None:
                    rating_target = sense_ratings.to(rating_pred.dtype)

            if rating_target is not None:
                rating_loss = nn.functional.mse_loss(rating_pred, rating_target)
                total_loss = lm_loss + self.rating_loss_weight * rating_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        During eval, if a rating head is enabled, return rating predictions so we can
        compute MAE/RMSE and early-stop on a more meaningful signal than LM loss.
        """
        if self.rating_head is None:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        inputs = self._prepare_inputs(inputs)
        ratings = inputs.get("ratings", None)
        sense_ratings = inputs.get("sense_ratings", None)
        # Don't pass auxiliary labels into the backbone model
        model_inputs = {k: v for k, v in inputs.items() if k not in {"ratings", "sense_ratings"}}

        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)
            loss = outputs.loss if "labels" in model_inputs else None

            hidden_states = outputs.hidden_states[-1]
            lengths = model_inputs.get("attention_mask", None)
            if lengths is not None:
                lengths = lengths.sum(dim=1).to(hidden_states.device)
            else:
                lengths = torch.full(
                    (hidden_states.size(0),),
                    hidden_states.size(1),
                    device=hidden_states.device,
                    dtype=torch.long,
                )
            idx = (lengths - 1).clamp(min=0, max=hidden_states.size(1) - 1)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            prompt_hidden = hidden_states[batch_idx, idx, :].to(self.rating_head.weight.dtype)
            rating_pred = self.rating_head(prompt_hidden)
            if rating_pred.ndim == 1:
                rating_pred = rating_pred.unsqueeze(-1)

        # Return preds/labels as tensors — Trainer's pad_across_processes requires
        # tensors, not numpy. compute_metrics will convert via np.asarray().
        preds = rating_pred.detach().float().cpu()
        if sense_ratings is not None:
            label_ids = sense_ratings.detach().float().cpu()
        elif ratings is not None:
            label_ids = ratings.detach().float().cpu().unsqueeze(-1)
        else:
            label_ids = None
        return (loss.detach() if loss is not None else None, preds, label_ids)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save model, plus processor and rating head for checkpoint compatibility."""
        super()._save(output_dir=output_dir, state_dict=state_dict)
        if output_dir is None:
            return
        out_path = Path(output_dir)
        if self.processor is not None:
            try:
                self.processor.save_pretrained(out_path)
            except Exception:
                pass
        if self.rating_head is not None:
            try:
                torch.save(self.rating_head.state_dict(), out_path / "rating_head.pt")
            except Exception:
                pass


# ============================================================================
# Data Collator for VLM Training
# ============================================================================

class GemmaVLMCollator:
    """
    Collator that processes batches for Gemma 3 VLM training.
    
    For Gemma 3 VLM, use processor.apply_chat_template with multimodal messages.
    The processor handles image token insertion automatically.
    """
    
    def __init__(self, processor, system_prompt: str, max_length: int = 2048):
        self.processor = processor
        self.system_prompt = system_prompt
        self.max_length = max_length
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Collect data
        images = [f['image'] for f in features]
        user_contents = [str(f.get('user_content', '')) for f in features]
        target_responses = [str(f.get('target_response', '')) for f in features]
        ratings = [f['rating'] for f in features]
        sense_ratings = [f.get('sense_ratings', [f['rating']] * 4) for f in features]
        
        # Build conversation format for each sample
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_pixel_values = []
        all_prompt_lengths = []
        
        for i, (image, user_content, target_response) in enumerate(
            zip(images, user_contents, target_responses)
        ):
            try:
                # Gemma 3 multimodal chat formatting is strict:
                # - Use processor.apply_chat_template
                # - Use `content` as a LIST of typed parts for *every* message (text/image),
                #   otherwise you'll hit errors like "string indices must be integers".
                prompt_messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": user_content},
                        ],
                    },
                ]

                full_messages = prompt_messages + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": target_response}],
                    }
                ]
                
                # Process full conversation (with response) for training
                full_inputs = self.processor.apply_chat_template(
                    full_messages,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                
                # Process prompt only (without response) to find split point
                prompt_inputs = self.processor.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                
                input_ids = full_inputs["input_ids"].squeeze(0)
                attention_mask = full_inputs["attention_mask"].squeeze(0)
                prompt_len = prompt_inputs["input_ids"].shape[1]
                
                # Create labels - mask prompt, only train on response
                labels = input_ids.clone()
                labels[:prompt_len] = -100
                
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                all_labels.append(labels)
                all_prompt_lengths.append(int(prompt_len))
                
                # Get pixel values
                if "pixel_values" in full_inputs:
                    all_pixel_values.append(full_inputs["pixel_values"].squeeze(0))
                    
            except Exception as e:
                feat = features[i] if i < len(features) else {}
                review_id = feat.get('review_id', i) if isinstance(feat, dict) else i
                print(f"Warning: Failed to process sample {i} (review_id={review_id}): {e}")
                # Create minimal dummy tensors - these will be masked anyway
                all_input_ids.append(torch.zeros(10, dtype=torch.long))
                all_attention_masks.append(torch.ones(10, dtype=torch.long))
                all_labels.append(torch.full((10,), -100, dtype=torch.long))
                all_prompt_lengths.append(10)
        
        # Pad sequences
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for input_ids, attention_mask, labels in zip(
            all_input_ids, all_attention_masks, all_labels
        ):
            pad_len = max_len - input_ids.shape[0]
            if pad_len > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), pad_token_id, dtype=torch.long)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_len, dtype=torch.long)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=torch.long)
                ])
            
            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)
            padded_labels.append(labels)
        
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels),
            "ratings": torch.tensor(ratings, dtype=torch.float32),
            "sense_ratings": torch.tensor(sense_ratings, dtype=torch.float32),
            "prompt_lengths": torch.tensor(all_prompt_lengths, dtype=torch.long),
        }
        
        # Add pixel values if available
        if all_pixel_values:
            try:
                batch["pixel_values"] = torch.stack(all_pixel_values)
            except Exception:
                # If pixel values have different shapes, skip stacking
                pass
        
        return batch


# ============================================================================
# Main QLoRA Trainer Class
# ============================================================================

class QLoRATrainer:
    """
    QLoRA Trainer for Gemma 3 27B IT.
    Uses 4-bit quantization with LoRA adapters.
    """
    
    # Use config prompts for train/inference alignment
    SYSTEM_PROMPT = SYSTEM_PROMPT
    
    def __init__(
        self,
        model_name: str = "google/gemma-3-27b-it",
        lora_config: Optional[Dict] = None,
        output_dir: str = "checkpoints/gemma3_qlora",
        device: str = "cuda",
        use_rating_head: bool = True,
        resume_from_checkpoint: Optional[str] = None,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Detect distributed training and map each process to its local GPU
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size > 1:
            self.device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
        else:
            self.device = device
        self.use_rating_head = use_rating_head
        
        # Default LoRA config
        if lora_config is None:
            lora_config = {
                # Reduced capacity to mitigate overfitting.
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                 'gate_proj', 'up_proj', 'down_proj'],
                # Slightly higher dropout helps generalization on small/templated data.
                'lora_dropout': 0.15,
                'bias': 'none',
                'task_type': TaskType.CAUSAL_LM
            }
        
        # Load processor
        print(f"Loading processor from {model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Ensure pad token
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
        # 4-bit Quantization config
        print("Configuring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        
        # Load model
        print(f"Loading model from {model_name} (4-bit)...")
        # Force model onto GPU 0 to avoid multi-GPU dispatch issues
        device_map = {"": local_rank} if world_size > 1 else {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=True
        )

        # Hard requirement: fail fast if FlashAttention2 isn't actually available/selected.
        from transformers.utils import is_flash_attn_2_available
        if not is_flash_attn_2_available():
            raise RuntimeError(
                "FlashAttention2 is not available but attn_implementation was set "
                "to 'flash_attention_2'. Fix your environment/container."
            )
        impl = getattr(self.model.config, "_attn_implementation", None) or getattr(
            self.model.config, "attn_implementation", None
        )
        if str(impl) != "flash_attention_2":
            raise RuntimeError(
                f"Expected model to use FlashAttention2, but config reports: {impl!r}"
            )
        
        # Prepare for k-bit training
        print("Preparing model for QLoRA training...")
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA or load existing adapter (for stage 2)
        if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
            base_path = Path(resume_from_checkpoint)
            # Resolve to checkpoint dir with adapter_model.safetensors
            if (base_path / "adapter_model.safetensors").exists():
                adapter_path = str(base_path)
            else:
                ckpt_dirs = sorted(
                    base_path.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[1]) if len(p.name.split("-")) > 1 and p.name.split("-")[1].isdigit() else 0
                )
                adapter_path = str(ckpt_dirs[-1]) if ckpt_dirs else str(base_path)
            print(f"Loading adapter from {adapter_path} (stage 2 / resume)...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=True)
        else:
            print("Applying LoRA adapters...")
            peft_config = LoraConfig(**lora_config)
            self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
        
        # Rating prediction head
        self.rating_head = None
        if use_rating_head:
            hidden_size = self._get_hidden_size()
            self.rating_head = nn.Linear(hidden_size, 4).to(self.device).to(torch.bfloat16)
            # Load existing rating head if resuming (check root and latest checkpoint-N)
            if resume_from_checkpoint:
                base_path = Path(resume_from_checkpoint)
                head_path = base_path / "rating_head.pt"
                if not head_path.exists():
                    ckpt_dirs = sorted(base_path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]) if len(p.name.split("-")) > 1 and p.name.split("-")[1].isdigit() else -1)
                    for sub in reversed(ckpt_dirs):
                        if (sub / "rating_head.pt").exists():
                            head_path = sub / "rating_head.pt"
                            break
                if head_path.exists():
                    state = torch.load(head_path, map_location=self.device, weights_only=True)
                    model_state = self.rating_head.state_dict()
                    compatible = {
                        k: v for k, v in state.items()
                        if k in model_state and model_state[k].shape == v.shape
                    }
                    self.rating_head.load_state_dict(compatible, strict=False)
                    if len(compatible) == len(model_state):
                        print(f"Rating head loaded from {head_path}")
                    else:
                        print(f"Rating head partially loaded from {head_path}; incompatible shape(s) reinitialized")
                else:
                    print(f"Rating head initialized (hidden_size={hidden_size})")
            else:
                print(f"Rating head initialized (hidden_size={hidden_size})")
    
    def _get_hidden_size(self) -> int:
        """Get hidden size from config."""
        config = self.model.config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and hasattr(config, "text_config"):
            hidden_size = getattr(config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not determine hidden size")
        return hidden_size
    
    def train(
        self,
        train_df,
        val_df=None,
        image_dir: str = "data/Apify_Yelp_photos",
        mammoth_target_lookup: Optional[Dict[str, Dict[str, str]]] = None,
        num_epochs: int = 1,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 64,
        learning_rate: float = 5e-6,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.05,
        max_grad_norm: float = 1.0,
        save_steps: int = 100,
        save_total_limit: Optional[int] = 3,
        eval_steps: Optional[int] = None,
        logging_steps: int = 10,
        max_steps: int = -1,
        max_length: int = 2048,
        optim: str = "paged_adamw_8bit",
        human_only: bool = False,
        prefer_human: bool = False,
        early_stop_patience: int = 3,
        early_stop_threshold: float = 0.0,
        user_prompt: str = USER_PROMPT,
        fixed_sense_order: bool = False,
    ):
        """Train the model. Use optim='adamw_torch' for multi-GPU (bitsandbytes+DDP is flaky)."""
        
        # Create datasets
        mammoth_lookup = mammoth_target_lookup or {}
        if human_only:
            print("Training mode: human_only (ratings + descriptors only, no mammoth)")
        elif prefer_human:
            print("Training mode: prefer_human (mammoth only when human descriptor empty)")
        elif mammoth_lookup:
            print(f"Using MAmmoTH-style targets for {len(mammoth_lookup):,} images (mammoth prose sanitized)")
        print("Creating VLM datasets...")
        train_dataset = GemmaVLMDataset(
            train_df,
            image_dir=image_dir,
            processor=self.processor,
            system_prompt=self.SYSTEM_PROMPT,
            mammoth_target_lookup=mammoth_lookup,
            max_length=max_length,
            human_only=human_only,
            prefer_human=prefer_human,
            user_prompt=user_prompt,
        )
        train_dataset.fixed_sense_order = fixed_sense_order
        
        val_dataset = None
        if val_df is not None:
            val_dataset = GemmaVLMDataset(
                val_df,
                image_dir=image_dir,
                processor=self.processor,
                system_prompt=self.SYSTEM_PROMPT,
                mammoth_target_lookup=mammoth_lookup,
                max_length=max_length,
                human_only=human_only,
                prefer_human=prefer_human,
                user_prompt=user_prompt,
            )
            val_dataset.fixed_sense_order = fixed_sense_order
        
        # Create collator
        collator = GemmaVLMCollator(
            processor=self.processor,
            system_prompt=self.SYSTEM_PROMPT,
            max_length=max_length
        )
        
        # Training arguments
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        do_eval = val_dataset is not None
        eval_every = eval_steps if eval_steps is not None else save_steps

        def compute_metrics(eval_pred):
            # When rating_head is enabled, prediction_step returns (rating_pred, label_ids)
            # where label_ids is sense_ratings when available.
            try:
                preds, labels = eval_pred
                preds = np.asarray(preds)
                labels = np.asarray(labels)

                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                if labels.ndim == 1:
                    labels = labels.reshape(-1, 1)

                dim = min(preds.shape[1], labels.shape[1])
                preds = preds[:, :dim]
                labels = labels[:, :dim]
                metrics = {}

                if dim == 0:
                    return {"sense_mae": float("nan"), "sense_rmse": float("nan")}

                if dim == 1:
                    mask = np.isfinite(preds[:, 0]) & np.isfinite(labels[:, 0])
                    p = preds[mask, 0]
                    t = labels[mask, 0]
                    if p.size == 0:
                        return {"sense_mae": float("nan"), "sense_rmse": float("nan")}
                    mae = float(np.mean(np.abs(p - t)))
                    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
                    return {
                        "sense_mae": mae,
                        "sense_rmse": rmse,
                        "rating_mae": mae,
                        "rating_rmse": rmse,
                    }

                sense_names = ["taste", "smell", "texture", "sound"]
                maes, rmses = [], []
                for i in range(min(4, dim)):
                    mask = np.isfinite(preds[:, i]) & np.isfinite(labels[:, i])
                    p = preds[mask, i]
                    t = labels[mask, i]
                    if p.size == 0:
                        mae_i = float("nan")
                        rmse_i = float("nan")
                    else:
                        mae_i = float(np.mean(np.abs(p - t)))
                        rmse_i = float(np.sqrt(np.mean((p - t) ** 2)))
                    metrics[f"{sense_names[i]}_mae"] = mae_i
                    metrics[f"{sense_names[i]}_rmse"] = rmse_i
                    maes.append(mae_i)
                    rmses.append(rmse_i)

                sense_mae = float(np.nanmean(maes)) if maes else float("nan")
                sense_rmse = float(np.nanmean(rmses)) if rmses else float("nan")
                metrics.update({
                    "sense_mae": sense_mae,
                    "sense_rmse": sense_rmse,
                    "rating_mae": sense_mae,
                    "rating_rmse": sense_rmse,
                })
                return metrics
            except Exception:
                return {}

        ta_kwargs = dict(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            bf16=True,
            gradient_checkpointing=True,
            optim=optim,
            lr_scheduler_type="cosine",
            report_to="none",
            logging_dir=str(self.output_dir / "logs"),
            eval_strategy="steps" if do_eval else "no",
            eval_steps=eval_every if do_eval else None,
            load_best_model_at_end=True if do_eval else False,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
        )
        if do_eval:
            ta_kwargs["metric_for_best_model"] = "sense_mae" if self.rating_head is not None else "loss"
            ta_kwargs["greater_is_better"] = False
        # DDP + gradient checkpointing can trigger "Expected to mark a variable ready only once"
        # with reentrant checkpointing. Prefer non-reentrant when supported.
        sig = inspect.signature(TrainingArguments.__init__)
        if ta_kwargs.get("gradient_checkpointing") and "gradient_checkpointing_kwargs" in sig.parameters:
            ta_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        if world_size > 1 and "ddp_find_unused_parameters" in sig.parameters:
            ta_kwargs["ddp_find_unused_parameters"] = False
        if world_size > 1 and "ddp_broadcast_buffers" in sig.parameters:
            ta_kwargs["ddp_broadcast_buffers"] = False
        training_args = TrainingArguments(**ta_kwargs)
        
        # Create custom trainer
        trainer = GemmaVLMTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics if (do_eval and self.rating_head is not None) else None,
            processor=self.processor,
            system_prompt=self.SYSTEM_PROMPT,
            rating_head=self.rating_head,
            rating_loss_weight=0.5
        )

        if do_eval and early_stop_patience and early_stop_patience > 0:
            from transformers import EarlyStoppingCallback
            trainer.add_callback(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stop_patience,
                    early_stopping_threshold=early_stop_threshold,
                )
            )
        
        # Train
        print("=" * 50)
        print("Starting QLoRA Training")
        print(f"  Model: {self.model_name}")
        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Val samples: {len(val_dataset) if val_dataset else 0:,}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Rating head: {'Enabled' if self.rating_head else 'Disabled'}")
        print("=" * 50)
        
        trainer.train()
        
        # Save
        print(f"Saving model to {self.output_dir}...")
        trainer.save_model()
        self.processor.save_pretrained(self.output_dir)
        
        if self.rating_head:
            torch.save(self.rating_head.state_dict(), self.output_dir / "rating_head.pt")
        
        print("Training complete!")


def main():
    """Main training function — human-annotated data only."""
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Gemma 3 27B VLM")
    parser.add_argument("--human_csv", type=str,
                        default="data/FINAL_DATASET_COMPLETE_with_rescaling.csv",
                        help="Path to FINAL_DATASET_COMPLETE_with_rescaling.csv (human-annotated sensory data)")
    parser.add_argument("--image_dir", type=str,
                        default="data/Images",
                        help="Directory containing images for the human-annotated data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/gemma3_qlora_human_sensory")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=64)
    # Overfitting-mitigating defaults (can override via CLI)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_steps", type=int, default=None, help="Run eval every N steps (defaults to --save_steps).")
    parser.add_argument("--val_max_samples", type=int, default=2000, help="Limit validation rows used for in-training eval/early stop.")
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--early_stop_threshold", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--no_rating_head", action="store_true")
    parser.add_argument("--mammoth_targets", type=str, default=None,
                        help="Path to mammoth_style_target_lookup.json (from precompute_mammoth_style_targets.py)")
    parser.add_argument("--require_prose_senses", type=int, default=1,
                        help="Require at least N senses to have prose (human or mammoth). Rows with header-only targets are excluded. Default 1. Set 0 to disable (may cause template/collapse issues).")
    parser.add_argument("--human_only", action="store_true",
                        help="Use only human ratings + descriptors (no mammoth). Best for grounding; avoids template overfit and rating collapse.")
    parser.add_argument("--prefer_human", action="store_true",
                        help="Use mammoth only when human descriptor is empty. Otherwise use human descriptor only.")
    parser.add_argument("--stage2_from", type=str, default=None,
                        help="Stage 2: load adapter from this checkpoint (from human_only stage 1), then train with mammoth for long reasoning. Requires --mammoth_targets. Uses lower LR and fewer steps.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Max number of checkpoints to keep. Set -1 to keep all.")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit",
                        help="Optimizer: paged_adamw_8bit (1-GPU) or adamw_torch (multi-GPU, more stable with DDP)")
    parser.add_argument("--retrain_mammoth_only", action="store_true",
                        help="Retraining Option: Filter dataset to ONLY include rows with MAmmoTH prose.")
    parser.add_argument("--fixed_sense_order", action="store_true",
                        help="Use fixed sense order (Taste,Smell,Texture,Sound) instead of random shuffle. Recommended for Stage-2 to match inference format.")
    
    args = parser.parse_args()
    
    # ---- Load human-annotated sensory data (ONLY training source) ----
    print(f"Loading human-annotated sensory data from {args.human_csv}...")
    df_reviews = load_human_sensory_data(
        csv_path=args.human_csv,
        image_dir=args.image_dir,
        require_all_caninfer=True,
    )
    print(f"Training on {len(df_reviews):,} human-annotated rows")
    
    # ---- Stage 2: load from human-only checkpoint, add mammoth for long reasoning ----
    stage2_from = getattr(args, "stage2_from", None)
    if stage2_from:
        if not args.mammoth_targets:
            raise SystemExit("--stage2_from requires --mammoth_targets. Stage 2 adds mammoth prose for long reasoning.")
        with open(args.mammoth_targets) as f:
            mammoth_target_lookup = json.load(f)
        print(f"Stage 2: Loading from {stage2_from}, adding mammoth for {len(mammoth_target_lookup):,} images")
        args.human_only = False
        args.prefer_human = False
        if args.lr >= 1e-5:
            args.lr = 5e-6
            print(f"Stage 2: Using lower LR {args.lr} to preserve grounding")
        # v2: removed auto-cap on max_steps to allow longer training (600-800+)
        # If you want to limit steps, pass --max_steps explicitly via CLI.
    elif args.mammoth_targets:
        with open(args.mammoth_targets) as f:
            mammoth_target_lookup = json.load(f)
        print(f"Loaded MAmmoTH-style targets for {len(mammoth_target_lookup):,} images")
    else:
        mammoth_target_lookup = None
    
    # ---- Filter to rows with rich targets (per PROGRESS_LOG: avoid header-only) ----
    if args.require_prose_senses > 0:
        ignore = {"", "nan", "not sure", "none", "n/a", "idk", "unsure"}
        def _has_prose(row, human_only_filter=False):
            img = (row.get("saved_path") or row.get("filename") or [""])
            img_name = img[0] if isinstance(img, list) else str(img)
            mammoth = (mammoth_target_lookup or {}).get(img_name, {}) or {}
            count = 0
            for s in ["taste", "smell", "texture", "sound"]:
                human = (str(row.get(f"{s}_desc", "") or "")).strip()
                if human.lower() in ignore:
                    human = ""
                expansion = (mammoth.get(s, "") or "").strip()
                if human_only_filter:
                    if human:
                        count += 1
                else:
                    if human or expansion:
                        count += 1
            return count
        df_reviews["_prose_count"] = df_reviews.apply(
            lambda r: _has_prose(r, human_only_filter=args.human_only), axis=1
        )
        before = len(df_reviews)
        df_reviews = df_reviews[df_reviews["_prose_count"] >= args.require_prose_senses].drop(columns=["_prose_count"]).reset_index(drop=True)
        mode_str = "human desc" if args.human_only else "prose (human or mammoth)"
        print(f"Filtered to {len(df_reviews):,} rows with ≥{args.require_prose_senses} sense(s) of {mode_str} (excluded {before - len(df_reviews):,})")

    if getattr(args, "retrain_mammoth_only", False):
        if mammoth_target_lookup is None:
            raise ValueError("--retrain_mammoth_only requires --mammoth_targets")
        print("RETRAINING OPTION: Filtering dataset to ONLY include rows explicitly matching MAmmoTH images.")
        
        def _has_mammoth_prose(row):
            img = (row.get("saved_path") or row.get("filename") or [""])
            img_name = img[0] if isinstance(img, list) else str(img)
            mammoth = mammoth_target_lookup.get(img_name, {}) or {}
            for s in ["taste", "smell", "texture", "sound"]:
                if (mammoth.get(s, "") or "").strip():
                    return True
            return False
            
        before_retrain = len(df_reviews)
        df_reviews = df_reviews[df_reviews.apply(_has_mammoth_prose, axis=1)].reset_index(drop=True)
        print(f"Filtered down to {len(df_reviews):,} rows (from {before_retrain:,}) that possess MAmmoTH prose.")

    if args.max_samples and args.max_samples < len(df_reviews):
        df_reviews = df_reviews.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print(f"  (sub-sampled to {len(df_reviews):,} rows)")
    
    # ---- Create image-level splits ----
    # All annotations for the same image go into the same split to
    # prevent data leakage (same image in train and test).
    print("Creating image-level splits...")
    train_df, val_df, test_df = create_image_level_splits(df_reviews)
    
    # ---- Initialize trainer ----
    print(f"\nInitializing QLoRA trainer...")
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }
    trainer = QLoRATrainer(
        model_name=args.model_name,
        lora_config=lora_config,
        output_dir=args.output_dir,
        use_rating_head=not args.no_rating_head,
        resume_from_checkpoint=stage2_from if stage2_from else None,
    )
    
    # Use a small validation subset for in-training eval + early stopping to reduce overfitting.
    val_for_train = val_df
    if args.val_max_samples is not None and args.val_max_samples > 0 and len(val_for_train) > args.val_max_samples:
        val_for_train = val_for_train.sample(n=args.val_max_samples, random_state=42).reset_index(drop=True)
        print(f"Using val subset for early stop: {len(val_for_train):,} rows (from {len(val_df):,})")

    selected_prompt = USER_PROMPT
    print(f"User prompt: USER_PROMPT (fixed_sense_order={args.fixed_sense_order})")

    trainer.train(
        train_df=train_df,
        val_df=val_for_train,
        image_dir=args.image_dir,
        mammoth_target_lookup=mammoth_target_lookup,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        max_length=args.max_length,
        save_steps=args.save_steps,
        save_total_limit=(None if args.save_total_limit is None or args.save_total_limit < 0 else args.save_total_limit),
        eval_steps=args.eval_steps,
        optim=args.optim,
        human_only=args.human_only,
        prefer_human=args.prefer_human,
        early_stop_patience=args.early_stop_patience,
        early_stop_threshold=args.early_stop_threshold,
        user_prompt=selected_prompt,
        fixed_sense_order=args.fixed_sense_order,
    )


if __name__ == "__main__":
    main()
