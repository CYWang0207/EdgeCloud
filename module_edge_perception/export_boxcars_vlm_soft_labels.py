"""Export BoxCars soft labels from a generative VLM.

The default ``choice_token`` method turns the 16-way classification problem
into a multiple-choice question.  The prompt contains the complete mapping
(``A. ford`` ... ``P. kia``), while the model is scored only on the next-token
logits of A--P.  This is a prompt-time verbalizer: it does not train or modify
the VLM.

``brand_sequence`` is kept as a diagnostic implementation of conventional
forced decoding.  It scores every brand continuation and length-normalizes
multi-token answers, so its output can be compared with the default method.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from boxcars_dataset import BoxCarsMultiView, VALID_SPLITS
from boxcars_drift_dataset import BoxCarsDriftWrapper
from train_adapter import dataset_fingerprint


LETTERS = tuple(chr(ord("A") + index) for index in range(16))


def parse_args():
    parser = argparse.ArgumentParser(
        description="用生成式 VLM 导出 BoxCars 16 类软标签"
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="vlm_soft_labels_boxcars.npz")
    parser.add_argument("--split", choices=VALID_SPLITS, default="validation")
    parser.add_argument(
        "--method", choices=("choice_token", "brand_sequence"),
        default="choice_token",
    )
    parser.add_argument(
        "--drift-schedule", default="none",
        help="none 表示原图；其他值传给 BoxCarsDriftWrapper（如 mixed）",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def build_prompt(classes):
    options = "\n".join(
        f"{letter}. {brand}" for letter, brand in zip(LETTERS, classes)
    )
    return (
        "The four images are observations of the same vehicle from one traffic "
        "camera. Identify its manufacturer. Choose exactly one option from the "
        "list below.\n"
        f"{options}\n"
        "Reply with only the option letter."
    )


def make_messages(images, prompt):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def to_pil_images(tensor, view_mask):
    count = int(view_mask.sum().item())
    return [TF.to_pil_image(tensor[index].clamp(0, 1)) for index in range(count)]


def model_device(model):
    return next(model.parameters()).device


def move_inputs(inputs, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def single_token_ids(tokenizer):
    """Find a shared prefix for which A--P are each exactly one token."""
    for prefix in (" ", "", "\n"):
        encoded = [tokenizer.encode(prefix + letter, add_special_tokens=False)
                   for letter in LETTERS]
        if all(len(ids) == 1 for ids in encoded):
            return prefix, [ids[0] for ids in encoded]
    details = {
        letter: tokenizer.encode(" " + letter, add_special_tokens=False)
        for letter in LETTERS
    }
    raise RuntimeError(
        "A-P 无法构造为统一的单 token 选项，不能使用 choice_token: "
        + json.dumps(details)
    )


def prepare_prompt(processor, images, prompt):
    messages = make_messages(images, prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text, processor(
        text=[text], images=images, padding=True, return_tensors="pt"
    )


def choice_token_scores(model, processor, images, prompt, token_ids):
    _, inputs = prepare_prompt(processor, images, prompt)
    inputs = move_inputs(inputs, model_device(model))
    with torch.inference_mode():
        output = model(**inputs, use_cache=False)
    return output.logits[0, -1, token_ids].float().cpu()


def continuation_ids(tokenizer, prompt_text, continuation):
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(
        prompt_text + continuation, add_special_tokens=False
    )
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("候选答案改变了提示词末尾分词边界")
    answer_ids = full_ids[len(prompt_ids):]
    if not answer_ids:
        raise RuntimeError(f"空候选答案: {continuation!r}")
    return prompt_ids, answer_ids


def brand_sequence_scores(model, processor, images, prompt, classes):
    prompt_text, visual_inputs = prepare_prompt(processor, images, prompt)
    tokenizer = processor.tokenizer
    scores = []
    for brand in classes:
        prompt_ids, answer_ids = continuation_ids(
            tokenizer, prompt_text, " " + brand
        )
        full_ids = torch.tensor(
            [prompt_ids + answer_ids], dtype=torch.long,
            device=model_device(model),
        )
        attention_mask = torch.ones_like(full_ids)
        inputs = move_inputs(visual_inputs, model_device(model))
        inputs["input_ids"] = full_ids
        inputs["attention_mask"] = attention_mask
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits
        start = len(prompt_ids) - 1
        answer_logits = logits[0, start:start + len(answer_ids)].float()
        log_probs = answer_logits.log_softmax(dim=-1)
        target = torch.tensor(answer_ids, device=log_probs.device)
        token_log_probs = log_probs.gather(1, target[:, None]).squeeze(1)
        scores.append(token_log_probs.mean().cpu())
    return torch.stack(scores)


def summarize(scores, labels, temperature):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = scores / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    predictions = probs.argmax(axis=1)
    entropy = -(probs * np.log(probs.clip(1e-30))).sum(axis=1)
    return probs.astype(np.float32), {
        "samples": int(len(labels)),
        "top1": float((predictions == labels).mean()),
        "mean_max_probability": float(probs.max(axis=1).mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_normalized_entropy": float(entropy.mean() / math.log(probs.shape[1])),
        "min_row_sum": float(probs.sum(axis=1).min()),
        "max_row_sum": float(probs.sum(axis=1).max()),
    }


def main():
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Qwen3-VL 依赖，请安装 transformers>=4.57 和 accelerate"
        ) from exc

    base_dataset = BoxCarsMultiView(
        args.dataset_path, split=args.split, task="make", num_views=4,
        # BoxCars crops have different spatial sizes within one track, while
        # the dataset returns a stacked tensor.  Match the existing MV-ViT
        # evaluation input instead of silently stretching at a later stage.
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    )
    dataset = base_dataset
    if args.drift_schedule != "none":
        dataset = BoxCarsDriftWrapper(
            base_dataset, schedule=args.drift_schedule,
            seed=args.seed, normalize=False,
        )

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        dtype="auto",
        trust_remote_code=args.trust_remote_code,
    ).eval()
    prompt = build_prompt(base_dataset.classes)
    token_prefix, token_ids = single_token_ids(processor.tokenizer)
    print(f"method={args.method} choice_token_prefix={token_prefix!r}")
    print(prompt, flush=True)

    limit = min(len(dataset), args.max_samples or len(dataset))
    all_scores, labels = [], []
    for index in range(limit):
        item = dataset[index]
        images, view_mask, label = item[:3]
        pil_images = to_pil_images(images, view_mask)
        if args.method == "choice_token":
            scores = choice_token_scores(
                model, processor, pil_images, prompt, token_ids
            )
        else:
            scores = brand_sequence_scores(
                model, processor, pil_images, prompt, base_dataset.classes
            )
        all_scores.append(scores.numpy())
        labels.append(int(label))
        if (index + 1) % args.log_every == 0 or index + 1 == limit:
            _, partial = summarize(all_scores, labels, args.temperature)
            print(
                f"{index + 1}/{limit} top1={partial['top1']:.4f} "
                f"max_p={partial['mean_max_probability']:.4f} "
                f"H_norm={partial['mean_normalized_entropy']:.4f}",
                flush=True,
            )

    score_array = np.asarray(all_scores, dtype=np.float32)
    probs, metrics = summarize(score_array, labels, args.temperature)
    split_key = "val" if args.split == "validation" else args.split
    payload = {
        f"{split_key}_logits": score_array,
        f"{split_key}_probs": probs,
        f"{split_key}_labels": np.asarray(labels, dtype=np.int64),
        "classes": np.asarray(base_dataset.classes),
        "method": np.asarray(args.method),
        "temperature": np.asarray(args.temperature),
        "drift_schedule": np.asarray(args.drift_schedule),
        "metrics_json": np.asarray(json.dumps(metrics, sort_keys=True)),
    }
    if limit == len(base_dataset) and args.drift_schedule == "none":
        fingerprint_key = "val_fingerprint" if args.split == "validation" else f"{args.split}_fingerprint"
        payload[fingerprint_key] = np.asarray(dataset_fingerprint(base_dataset))
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    np.savez(args.output, **payload)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
