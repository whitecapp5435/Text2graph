"""
  /home/jpark284/.conda/envs/pj2-qwen/bin/python src/infer_hf.py \
    --repo_id jpark284/Qwen3-0.6b \
    --text "Junha is student, he is in Arizona State University."
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def apply_chat_template_compat(tokenizer, messages, **kwargs):
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_prompt_template_file(path: str) -> Tuple[str, str]:
    """
    text2graph_prompt_format.txt 형식:
      SYSTEM:
      ...

      USER:
      ... <INPUT_TEXT> ...
    """
    txt = open(path, "r", encoding="utf-8").read()
    if "SYSTEM:" not in txt or "USER:" not in txt:
        raise ValueError("prompt template must contain 'SYSTEM:' and 'USER:' blocks")
    system = txt.split("SYSTEM:", 1)[1].split("USER:", 1)[0].strip()
    user_template = txt.split("USER:", 1)[1].strip()
    if "<INPUT_TEXT>" not in user_template:
        raise ValueError("prompt template USER block must contain '<INPUT_TEXT>' placeholder")
    return system, user_template


def load_prompt_template_from_text2graph(data_path: str) -> List[Dict[str, str]]:
    data = json.load(open(data_path, "r", encoding="utf-8"))
    for item in data:
        if not (isinstance(item, dict) and isinstance(item.get("prompt"), list)):
            continue
        msgs = [m for m in item["prompt"] if isinstance(m, dict) and "role" in m and "content" in m]
        if any(m["role"] == "system" for m in msgs) and any(m["role"] == "user" for m in msgs):
            return msgs
    raise RuntimeError(f"Could not find prompt template in {data_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, default="jpark284/Qwen3-0.6b", help="HF repo id, e.g. jpark284/Qwen3-0.6b")
    parser.add_argument("--text", type=str, default="Junha is student, he is in Arizona State University.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--enable_thinking", action="store_true", help="Qwen3 thinking mode (default: off)")
    parser.add_argument(
        "--prompt_template_file",
        type=str,
        default="text2graph_prompt_format.txt",
        help="SYSTEM/USER template file path",
    )
    parser.add_argument(
        "--fallback_data_path",
        type=str,
        default="text2graph.json",
        help="alternative system prompt dataset path if prompt_template_file is not found",
    )
    parser.add_argument("--json_only", action="store_true")
    args = parser.parse_args()

    tokenizer_kwargs = {"trust_remote_code": True}
    try:
        tok = AutoTokenizer.from_pretrained(args.repo_id, fix_mistral_regex=True, **tokenizer_kwargs)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(args.repo_id, **tokenizer_kwargs)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.repo_id,
        device_map=args.device_map,
        torch_dtype=_dtype(args.dtype),
        trust_remote_code=True,
    )
    model.eval()

    if args.prompt_template_file and os.path.exists(args.prompt_template_file):
        system, user_template = parse_prompt_template_file(args.prompt_template_file)
        user = user_template.replace("<INPUT_TEXT>", args.text)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    else:
        tmpl = load_prompt_template_from_text2graph(args.fallback_data_path)
        messages = []
        for m in tmpl:
            if m["role"] == "user":
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Analyze this text, identify the entities, and extract meaningful relationships as per given instructions: "
                            + args.text
                        ),
                    }
                )
            else:
                messages.append({"role": m["role"], "content": m["content"]})

    prompt = apply_chat_template_compat(
        tok,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    inputs = tok(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    completion = tok.decode(out_ids[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()

    if not args.json_only:
        print("=== Prompt (system+user) ===")
        for m in messages:
            print(f"[{m['role']}]\n{m['content']}\n")
        print("=== Model Output (raw) ===")
    print(completion)

    # best-effort parse + pretty print
    try:
        obj = json.loads(completion)
        if not args.json_only:
            print("\n=== Parsed JSON ===")
            print(json.dumps(obj, ensure_ascii=False, indent=2))
    except Exception:
        pass


if __name__ == "__main__":
    main()

