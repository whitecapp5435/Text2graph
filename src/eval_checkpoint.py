"""
SFT 체크포인트 간단 테스트(추론/정확도) 스크립트
--------------------------------------------

목적
- `out_sft/checkpoint-XXXX` 같은 체크포인트가 "진짜로 JSON text2graph를 잘 뱉는지" 빠르게 확인합니다.
- LoRA 어댑터 체크포인트(PEFT)인지, full model 체크포인트인지 자동으로 감지해 로드합니다.

사용 예시

1) LoRA 체크포인트(대부분 SFTTrainer+LoRA는 adapter 형태로 저장됨)
  /home/jpark284/.conda/envs/pj2-qwen/bin/python src/eval_checkpoint.py \
    --checkpoint_dir out_sft/checkpoint-1000 \
    --base_model unsloth/Qwen3-0.6B \
    --data_path text2graph.json \
    --num_samples 8 \
    --max_new_tokens 512

2) full model로 저장된 경우(merged 등)
  /home/jpark284/.conda/envs/pj2-qwen/bin/python src/eval_checkpoint.py \
    --checkpoint_dir out_sft/merged \
    --data_path text2graph.json \
    --num_samples 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from typing import Any, Dict, List, Optional, Tuple

import torch

from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


def apply_chat_template_compat(tokenizer, messages, **kwargs):
    """
    tokenizer.apply_chat_template 호출 호환 래퍼.
    - Qwen3 계열은 enable_thinking 인자를 지원
    - 다른 모델/버전은 지원하지 않을 수 있어 TypeError가 날 수 있음
    """
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


class StopOnJsonCriteria(StoppingCriteria):
    """
    생성 중 "완전한 1개 JSON object"가 만들어지면 즉시 중단하기 위한 stopping criteria.

    - 모델이 JSON을 다 만들고도 EOS를 안 내거나, 계속 반복 생성하는 경우를 방지합니다.
    - 동작 방식:
      1) prompt 길이 이후로 생성된 토큰만 decode
      2) braces 균형이 맞는 JSON candidate를 찾고 json.loads가 되면 stop
    """

    def __init__(self, tokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
        # input_ids: (batch=1, seq_len)
        seq = input_ids[0]
        if seq.shape[0] <= self.prompt_len:
            return False
        gen_ids = seq[self.prompt_len :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # 빠른 필터: 닫는 brace가 없으면 불가능
        if "}" not in text:
            return False
        # 실제 JSON 추출/파싱 시도
        extracted = extract_json_from_text(text)
        # "제대로 된 스키마"가 나왔을 때만 멈추게 해두면 evaluation이 안정적입니다.
        return any(validate_json_structure(x) for x in extracted)


def extract_json_from_text(text: str) -> List[Dict[str, Any]]:
    """
    문자열 안에 있는 JSON object 후보들을 { ... } 균형괄호 기준으로 찾아서 파싱합니다.
    - 모델이 JSON 앞/뒤로 군더더기 텍스트를 붙여도 JSON 부분만 추출하기 위한 함수
    """
    json_start = 0
    close_brace_count = 0
    extracted_jsons: List[Dict[str, Any]] = []

    for idx, char in enumerate(text):
        if char == "{":
            if close_brace_count == 0:
                json_start = idx
            close_brace_count += 1
        elif char == "}":
            close_brace_count -= 1
            if close_brace_count == 0:
                candidate = text[json_start : idx + 1]
                try:
                    extracted_jsons.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
    return extracted_jsons


def validate_json_structure(data: Any) -> bool:
    required_keys = {"entities", "relations"}
    entity_required_keys = {"id", "text", "type"}
    relation_required_keys = {"head", "tail", "type"}

    if not isinstance(data, dict) or not required_keys.issubset(data.keys()):
        return False

    if not isinstance(data["entities"], list):
        return False
    for entity in data["entities"]:
        if not isinstance(entity, dict) or not entity_required_keys.issubset(entity.keys()):
            return False
        if not isinstance(entity["id"], int):
            return False
        if not isinstance(entity["text"], str) or not isinstance(entity["type"], str):
            return False

    if not isinstance(data["relations"], list):
        return False
    for rel in data["relations"]:
        if not isinstance(rel, dict) or not relation_required_keys.issubset(rel.keys()):
            return False
        if not isinstance(rel["head"], str) or not isinstance(rel["tail"], str) or not isinstance(rel["type"], str):
            return False

    return True


def compute_f1(pred_set: set, true_set: set) -> float:
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def entities_f1(pred: Dict[str, Any], gold: Dict[str, Any]) -> float:
    pred_set = {f"{e.get('text')}_{e.get('type')}" for e in pred.get("entities", []) if isinstance(e, dict)}
    gold_set = {f"{e.get('text')}_{e.get('type')}" for e in gold.get("entities", []) if isinstance(e, dict)}
    return compute_f1(pred_set, gold_set)


def relations_f1(pred: Dict[str, Any], gold: Dict[str, Any]) -> float:
    pred_set = {f"{r.get('head')}_{r.get('tail')}_{r.get('type')}" for r in pred.get("relations", []) if isinstance(r, dict)}
    gold_set = {f"{r.get('head')}_{r.get('tail')}_{r.get('type')}" for r in gold.get("relations", []) if isinstance(r, dict)}
    return compute_f1(pred_set, gold_set)


def _safe_list_len(x: Any) -> int:
    return int(len(x)) if isinstance(x, list) else 0


def _fmt_mean_median_max(values: List[int]) -> str:
    if not values:
        return "n/a"
    mean_v = float(sum(values) / len(values))
    med_v = float(statistics.median(values))
    max_v = int(max(values))
    return f"{mean_v:.2f}/{med_v:.1f}/{max_v}"


def _dtype_from_arg(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown --dtype {dtype}. Use one of: auto|bf16|fp16|fp32")


def load_model_and_tokenizer(
    checkpoint_dir: str,
    base_model: Optional[str],
    device_map: str,
    dtype: str,
    load_in_4bit: bool,
) -> Tuple[torch.nn.Module, Any]:
    """
    - adapter_config.json이 있으면 PEFT LoRA 어댑터 체크포인트로 보고 base+adapter로 로드
    - 없으면 checkpoint_dir 자체를 full model로 로드
    """
    is_adapter = os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))

    torch_dtype = _dtype_from_arg(dtype)
    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if is_adapter:
        peft_cfg = PeftConfig.from_pretrained(checkpoint_dir)
        base = base_model or getattr(peft_cfg, "base_model_name_or_path", None)
        if not base:
            raise ValueError(
                "LoRA(adapter) 체크포인트로 보이는데 base model을 찾을 수 없습니다. "
                "--base_model을 명시해주세요. 예: --base_model unsloth/Qwen3-0.6B"
            )
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            base,
            device_map=device_map,
            torch_dtype=None if load_in_4bit else torch_dtype,
            quantization_config=quant_cfg,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, checkpoint_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            device_map=device_map,
            torch_dtype=None if load_in_4bit else torch_dtype,
            quantization_config=quant_cfg,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    enable_thinking: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> Tuple[str, bool, int]:
    prompt_text = apply_chat_template_compat(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    stopping_criteria = StoppingCriteriaList([StopOnJsonCriteria(tokenizer, prompt_len=inputs["input_ids"].shape[-1])])

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
    )
    if hasattr(model.generation_config, "min_p"):
        gen_kwargs["min_p"] = min_p

    output_ids = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    completion_ids = output_ids[0, prompt_len:]
    gen_tokens = int(completion_ids.shape[0])
    hit_max_new_tokens = gen_tokens >= int(max_new_tokens)
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip(), hit_max_new_tokens, gen_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None, help="LoRA adapter인 경우 베이스 모델 경로/이름 (예: unsloth/Qwen3-0.6B)")
    parser.add_argument("--data_path", type=str, default="text2graph.json")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bf16", help="auto|bf16|fp16|fp32")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true", help="Qwen3 thinking 모드(기본: off)")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="예측 결과를 JSONL로 저장할 경로. 예: out_eval/preds.jsonl",
    )

    # 생성 샘플링 옵션 (JSON 추출은 보통 greedy가 안정적이라 do_sample=False 기본)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.data_path, encoding="utf-8") as f:
        data = json.load(f)

    indices = list(range(len(data)))
    random.shuffle(indices)
    indices = indices[: min(args.num_samples, len(indices))]

    model, tokenizer = load_model_and_tokenizer(
        checkpoint_dir=args.checkpoint_dir,
        base_model=args.base_model,
        device_map=args.device_map,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    json_ok = 0
    structure_ok = 0
    ent_f1_sum = 0.0
    rel_f1_sum = 0.0
    hit_max_count = 0
    gen_token_sum = 0
    gen_token_max = 0
    pred_ent_counts: List[int] = []
    pred_rel_counts: List[int] = []
    gold_ent_counts: List[int] = []
    gold_rel_counts: List[int] = []
    pred_json_invalid = 0
    gold_json_invalid = 0

    save_f = None
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        save_f = open(args.save_path, "w", encoding="utf-8")

    for i, idx in enumerate(indices, start=1):
        item = data[idx]
        messages = item["prompt"]
        gold_text = item["solution"]

        pred_text, hit_max_new_tokens, gen_tokens = generate_one(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
        )
        if hit_max_new_tokens:
            hit_max_count += 1
        gen_token_sum += gen_tokens
        gen_token_max = max(gen_token_max, gen_tokens)

        pred_jsons = extract_json_from_text(pred_text)
        gold_jsons = extract_json_from_text(gold_text)

        pred_json = pred_jsons[0] if len(pred_jsons) == 1 else None
        gold_json = gold_jsons[0] if len(gold_jsons) == 1 else None

        if pred_json is not None:
            json_ok += 1
            if validate_json_structure(pred_json):
                structure_ok += 1
            pred_ent_counts.append(_safe_list_len(pred_json.get("entities")))
            pred_rel_counts.append(_safe_list_len(pred_json.get("relations")))
        else:
            pred_json_invalid += 1

        if gold_json is not None:
            gold_ent_counts.append(_safe_list_len(gold_json.get("entities")))
            gold_rel_counts.append(_safe_list_len(gold_json.get("relations")))
        else:
            gold_json_invalid += 1

        e_f1 = 0.0
        r_f1 = 0.0
        if pred_json is not None and gold_json is not None:
            e_f1 = entities_f1(pred_json, gold_json)
            r_f1 = relations_f1(pred_json, gold_json)
        ent_f1_sum += e_f1
        rel_f1_sum += r_f1

        if save_f is not None:
            user_text = ""
            for m in messages:
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                    break
            record = {
                "idx": idx,
                "user": user_text,
                "gold_text": gold_text,
                "pred_text": pred_text,
                "hit_max_new_tokens": bool(hit_max_new_tokens),
                "gen_tokens": int(gen_tokens),
                "pred_json_ok": pred_json is not None,
                "pred_structure_ok": bool(pred_json is not None and validate_json_structure(pred_json)),
                "entities_f1": float(e_f1),
                "relations_f1": float(r_f1),
                "gold_entities_count": _safe_list_len(gold_json.get("entities")) if gold_json is not None else None,
                "gold_relations_count": _safe_list_len(gold_json.get("relations")) if gold_json is not None else None,
                "pred_entities_count": _safe_list_len(pred_json.get("entities")) if pred_json is not None else None,
                "pred_relations_count": _safe_list_len(pred_json.get("relations")) if pred_json is not None else None,
                "gold_json": gold_json,
                "pred_json": pred_json,
            }
            save_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if args.verbose:
            user_text = ""
            for m in messages:
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                    break
            print("=" * 80)
            print(f"[{i}/{len(indices)}] idx={idx}")
            print("USER (앞 300자):", user_text[:300].replace("\n", " "))
            print("PRED (앞 600자):", pred_text[:600].replace("\n", " "))
            print(f"hit_max_new_tokens={hit_max_new_tokens} (gen_tokens={gen_tokens}, max_new_tokens={args.max_new_tokens})")
            print(f"json_ok={pred_json is not None} structure_ok={pred_json is not None and validate_json_structure(pred_json)} ent_f1={e_f1:.3f} rel_f1={r_f1:.3f}")
            if gold_json is not None and pred_json is not None:
                print(
                    "counts(ent/rel) gold="
                    f"{_safe_list_len(gold_json.get('entities'))}/{_safe_list_len(gold_json.get('relations'))} "
                    "pred="
                    f"{_safe_list_len(pred_json.get('entities'))}/{_safe_list_len(pred_json.get('relations'))}"
                )

    n = len(indices)
    print("\n=== Summary ===")
    print(f"samples: {n}")
    print(f"json_ok: {json_ok}/{n} ({(json_ok/n*100):.1f}%)")
    print(f"structure_ok: {structure_ok}/{n} ({(structure_ok/n*100):.1f}%)")
    print(f"hit_max_new_tokens: {hit_max_count}/{n} ({(hit_max_count/n*100):.1f}%)")
    print(f"gen_tokens(avg/max): {(gen_token_sum/n):.1f}/{gen_token_max}")
    print(f"entities_f1(avg): {ent_f1_sum/n:.4f}")
    print(f"relations_f1(avg): {rel_f1_sum/n:.4f}")
    print("")
    print("=== Count Stats (mean/median/max) ===")
    print(f"pred_json_invalid: {pred_json_invalid}/{n} ({(pred_json_invalid/n*100):.1f}%)")
    print(f"gold_json_invalid: {gold_json_invalid}/{n} ({(gold_json_invalid/n*100):.1f}%)")
    print(f"gold_entities_count: {_fmt_mean_median_max(gold_ent_counts)}")
    print(f"gold_relations_count: {_fmt_mean_median_max(gold_rel_counts)}")
    print(f"pred_entities_count: {_fmt_mean_median_max(pred_ent_counts)}")
    print(f"pred_relations_count: {_fmt_mean_median_max(pred_rel_counts)}")
    if gold_rel_counts:
        gold_rel_zero = sum(1 for x in gold_rel_counts if x == 0)
        print(f"gold_relations_zero: {gold_rel_zero}/{len(gold_rel_counts)} ({(gold_rel_zero/len(gold_rel_counts)*100):.1f}%)")
    if pred_rel_counts:
        pred_rel_zero = sum(1 for x in pred_rel_counts if x == 0)
        print(f"pred_relations_zero: {pred_rel_zero}/{len(pred_rel_counts)} ({(pred_rel_zero/len(pred_rel_counts)*100):.1f}%)")

    if save_f is not None:
        save_f.close()
        print("")
        print(f"saved: {args.save_path}")


if __name__ == "__main__":
    main()
