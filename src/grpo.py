"""
Text2Graph GRPO 학습 스크립트 (Unsloth + TRL)
---------------------------------------------

이 스크립트는 `text2graph.json` (prompt/messages + solution(JSON)) 데이터로 GRPO를 수행합니다.

핵심 포인트:
- Qwen3는 기본적으로 thinking(<think>...</think>) 모드가 켜질 수 있어 JSON-only 작업에는 방해가 됩니다.
  => tokenizer.apply_chat_template를 래핑해서 enable_thinking=False가 기본이 되게 처리합니다.
- vLLM(colocate)로 rollout 생성 속도를 올립니다. (TRL 내장 vLLM)
- SFT 체크포인트(LoRA adapter)를 초기 가중치로 로드해서 GRPO를 이어갑니다.
"""

from __future__ import annotations

import os
import sys

# ---- TorchDynamo / torch.compile 관련 안전장치 ----
# Unsloth+Torch 조합에서 Dynamo symbolic shape 에러가 날 수 있어 선택적으로 비활성화합니다.
_PRE_DISABLE_TORCHDYNAMO = (
    "--disable_torchdynamo" in sys.argv
    or os.environ.get("TORCHDYNAMO_DISABLE", "0") in ("1", "True", "true")
)
_PRE_DYNAMO_SUPPRESS_ERRORS = (
    "--dynamo_suppress_errors" in sys.argv
    or os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", "0") in ("1", "True", "true")
)

# Unsloth가 내부적으로 torch.compile 경로를 타는 경우가 있어,
# dynamo를 끄는 플래그를 줬다면 Unsloth compile도 같이 꺼주는 게 안전합니다.
if _PRE_DISABLE_TORCHDYNAMO and "UNSLOTH_COMPILE_DISABLE" not in os.environ:
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import argparse
import json
import logging
import random
from typing import Any, Dict, List

import torch
try:  # pragma: no cover
    import torch._dynamo

    if _PRE_DISABLE_TORCHDYNAMO:
        torch._dynamo.config.disable = True
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    if _PRE_DYNAMO_SUPPRESS_ERRORS:
        torch._dynamo.config.suppress_errors = True
        os.environ["TORCHDYNAMO_SUPPRESS_ERRORS"] = "1"
except Exception:
    pass

from datasets import Dataset as HFDataset
from peft import PeftModel
from transformers import AutoTokenizer
from trl import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer as TRL_GRPOTrainer

logger = logging.getLogger(__name__)


class SafeGRPOTrainer(TRL_GRPOTrainer):
    """
    Unsloth+TRL GRPO 경로에서 rollout 생성 시 model.for_inference()로 전환된 상태가
    학습 스텝에 남아 있으면 loss가 grad_fn 없이 만들어져
    `element 0 of tensors does not require grad` 에러가 날 수 있습니다.

    training_step 시작 시점에 강제로 training 모드로 되돌려 안정성을 확보합니다.
    """

    def training_step(self, model, inputs, *args, **kwargs):
        try:
            if hasattr(model, "for_training"):
                try:
                    model.for_training(use_gradient_checkpointing=True)
                except TypeError:
                    model.for_training()
            model.train()
        except Exception:
            pass
        with torch.enable_grad():
            return super().training_step(model, inputs, *args, **kwargs)


def _completion_texts(completions) -> List[str]:
    """
    TRL GRPOTrainer에서 completions 포맷은 2가지 중 하나:
    - conversational: [[{"role":"assistant","content":"..."}], ...]
    - standard: ["...", "...", ...]

    reward 함수는 텍스트만 필요하므로 여기서 통일합니다.
    """
    if not completions:
        return []
    first = completions[0]
    if isinstance(first, str):
        return completions
    # conversational: list[list[dict]]
    if isinstance(first, list) and first and isinstance(first[0], dict):
        return [c[0].get("content", "") if c and isinstance(c[0], dict) else "" for c in completions]
    # fallback
    return [str(c) for c in completions]


def patch_tokenizer_apply_chat_template(tokenizer, enable_thinking: bool):
    """
    TRL 내부에서 tokenizer.apply_chat_template를 호출할 때 enable_thinking 인자를 넘기지 않습니다.
    Qwen3는 enable_thinking의 default가 True라서 <think> 블록이 섞일 수 있어,
    여기서 기본값을 강제로 지정하는 래퍼를 씌웁니다.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return tokenizer

    original = tokenizer.apply_chat_template

    def wrapped(messages, **kwargs):
        kwargs.setdefault("enable_thinking", enable_thinking)
        try:
            return original(messages, **kwargs)
        except TypeError:
            # enable_thinking을 지원하지 않는 토크나이저인 경우
            kwargs.pop("enable_thinking", None)
            return original(messages, **kwargs)

    tokenizer.apply_chat_template = wrapped
    return tokenizer


def extract_json_from_text(text: str) -> List[Dict[str, Any]]:
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
                extracted_json = text[json_start : idx + 1]
                try:
                    extracted_jsons.append(json.loads(extracted_json))
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
    for relation in data["relations"]:
        if not isinstance(relation, dict) or not relation_required_keys.issubset(relation.keys()):
            return False
        if not isinstance(relation["head"], str) or not isinstance(relation["tail"], str) or not isinstance(
            relation["type"], str
        ):
            return False

    return True


def compute_f1(pred_set: set, true_set: set) -> float:
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def _kw_scalar(kwargs: Any, key: str, i: int, default: float) -> float:
    """
    TRL은 reward 함수에 dataset column을 kwargs로 넘기는데, 배치 단위로 list가 들어옵니다.
    이 유틸은 scalar/list 모두에서 안전하게 float로 꺼냅니다.
    """
    if not isinstance(kwargs, dict):
        return float(default)
    v = kwargs.get(key, default)
    if isinstance(v, list):
        try:
            return float(v[i])
        except Exception:
            return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def get_entities_f1_score(pred_entities, true_entities) -> float:
    pred_set = {f"{entity.get('text')}_{entity.get('type')}" for entity in pred_entities if isinstance(entity, dict)}
    true_set = {f"{entity.get('text')}_{entity.get('type')}" for entity in true_entities if isinstance(entity, dict)}
    return compute_f1(pred_set, true_set)


def get_relations_f1_score(pred_relations, true_relations) -> float:
    pred_set = {f"{rel.get('head')}_{rel.get('tail')}_{rel.get('type')}" for rel in pred_relations if isinstance(rel, dict)}
    true_set = {f"{rel.get('head')}_{rel.get('tail')}_{rel.get('type')}" for rel in true_relations if isinstance(rel, dict)}
    return compute_f1(pred_set, true_set)


# ---- Reward functions ----
# TRL은 reward_func(prompts=..., completions=..., completion_ids=..., solution=...) 형태로 호출합니다.

def json_consistency_reward(completions, solution, **kwargs):
    contents = _completion_texts(completions)
    rewards = []
    for content, _sol in zip(contents, solution):
        extracted_jsons = extract_json_from_text(content)
        rewards.append(1.0 if len(extracted_jsons) == 1 else 0.0)
    return rewards


def json_structure_reward(completions, solution, **kwargs):
    contents = _completion_texts(completions)
    rewards = []
    for content, _sol in zip(contents, solution):
        extracted_jsons = extract_json_from_text(content)
        if len(extracted_jsons) == 1 and validate_json_structure(extracted_jsons[0]):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def f1_entities_reward(completions, solution, **kwargs):
    contents = _completion_texts(completions)
    f1_power_kw = kwargs.get("f1_power", 0.5) if isinstance(kwargs, dict) else 0.5
    rewards = []
    for i, (content, sol) in enumerate(zip(contents, solution)):
        # TRL은 dataset 컬럼을 kwargs로 넘길 때 batch 단위 list로 전달합니다.
        # - f1_power가 list면 샘플별 값을 사용하고, 아니면 scalar로 처리합니다.
        if isinstance(f1_power_kw, list):
            try:
                f1_power = float(f1_power_kw[i])
            except Exception:
                f1_power = 0.5
        else:
            try:
                f1_power = float(f1_power_kw)
            except Exception:
                f1_power = 0.5

        extracted_jsons_pred = extract_json_from_text(content)
        extracted_jsons_true = extract_json_from_text(sol)

        if len(extracted_jsons_pred) == 1 and len(extracted_jsons_true) == 1:
            json_pred = extracted_jsons_pred[0]
            json_true = extracted_jsons_true[0]
            try:
                f1 = get_entities_f1_score(json_pred.get("entities", []), json_true.get("entities", []))
                rewards.append(float(max(0.0, min(1.0, f1))) ** f1_power)
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def f1_relations_reward(completions, solution, **kwargs):
    """
    Relations reward (shaped):
    - head/tail만 맞으면 부분점수 (type 무시)
    - head/tail까지 맞춘 것들에 대해 type까지 맞으면 추가 점수
    - head/tail 방향이 뒤집힌 경우는 discount된 점수

    이렇게 하면 relation type 동의어/변형 때문에 reward가 0으로 떨어지는 현상을 완화하고,
    점진적으로 gold의 type 표현을 따라가도록 유도할 수 있습니다.
    """
    contents = _completion_texts(completions)
    rewards: List[float] = []
    for i, (content, sol) in enumerate(zip(contents, solution)):
        f1_power = _kw_scalar(kwargs, "f1_power", i, 0.5)
        ht_weight = _kw_scalar(kwargs, "rel_ht_weight", i, 0.7)
        type_weight = _kw_scalar(kwargs, "rel_type_weight", i, 0.3)
        swapped_discount = _kw_scalar(kwargs, "rel_swapped_discount", i, 0.5)

        pred_list = extract_json_from_text(content)
        gold_list = extract_json_from_text(sol)
        if len(pred_list) != 1 or len(gold_list) != 1:
            rewards.append(0.0)
            continue
        pred = pred_list[0]
        gold = gold_list[0]
        if not (isinstance(pred, dict) and isinstance(gold, dict)):
            rewards.append(0.0)
            continue

        pred_rels = pred.get("relations") if isinstance(pred.get("relations"), list) else []
        gold_rels = gold.get("relations") if isinstance(gold.get("relations"), list) else []

        def _pairs_and_types(rel_list):
            ht = set()
            ht_swapped = set()
            ht_to_type = {}
            for r in rel_list:
                if not isinstance(r, dict):
                    continue
                h = r.get("head")
                t = r.get("tail")
                ty = r.get("type")
                if not isinstance(h, str) or not isinstance(t, str):
                    continue
                ht.add((h, t))
                ht_swapped.add((t, h))
                if isinstance(ty, str):
                    ht_to_type[(h, t)] = ty
            return ht, ht_swapped, ht_to_type

        gold_ht, _gold_swapped, gold_ht2type = _pairs_and_types(gold_rels)
        pred_ht, pred_swapped, pred_ht2type = _pairs_and_types(pred_rels)

        # head/tail F1 (directional), plus discounted swapped F1
        ht_f1 = compute_f1(pred_ht, gold_ht)
        swapped_f1 = compute_f1(pred_swapped, gold_ht)  # pred swapped matches gold directional
        ht_score = max(float(ht_f1), float(swapped_discount) * float(swapped_f1))

        # type accuracy on matched directional head/tail pairs
        matched = pred_ht & gold_ht
        if not matched:
            type_acc = 0.0
        else:
            correct = 0
            total = 0
            for pair in matched:
                gt = gold_ht2type.get(pair)
                pt = pred_ht2type.get(pair)
                if isinstance(gt, str) and isinstance(pt, str):
                    total += 1
                    if gt == pt:
                        correct += 1
            type_acc = float(correct / max(total, 1))

        combined = float(max(0.0, min(1.0, float(ht_weight) * ht_score + float(type_weight) * type_acc)))
        rewards.append(combined ** float(max(0.0, f1_power)))

    return rewards


def length_penalty_reward(completions, completion_ids=None, **kwargs):
    """
    너무 긴 출력(끝을 못 내고 늘어지는 케이스)을 약하게 패널티.
    - completion_ids는 TRL이 전달해주는 토큰 id 리스트(list[list[int]])입니다.
    """
    contents = _completion_texts(completions)
    if completion_ids is None:
        return [0.0 for _ in contents]
    rewards = []
    for content, ids in zip(contents, completion_ids):
        # JSON이 이미 잘 나오면 길이 패널티를 주지 않습니다.
        extracted = extract_json_from_text(content)
        if len(extracted) == 1 and validate_json_structure(extracted[0]):
            rewards.append(0.0)
        else:
            # 길수록 더 음수. scale은 과하게 주면 f1 학습을 방해하니 약하게.
            rewards.append(-0.0005 * float(len(ids)))
    return rewards


reward_funcs_registry = {
    "json_consistency": json_consistency_reward,
    "json_structure": json_structure_reward,
    # 아래 3개는 "부분점수(soft reward)" 계열
    "json_progress": None,  # 아래에서 정의 후 덮어씀
    "count_coverage": None,
    "type_consistency": None,
    "f1_ents": f1_entities_reward,
    "f1_rels": f1_relations_reward,
    "length_penalty": length_penalty_reward,
}

def json_progress_reward(completions, solution, **kwargs):
    """
    JSON을 '어디까지' 맞췄는지에 따라 0~1의 부분점수를 줍니다.
    - 파싱 실패면 0
    - 파싱 성공 후 스키마/필드 타입이 맞을수록 점수 상승
    """
    contents = _completion_texts(completions)
    rewards: List[float] = []
    for content, _sol in zip(contents, solution):
        extracted = extract_json_from_text(content)
        if len(extracted) != 1:
            # 파싱 가능한 JSON이 아직 없을 때도, "형식으로 가는 방향"을 약하게 보상해
            # 초반 학습이 0-reward에 붙잡히지 않도록 합니다.
            score = 0.0
            if "{" in content:
                score += 0.05
            if "\"entities\"" in content:
                score += 0.05
            if "\"relations\"" in content:
                score += 0.05
            # 너무 긴 잡음은 패널티(길이 패널티가 별도로 있어 여기서는 약하게)
            rewards.append(float(max(0.0, min(0.15, score))))
            continue

        obj = extracted[0]
        score = 0.0

        # top-level keys
        has_entities = isinstance(obj, dict) and "entities" in obj and isinstance(obj.get("entities"), list)
        has_relations = isinstance(obj, dict) and "relations" in obj and isinstance(obj.get("relations"), list)
        if has_entities:
            score += 0.2
        if has_relations:
            score += 0.2

        # entities field/type correctness
        if has_entities:
            ents = obj.get("entities", [])
            if len(ents) == 0:
                score += 0.05  # 빈 리스트라도 형식은 맞춘 것
            else:
                ok = 0
                for e in ents:
                    if not isinstance(e, dict):
                        continue
                    if not {"id", "text", "type"}.issubset(e.keys()):
                        continue
                    if not isinstance(e.get("id"), int):
                        continue
                    if not isinstance(e.get("text"), str) or not isinstance(e.get("type"), str):
                        continue
                    ok += 1
                score += 0.25 * (ok / max(len(ents), 1))

        # relations field/type correctness
        if has_relations:
            rels = obj.get("relations", [])
            if len(rels) == 0:
                score += 0.05
            else:
                ok = 0
                for r in rels:
                    if not isinstance(r, dict):
                        continue
                    if not {"head", "tail", "type"}.issubset(r.keys()):
                        continue
                    if not isinstance(r.get("head"), str) or not isinstance(r.get("tail"), str) or not isinstance(
                        r.get("type"), str
                    ):
                        continue
                    ok += 1
                score += 0.25 * (ok / max(len(rels), 1))

        # normalize to 0..1
        rewards.append(float(max(0.0, min(1.0, score))))
    return rewards


def count_coverage_reward(completions, solution, **kwargs):
    """
    정답 대비 엔티티/관계 개수가 얼마나 비슷한지에 대한 soft reward (0~1).
    - F1이 0인 초기 구간에서도 "너무 적게/너무 많이" 뽑는 것을 줄이는 데 도움.
    """
    contents = _completion_texts(completions)
    rewards: List[float] = []
    for content, sol in zip(contents, solution):
        pred_list = extract_json_from_text(content)
        gold_list = extract_json_from_text(sol)
        if len(pred_list) != 1 or len(gold_list) != 1:
            rewards.append(0.0)
            continue
        pred = pred_list[0]
        gold = gold_list[0]
        if not (isinstance(pred, dict) and isinstance(gold, dict)):
            rewards.append(0.0)
            continue
        pred_ents = pred.get("entities") if isinstance(pred.get("entities"), list) else []
        pred_rels = pred.get("relations") if isinstance(pred.get("relations"), list) else []
        gold_ents = gold.get("entities") if isinstance(gold.get("entities"), list) else []
        gold_rels = gold.get("relations") if isinstance(gold.get("relations"), list) else []

        def _count_score(n_pred: int, n_true: int) -> float:
            denom = max(int(n_true), 1)
            # 1 - relative error, clamp to [0,1]
            return float(max(0.0, 1.0 - abs(int(n_pred) - int(n_true)) / denom))

        ent_score = _count_score(len(pred_ents), len(gold_ents))
        rel_score = _count_score(len(pred_rels), len(gold_rels))
        rewards.append(0.5 * ent_score + 0.5 * rel_score)
    return rewards


def type_consistency_reward(completions, solution, **kwargs):
    """
    타입/필드 정합성에 대한 부분점수(0~1).
    - entity.id 유니크/정수 여부
    - relation.head/tail이 pred entities의 text를 참조하는지
    """
    contents = _completion_texts(completions)
    rewards: List[float] = []
    for content, _sol in zip(contents, solution):
        pred_list = extract_json_from_text(content)
        if len(pred_list) != 1:
            rewards.append(0.0)
            continue
        pred = pred_list[0]
        if not isinstance(pred, dict):
            rewards.append(0.0)
            continue
        ents = pred.get("entities") if isinstance(pred.get("entities"), list) else []
        rels = pred.get("relations") if isinstance(pred.get("relations"), list) else []

        # entity id uniqueness
        ids: List[int] = []
        texts: List[str] = []
        for e in ents:
            if not isinstance(e, dict):
                continue
            if isinstance(e.get("id"), int):
                ids.append(int(e["id"]))
            if isinstance(e.get("text"), str):
                texts.append(e["text"])
        id_unique_score = 0.0
        if len(ids) > 0:
            id_unique_score = float(len(set(ids)) / len(ids))

        # relation head/tail must refer to some entity text (string match)
        text_set = set(texts)
        ref_ok = 0
        ref_total = 0
        for r in rels:
            if not isinstance(r, dict):
                continue
            head = r.get("head")
            tail = r.get("tail")
            if not isinstance(head, str) or not isinstance(tail, str):
                continue
            ref_total += 1
            if head in text_set and tail in text_set:
                ref_ok += 1
        ref_score = float(ref_ok / ref_total) if ref_total > 0 else 0.0

        rewards.append(0.5 * id_unique_score + 0.5 * ref_score)
    return rewards


# registry placeholder overwrite
reward_funcs_registry["json_progress"] = json_progress_reward
reward_funcs_registry["count_coverage"] = count_coverage_reward
reward_funcs_registry["type_consistency"] = type_consistency_reward


def _norm_rel_type(t: Any) -> str:
    if not isinstance(t, str):
        return ""
    return " ".join(t.strip().lower().split())


def gold_type_bonus_reward(completions, solution, **kwargs):
    """
    Soft reward: 정답(gold)에서 등장한 relation type이면 보너스.

    - 예측이 gold type vocabulary를 '따라 쓰도록' 유도하지만,
      gold에 없는 type을 썼다고 해서 패널티는 주지 않습니다(0점).
    - 점수는 0~1:
      (gold에 존재하는 type을 사용한 relation 개수) / (전체 relation 개수)
    """
    contents = _completion_texts(completions)
    rewards: List[float] = []

    for content, sol in zip(contents, solution):
        gold_jsons = extract_json_from_text(sol if isinstance(sol, str) else str(sol))
        if len(gold_jsons) != 1 or not isinstance(gold_jsons[0], dict):
            rewards.append(0.0)
            continue

        gold_types = {
            _norm_rel_type(r.get("type"))
            for r in gold_jsons[0].get("relations", [])
            if isinstance(r, dict)
        }
        gold_types.discard("")
        if not gold_types:
            rewards.append(0.0)
            continue

        pred_jsons = extract_json_from_text(content)
        if len(pred_jsons) != 1 or not isinstance(pred_jsons[0], dict):
            rewards.append(0.0)
            continue

        rels = pred_jsons[0].get("relations", [])
        if not isinstance(rels, list) or len(rels) == 0:
            rewards.append(0.0)
            continue

        total = 0
        hit = 0
        for r in rels:
            if not isinstance(r, dict):
                continue
            t = _norm_rel_type(r.get("type"))
            if not t:
                continue
            total += 1
            if t in gold_types:
                hit += 1
        rewards.append(float(hit / max(total, 1)))

    return rewards


reward_funcs_registry["gold_type_bonus"] = gold_type_bonus_reward


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="GRPO training for Text2Graph (Unsloth + TRL)")

    # Data / model
    parser.add_argument("--data_path", type=str, default="text2graph.json")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen3-0.6B")
    parser.add_argument("--sft_adapter", type=str, default=None, help="SFT LoRA adapter checkpoint dir (e.g., out_sft/checkpoint-1000)")
    parser.add_argument("--output_dir", type=str, default="out_grpo")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="디버깅/빠른 실험용: train split에서 앞쪽 N개만 사용",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="디버깅/빠른 실험용: eval split에서 앞쪽 N개만 사용",
    )

    # Qwen3 thinking
    parser.add_argument("--enable_thinking", action="store_true", help="Enable Qwen3 thinking mode (default: disabled).")

    # Unsloth / model loading
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Max sequence length for Unsloth model patching.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4bit (QLoRA).")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--use_vllm", action="store_true", help="Use TRL's vLLM rollout (faster, requires vllm install).")

    # Rewards
    parser.add_argument(
        "--reward_funcs",
        nargs="+",
        default=["json_progress", "count_coverage", "type_consistency", "f1_ents", "f1_rels", "length_penalty"],
        choices=sorted(reward_funcs_registry.keys()),
    )
    parser.add_argument(
        "--reward_weights",
        nargs="+",
        type=float,
        default=None,
        help="reward_funcs와 같은 순서/길이의 가중치 리스트. 미지정 시 기본값 사용.",
    )
    parser.add_argument(
        "--f1_power",
        type=float,
        default=0.5,
        help="F1 shaping: reward = f1 ** f1_power (기본 0.5 = sqrt).",
    )
    parser.add_argument("--rel_ht_weight", type=float, default=0.7, help="f1_rels: head/tail 부분점수 가중치 (0~1)")
    parser.add_argument("--rel_type_weight", type=float, default=0.3, help="f1_rels: type 정확도 보너스 가중치 (0~1)")
    parser.add_argument("--rel_swapped_discount", type=float, default=0.5, help="f1_rels: head/tail 스왑 매칭 discount (0~1)")

    # GRPO hyperparams (subset)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_strategy", type=str, default="no", choices=["no", "steps"])
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="GRPO 재시작용. 예: out_grpo/checkpoint-200",
    )

    # Generation behavior
    parser.add_argument("--stop_strings", nargs="+", default=["]}"], help="Stop strings for vLLM generation.")
    parser.add_argument("--mask_truncated_completions", action="store_true", help="Ignore truncated completions in loss (recommended).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    # Precision
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    # TorchDynamo options (workaround flags)
    parser.add_argument("--disable_torchdynamo", action="store_true")
    parser.add_argument("--dynamo_suppress_errors", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)

    # ---- Load dataset ----
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    random.shuffle(raw)
    split = int(len(raw) * args.train_split)
    train_data = raw[:split]
    eval_data = raw[split:]

    if args.max_train_samples is not None:
        train_data = train_data[: max(0, int(args.max_train_samples))]
    if args.max_eval_samples is not None:
        eval_data = eval_data[: max(0, int(args.max_eval_samples))]

    train_dataset = HFDataset.from_list(train_data)
    eval_dataset = HFDataset.from_list(eval_data) if args.eval_strategy != "no" else None

    logger.info("Dataset sizes - train=%d eval=%d", len(train_dataset), len(eval_data))

    # ---- Load model/tokenizer (Unsloth) ----
    try:
        from unsloth import FastLanguageModel
    except Exception as e:
        raise RuntimeError("Unsloth가 설치되어 있어야 합니다. `pip install unsloth` 후 다시 실행하세요.") from e

    model_init_kwargs = dict(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=False,
    )
    # 버전에 따라 gpu_memory_utilization 인자가 없을 수 있어 안전하게 처리
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            **model_init_kwargs,
        )
    except TypeError:
        model, tokenizer = FastLanguageModel.from_pretrained(**model_init_kwargs)

    tokenizer = patch_tokenizer_apply_chat_template(tokenizer, enable_thinking=args.enable_thinking)

    # ---- Load SFT adapter (optional) ----
    if args.sft_adapter:
        logger.info("Loading SFT adapter from %s", args.sft_adapter)
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    else:
        # SFT adapter가 없으면, GRPO용 LoRA를 새로 붙여 학습할 수 있도록 합니다.
        # (실무에서는 SFT→GRPO가 일반적이라 기본 흐름은 sft_adapter 사용을 권장)
        default_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=default_target_modules,
            lora_alpha=64,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    model.config.use_cache = False

    # vLLM는 선택사항. 설치가 안 되어 있으면 자동으로 비활성화합니다.
    if args.use_vllm:
        try:  # pragma: no cover
            import vllm  # noqa: F401
        except Exception:
            logger.warning("vLLM이 설치되어 있지 않아 --use_vllm을 비활성화합니다. (vLLM 없이도 동작은 합니다)")
            args.use_vllm = False

    # ---- Trainable params sanity check ----
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    logger.info("Trainable params: %d / %d (%.4f%%)", trainable, total, (100.0 * trainable / max(total, 1)))
    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters found (requires_grad=False for all params). "
            "If you're resuming from an SFT LoRA adapter, make sure it's loaded with is_trainable=True and that "
            "the adapter directory is correct."
        )

    # NOTE: TRL의 mask_truncated_completions는 'EOS가 안 나온 샘플'을 통째로 마스킹합니다.
    # vLLM이 없으면(stop strings를 강제하기 어렵고) EOS가 자주 안 나와서 배치 전체가 마스킹될 수 있고,
    # 그 경우 loss가 상수 0이 되어 `does not require grad` 에러가 날 수 있습니다.
    if args.mask_truncated_completions and not args.use_vllm:
        logger.warning(
            "mask_truncated_completions=True but use_vllm=False. Disabling mask_truncated_completions to avoid "
            "zero-mask batches causing a no-grad loss."
        )
        args.mask_truncated_completions = False

    # ---- Build GRPOConfig ----
    bf16 = args.mixed_precision == "bf16"
    fp16 = args.mixed_precision == "fp16"

    # reward 가중치 준비 (GRPOConfig.reward_weights로 전달)
    default_weight_map = {
        "json_consistency": 0.2,
        "json_structure": 0.2,
        "json_progress": 0.35,
        "count_coverage": 0.35,
        "type_consistency": 0.35,
        "gold_type_bonus": 0.35,
        "f1_ents": 1.5,
        "f1_rels": 1.0,
        "length_penalty": 1.0,
    }
    if args.reward_weights is not None:
        if len(args.reward_weights) != len(args.reward_funcs):
            raise ValueError(
                f"--reward_weights 길이({len(args.reward_weights)})가 --reward_funcs 길이({len(args.reward_funcs)})와 같아야 합니다."
            )
        reward_weights = args.reward_weights
    else:
        reward_weights = [float(default_weight_map.get(name, 1.0)) for name in args.reward_funcs]

    # vLLM 관련 인자들은 설치되어 있을 때만 쓰는 게 안전합니다.
    # (환경에 vllm이 없으면 GRPOTrainer가 ImportError를 냅니다.)
    generation_kwargs: Dict[str, Any] = {}
    if args.use_vllm:
        generation_kwargs = {"stop": args.stop_strings, "include_stop_str_in_output": True}
    else:
        # TRL은 non-vLLM 경로에서 generation_kwargs로 GenerationConfig를 생성합니다.
        # (즉 stopping_criteria 같은 generate() 전용 인자는 넣으면 ValueError가 납니다.)
        # model별 generation_config의 max_length 경고를 피하려면 명시적으로 max_new_tokens를 지정하는 게 안전합니다.
        generation_kwargs["max_new_tokens"] = int(args.max_completion_length)
        # TRL(non-vLLM)은 generation_kwargs로 GenerationConfig를 만들기 때문에
        # sampling 파라미터를 여기서 명시적으로 넣는 편이 안정적입니다.
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = float(args.temperature)
        generation_kwargs["top_p"] = float(args.top_p)
        generation_kwargs["top_k"] = int(args.top_k)
        generation_kwargs["min_p"] = float(args.min_p)
        generation_kwargs["repetition_penalty"] = float(args.repetition_penalty)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        remove_unused_columns=False,  # reward에서 solution을 쓰므로 반드시 False
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        reward_weights=reward_weights,
        generation_kwargs=generation_kwargs if generation_kwargs else None,
        mask_truncated_completions=args.mask_truncated_completions,
    )

    reward_funcs = [reward_funcs_registry[name] for name in args.reward_funcs]
    logger.info("Reward funcs: %s", args.reward_funcs)

    # TRL은 reward 함수에 dataset의 추가 컬럼을 kwargs로 넘깁니다.
    # f1 shaping 파라미터도 reward 함수가 볼 수 있게 constant column으로 주입합니다.
    train_dataset = train_dataset.add_column("f1_power", [args.f1_power] * len(train_dataset))
    train_dataset = train_dataset.add_column("rel_ht_weight", [args.rel_ht_weight] * len(train_dataset))
    train_dataset = train_dataset.add_column("rel_type_weight", [args.rel_type_weight] * len(train_dataset))
    train_dataset = train_dataset.add_column("rel_swapped_discount", [args.rel_swapped_discount] * len(train_dataset))
    if eval_dataset is not None:
        eval_dataset = eval_dataset.add_column("f1_power", [args.f1_power] * len(eval_dataset))
        eval_dataset = eval_dataset.add_column("rel_ht_weight", [args.rel_ht_weight] * len(eval_dataset))
        eval_dataset = eval_dataset.add_column("rel_type_weight", [args.rel_type_weight] * len(eval_dataset))
        eval_dataset = eval_dataset.add_column("rel_swapped_discount", [args.rel_swapped_discount] * len(eval_dataset))

    trainer = SafeGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Unsloth는 GRPOTrainer init 후 model.for_inference()를 호출하는 경우가 있어,
    # 학습 시작 전에 for_training()으로 확실히 되돌려 둡니다. (no-grad loss 방지)
    try:  # pragma: no cover
        if hasattr(trainer.model, "for_training"):
            try:
                trainer.model.for_training(use_gradient_checkpointing=True)
            except TypeError:
                trainer.model.for_training()
        trainer.model.train()
    except Exception:
        pass

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
