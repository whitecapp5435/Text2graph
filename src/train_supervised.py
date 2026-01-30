"""
Text2Graph SFT(지도학습) 스크립트
-------------------------------

이 파일은 `prompt`(대화 메시지 리스트) → `solution`(정답 JSON) 형태의 데이터를 사용해
언어모델을 **지도학습(SFT)** 으로 미리 학습시키는 코드입니다.

왜 SFT를 먼저 하냐?
- 작은 모델(SLM)은 처음부터 GRPO/RL을 걸면 출력이 JSON 포맷을 못 맞춰 reward가 0으로만 나오는 구간이 길어집니다.
- 그래서 SFT로 먼저 “항상 JSON 형태로 출력하는 습관(포맷)”과 “기본 text2graph 동작”을 익히게 만든 뒤,
  GRPO 단계에서 F1/json 구조 보상으로 성능을 더 끌어올리는 흐름을 씁니다.

입력 데이터 포맷(예: `text2graph.json`)은 대략 아래 형태를 기대합니다.
[
  {
    "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    "solution": "{\"entities\": [...], \"relations\": [...]}"   # 문자열/dict 모두 가능
  },
  ...
]

주의:
- 이 스크립트는 기본적으로 `labels = input_ids`로 두어(teacher forcing) 전체 토큰에 대해 loss를 계산합니다.
  다만 padding 토큰에 대해서는 loss를 계산하지 않도록 `labels=-100` 마스킹을 합니다.
- “프롬프트 토큰은 학습에서 제외하고, assistant 응답(정답 JSON) 구간만 학습”하고 싶다면
  별도의 마스킹 로직(또는 TRL의 completion-only collator)을 추가하는 것이 일반적입니다.

- Qwen3 계열은 `tokenizer.apply_chat_template(..., enable_thinking=True)`가 기본이라 `<think>...</think>`가
  자동으로 붙을 수 있습니다. JSON-only 학습을 원하면 `--enable_thinking`을 켜지 말고(기본 False),
  본 스크립트가 `enable_thinking=False`로 프롬프트를 만들도록 두는 것이 안전합니다.
"""

import os
import sys

# ---- TorchDynamo / torch.compile 관련 안전장치 ----
# Unsloth의 fused loss / torch.compile 경로가 특정 torch/nightly 조합에서 깨질 수 있습니다.
# (대표적으로 "Cannot call numel() on tensor with symbolic sizes/strides" 류)
#
# 아래 플래그들은 "가능하면 torch.compile을 끄거나, 컴파일 에러를 무시하고 eager로 폴백"하기 위한 옵션입니다.
# - CLI: --disable_torchdynamo / --dynamo_suppress_errors
# - ENV: TORCHDYNAMO_DISABLE=1 / TORCHDYNAMO_SUPPRESS_ERRORS=1
#
# 주의: argparse 파싱 이전이지만, sys.argv 문자열 검색으로 미리 적용합니다.
_PRE_DISABLE_TORCHDYNAMO = (
    "--disable_torchdynamo" in sys.argv
    or os.environ.get("TORCHDYNAMO_DISABLE", "0") in ("1", "True", "true")
)
_PRE_DYNAMO_SUPPRESS_ERRORS = (
    "--dynamo_suppress_errors" in sys.argv
    or os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", "0") in ("1", "True", "true")
)

# Unsloth는 transformers/trl/peft 등을 import하기 전에 먼저 import해야 패치(최적화)가 온전히 적용됩니다.
# - 설치되어 있지 않은 환경도 있으니, 실패해도 무시합니다.
try:  # pragma: no cover
    import unsloth  # noqa: F401
except Exception:
    pass

import json
import copy
import logging
import argparse
from tqdm import tqdm

import bitsandbytes as bnb
import torch
from torch.utils.data import Dataset

# torch._dynamo 설정(가능할 때만)
try:  # pragma: no cover
    import torch._dynamo

    if _PRE_DISABLE_TORCHDYNAMO:
        torch._dynamo.config.disable = True
        # 일부 라이브러리는 ENV도 확인하므로 같이 세팅
        os.environ["TORCHDYNAMO_DISABLE"] = "1"

    if _PRE_DYNAMO_SUPPRESS_ERRORS:
        torch._dynamo.config.suppress_errors = True
        os.environ["TORCHDYNAMO_SUPPRESS_ERRORS"] = "1"
except Exception:
    pass

from peft import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def apply_chat_template_compat(tokenizer, messages, **kwargs):
    """
    tokenizer.apply_chat_template 호출을 안전하게 래핑합니다.

    - Qwen3처럼 `enable_thinking` 인자를 지원하는 토크나이저도 있고,
      다른 모델 토크나이저는 해당 인자를 지원하지 않아 TypeError가 날 수 있습니다.
    - 그래서 enable_thinking을 포함해 호출해보고, 실패하면 enable_thinking만 제거해 다시 시도합니다.
    """
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)

# Define the dataset class
class Text2JSONDataset(Dataset):
    """
    JSON 파일에서 읽어온 list[dict] 데이터를 torch Dataset으로 감싸는 래퍼입니다.

    각 샘플은:
    - item["prompt"]: chat template에 들어갈 messages(list[{"role","content"}])
    - item["solution"]: 정답 JSON(문자열 또는 dict)
    """

    def __init__(self, data, tokenizer, max_length=1024, enable_thinking: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # ⚠️ 중요: item["prompt"]는 list 객체라서 그대로 extend하면 원본 데이터가 "제자리 수정"됩니다.
        # DataLoader가 여러 epoch를 돌면 같은 샘플에 assistant 메시지가 계속 누적되는 버그가 생길 수 있어
        # 반드시 복사본을 만들어 사용합니다.
        chat = list(item['prompt'])

        output = item['solution']

        # 정답(JSON)을 assistant 메시지로 붙여서 학습 시퀀스를 만듭니다.
        # 여기서 중요한 포인트:
        # - SFT에서 일반적으로 "프롬프트 부분"은 조건(condition)으로만 쓰고 loss는 주지 않습니다.
        # - 대신 assistant 답변(=정답 JSON) 구간만 loss를 주는 completion-only SFT가 흔합니다.
        #   (프롬프트를 예측/복제하도록 학습하는 건 불필요하고, 정답 쪽 학습 효율이 떨어질 수 있음)
        chat_with_answer = chat + [{"role": "assistant", "content": str(output)}]

        # 전체 시퀀스(프롬프트+정답) 렌더링
        full_text = apply_chat_template_compat(
            self.tokenizer,
            chat_with_answer,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )

        # 프롬프트(assistant 시작 토큰까지) 렌더링
        # add_generation_prompt=True는 "assistant가 이제부터 답을 생성해야 함"을 나타내는 특수 토큰/서식을 붙입니다.
        prompt_text = apply_chat_template_compat(
            self.tokenizer,
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        # max_length로 잘라서 고정 길이 padding을 합니다.
        # (긴 샘플이 많으면 학습이 느려지고, 너무 많이 잘리면 정답이 잘려 학습이 안 될 수 있습니다.)
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Causal LM 학습용 labels: 기본은 input_ids 복사본(teacher forcing).
        labels_ids = input_ids.clone()

        # ✅ completion-only 마스킹:
        # prompt_text 길이만큼(=assistant 답변이 시작되기 전까지) labels를 -100으로 설정해 loss에서 제외합니다.
        # 주의: prompt_text도 같은 max_length/truncation 기준으로 길이를 계산해야 경계가 어긋나지 않습니다.
        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )["input_ids"].squeeze(0)
        prompt_len = int(prompt_ids.shape[0])
        labels_ids[:prompt_len] = -100

        # ✅ padding 위치도 loss에서 제외합니다.
        # (Transformers의 CausalLM loss는 attention_mask를 자동으로 쓰지 않고 labels==-100만 무시합니다.)
        labels_ids[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_ids
        }

# Helper function to find linear module names
def find_all_linear_names(model, quantize=False):
    """
    LoRA를 걸 Linear 레이어 이름들을 자동으로 찾습니다.

    - quantize=True이면 bitsandbytes의 Linear4bit 레이어를 대상으로 하고,
      아니면 torch.nn.Linear를 대상으로 합니다.
    - 반환된 모듈 이름 목록(target_modules)을 LoRAConfig에 넣습니다.
    """
    cls = bnb.nn.Linear4bit if quantize else torch.nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# Main function
def main(args):
    logger.info("Starting script with configuration: %s", args)

    QUANTIZE = args.quantize
    USE_LORA = args.use_lora
    USE_UNSLOTH = args.use_unsloth
    
    model_path = args.model_path

    # Transformers에서 `device_map`은 보통 "auto" 또는 dict를 사용합니다.
    # (torch.device 객체를 그대로 넣으면 버전/환경에 따라 동작하지 않을 수 있습니다.)
    device_map = "auto" if torch.cuda.is_available() else None

    # ---- 모델/토크나이저 로딩 ----
    # 옵션 1) 기본: transformers로 로딩 + (선택) peft_config를 SFTTrainer에 넘겨 LoRA 적용
    # 옵션 2) --use_unsloth: Unsloth로 로딩 + (선택) Unsloth로 LoRA 적용 (이 경우 peft_config는 None)
    if USE_UNSLOTH:
        try:
            from unsloth import FastLanguageModel
        except Exception as e:
            raise RuntimeError(
                "Unsloth를 사용하려면 `unsloth` 패키지가 설치되어 있어야 합니다. "
                "Colab/로컬에서 `pip install unsloth` 후 다시 실행하세요."
            ) from e

        # Unsloth는 load_in_4bit로 4bit 로딩을 제어합니다.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=args.max_length,
            load_in_4bit=QUANTIZE,
            fast_inference=False,  # SFT 단계에서는 vLLM fast inference가 필수는 아님
            token=args.hf_token,
        )

        if USE_LORA:
            # Qwen 계열에서 일반적으로 사용하는 LoRA target_modules 기본값
            default_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
            target_modules = (
                [m.strip() for m in args.unsloth_target_modules.split(",") if m.strip()]
                if args.unsloth_target_modules
                else default_target_modules
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                target_modules=target_modules,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
                random_state=3407,
            )
        peft_config = None
    else:
        if QUANTIZE:
            # 4bit 양자화(QLoRA류) 설정: VRAM 절약을 위해 사용합니다.
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
            )
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            # torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            trust_remote_code=True,
            token=args.hf_token,
            # attn_implementation="flash_attention_2"
        )

        # 토크나이저 로드 + pad 토큰 설정
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # tokenizer.chat_template = CHAT_TEMPLATE

        if USE_LORA:
            # LoRA를 사용할 경우: target_modules를 자동 탐색 후 LoRAConfig를 생성합니다.
            modules = find_all_linear_names(model, quantize=QUANTIZE)

            peft_config = LoraConfig(
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=modules,
            )
        else:
            peft_config = None

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 데이터 로드 및 train/test split
    with open(args.data_path, encoding='utf-8') as f:
        data = json.load(f)
    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]
    train_dataset = Text2JSONDataset(
        train_data,
        tokenizer,
        max_length=args.max_length,
        enable_thinking=args.enable_thinking,
    )
    test_dataset = Text2JSONDataset(
        test_data,
        tokenizer,
        max_length=args.max_length,
        enable_thinking=args.enable_thinking,
    )

    logger.info("Dataset lengths - Train: %d, Test: %d", len(train_dataset), len(test_dataset))

    # Hugging Face TrainingArguments (학습 루프 공통 설정)
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_32bit",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=False,
        lr_scheduler_type=args.lr_scheduler_type,
        save_total_limit=3,
        report_to="none",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
    )

    # TRL의 SFTTrainer: (모델, 데이터, LoRAConfig)를 묶어 SFT 학습을 수행합니다.
    # TRL 버전에 따라 SFTTrainer 인자명이 다릅니다.
    # - 어떤 버전은 `tokenizer=`를,
    # - 어떤 버전은 `processing_class=`(tokenizer/processor)를 사용합니다.
    # 아래는 양쪽 모두에서 동작하도록 호환 처리합니다.
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
        )

    # 실제 학습 시작
    trainer.train()

    if USE_LORA:
        # LoRA로 학습한 경우, LoRA 어댑터를 베이스 모델에 merge해서 단일 모델로 저장합니다.
        # (추론/배포 시 LoRA 로딩이 번거롭다면 merge가 편합니다.)
        logger.info("Merging LoRA weights into the base model (since use_lora=True).")
        trainer.model = trainer.model.merge_and_unload()
        trainer.model.save_pretrained(os.path.join(args.output_dir, 'merged'))
        tokenizer.save_pretrained(os.path.join(args.output_dir, 'merged'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text2JSON Dataset Training Script")

    parser.add_argument('--model_path', type=str, required=True, help="Path to the model.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the training dataset.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save trained models.")
    parser.add_argument('--hf_token', type=str, required=False, help="Hugging Face authentication token.")
    parser.add_argument('--max_length', type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument('--num_train_epochs', type=int, default=2, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=2, help="Training batch size.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Enable gradient checkpointing.")
    parser.add_argument('--learning_rate', type=float, default=5e-6, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay.")
    parser.add_argument('--fp16', action='store_true', help="Enable FP16 training.")
    parser.add_argument('--bf16', action='store_true', help="Enable BF16 training.")
    parser.add_argument('--max_grad_norm', type=float, default=0.9, help="Maximum gradient norm.")
    parser.add_argument('--max_steps', type=int, default=-1, help="Maximum training steps.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument('--logging_steps', type=int, default=1, help="Logging steps.")
    parser.add_argument('--eval_steps', type=int, default=10000, help="Evaluation steps.")
    parser.add_argument('--save_steps', type=int, default=1000, help="Save steps.")
    parser.add_argument('--quantize', action='store_true', help="Enable quantization.")
    parser.add_argument('--use_lora', action='store_true', help="Enable LoRA training.")
    parser.add_argument('--use_unsloth', action='store_true', help="Use Unsloth FastLanguageModel for model loading/LoRA.")
    parser.add_argument('--unsloth_target_modules', type=str, default=None, help="Comma-separated LoRA target modules for Unsloth (default: Qwen-style proj layers).")
    parser.add_argument('--lora_r', type=int, default=64, help="LoRA rank (r).")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha.")
    parser.add_argument('--lora_dropout', type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument('--enable_thinking', action='store_true', help="Enable Qwen3 thinking mode in chat template (default: disabled).")
    parser.add_argument('--disable_torchdynamo', action='store_true', help="Disable TorchDynamo/torch.compile (workaround for some Unsloth+Torch issues).")
    parser.add_argument('--dynamo_suppress_errors', action='store_true', help="Suppress TorchDynamo errors and fallback to eager when compilation fails.")

    args = parser.parse_args()

    main(args)
