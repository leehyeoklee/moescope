"""모델 사양. 라우팅 수집 때와 동일한 체크포인트·revision을 고정한다.

분류기는 수집 때의 라우팅 분포에 맞춰 학습되므로, 서빙/측정 모델도 같은 체크포인트여야 한다.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    revision: str
    layers: int
    experts: int
    top_k: int
    shared_experts: int  # 공유 Expert(모든 rank에서 계산, 통신 없음 → 특징에서 제외)


MODELS = {
    "olmoe": ModelSpec("olmoe", "allenai/OLMoE-1B-7B-0125-Instruct",
                       "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e", 16, 64, 8, 0),
    "granite": ModelSpec("granite", "ibm-granite/granite-3.0-3b-a800m-instruct",
                         "51ced9b4d6ecb3a6671e2e52c937bc12a4a470a5", 32, 40, 8, 0),
    "qwen": ModelSpec("qwen", "Qwen/Qwen1.5-MoE-A2.7B-Chat",
                      "ec052fda178e241c7c443468d2fa1db6618996be", 24, 60, 4, 4),
}


def get(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model '{key}'. available: {list(MODELS)}")
    return MODELS[key]
