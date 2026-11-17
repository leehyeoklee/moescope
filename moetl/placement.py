"""Expert→rank 선형 배치, 라우팅→목적지 GPU별 토큰 수(계산값).

이 계산값이 실제 EP all_to_all 송신 수(측정값)와 일치해야 한다(핵심 주장).
정의:
  rank(expert) = expert // (experts // gpus)              # 선형 배치
  목적지 r 송신 수 = (토큰,선택Expert) 쌍 중 그 Expert가 r에 있는 것의 수 (top-k 축 합)
같은 rank에 선택 Expert가 둘이면 그 토큰은 r로 2번 간다(dropless, per (token,expert)).
"""
import numpy as np


def rank_of(expert, experts: int, gpus: int):
    return np.asarray(expert) // (experts // gpus)


def dest_counts_full(routing, experts: int, gpus: int):
    """라우팅(..., k) → 전체 rank별 송신 수 (..., gpus)."""
    gpu = np.asarray(routing).astype(np.int64) // (experts // gpus)
    return np.stack([(gpu == r).sum(axis=-1) for r in range(gpus)], axis=-1)


def dest_counts_feat(routing, experts: int, gpus: int):
    """특징용: rank 0(요청 보유 GPU) 제외 (..., gpus-1)."""
    gpu = np.asarray(routing).astype(np.int64) // (experts // gpus)
    return np.stack([(gpu == r).sum(axis=-1) for r in range(1, gpus)], axis=-1)
