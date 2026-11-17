"""라우팅 → 통신량 특징. 측정값이든 계산값이든 같은 함수를 통과시키면 같은 특징이 된다."""
import numpy as np

from .placement import dest_counts_feat


def volume_features(routing, experts: int, gpus: int):
    """출력(E2) 특징: 스텝별 라우팅 → 층별 목적지 GPU 송신 수.

    routing(N, layers, k) → (N, layers*(gpus-1)).
    """
    dc = dest_counts_feat(routing, experts, gpus)
    return dc.reshape(len(dc), -1).astype(np.float32)


def prompt_features(e1_routing, lengths, experts: int, gpus: int):
    """입력(E1) 특징: 프롬프트 전 토큰 송신 수 합 / 길이 정규화.

    e1_routing(sum_tokens, layers, k), lengths(P,) → (P, layers*(gpus-1)).
    """
    per_token = dest_counts_feat(e1_routing, experts, gpus)
    lengths = np.asarray(lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    rows = [per_token[s:s + n].sum(axis=0).reshape(-1) / n for s, n in zip(starts, lengths)]
    return np.stack(rows).astype(np.float32)
