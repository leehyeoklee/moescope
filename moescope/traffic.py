"""라우팅 → 통신량 특징. 측정값이든 계산값이든 같은 함수를 통과시키면 같은 특징이 된다."""
import numpy as np

from .placement import dest_counts_feat


def volume_features(routing, experts: int, gpus: int):
    """출력(decode) 특징: 스텝별 라우팅 → 층별 목적지 GPU 송신 수.

    routing(N, layers, k) → (N, layers*(gpus-1)).
    """
    dc = dest_counts_feat(routing, experts, gpus)
    return dc.reshape(len(dc), -1).astype(np.float32)


def prompt_features(routing, lengths, experts: int, gpus: int):
    """입력(prefill) 특징: 프롬프트 전 토큰 송신 수 합 / 길이 정규화.

    routing(sum_tokens, layers, k), lengths(P,) → (P, layers*(gpus-1)).
    """
    per_token = dest_counts_feat(routing, experts, gpus)
    lengths = np.asarray(lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    rows = [per_token[s:s + n].sum(axis=0).reshape(-1) / n for s, n in zip(starts, lengths)]
    return np.stack(rows).astype(np.float32)


def measured_features(measured):
    """EP 서빙에서 단계별로 측정한 통신량 (N, layers, gpus) → (N, layers*(gpus-1)).

    요청을 가진 rank 0으로 가는 값은 GPU 간 전송이 아니므로 제외한다(volume_features와 같은 형태).
    """
    m = np.asarray(measured)[..., 1:]
    return m.reshape(len(m), -1).astype(np.float32)


def measured_prompt_features(measured, lengths):
    """EP 서빙의 prefill에서 프롬프트 단위로 측정한 통신량 (P, layers, gpus) → 입력 토큰 수로 나눈 값."""
    m = np.asarray(measured)[..., 1:].astype(np.float32) / np.asarray(lengths, np.float32)[:, None, None]
    return m.reshape(len(m), -1)
