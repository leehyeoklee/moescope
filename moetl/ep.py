"""실제 Expert 병렬 all_to_all 디스패치 + 통신량 측정 (gloo/NCCL 공용).

dropless(per (token,expert)): 토큰을 선택 Expert가 있는 rank로 보낸다.
측정값 = 이 rank가 각 목적지 rank로 보낸 행 수(input_split_sizes).
모든 rank 측정값의 합 == placement.dest_counts_full(계산값)  ← 핵심 주장의 코드 입증.
gloo(CPU)와 NCCL(GPU)은 같은 집합통신 API라 백엔드만 바뀐다.
"""
import numpy as np
import torch
import torch.distributed as dist

from .placement import rank_of


def _a2a(send, in_splits, out_splits, group=None):
    out = send.new_empty((int(sum(out_splits)),) + tuple(send.shape[1:]))
    dist.all_to_all_single(out, send.contiguous(),
                           output_split_sizes=list(map(int, out_splits)),
                           input_split_sizes=list(map(int, in_splits)), group=group)
    return out


def ep_layer(local_hidden, local_routing, local_gate, experts, gpus,
             expert_fn, rank, group=None):
    """한 MoE 층의 EP 디스패치→Expert 계산→결합. (out, 측정 송신 수) 반환.

    local_hidden (n_local, H)   이 rank가 맡은 토큰 입력
    local_routing(n_local, k)   top-k 전역 Expert id
    local_gate   (n_local, k)   결합 가중치(None이면 1)
    expert_fn(global_expert_id, rows) -> rows   로컬 Expert 계산
    """
    device = local_hidden.device
    H = local_hidden.shape[1]
    n_local, k = local_routing.shape
    if local_gate is None:
        local_gate = torch.ones_like(local_routing, dtype=local_hidden.dtype)

    tok_idx = torch.arange(n_local, device=device).repeat_interleave(k)
    slot_idx = torch.arange(k, device=device).repeat(n_local)
    exp_flat = local_routing.reshape(-1)
    dest = torch.as_tensor(rank_of(exp_flat.cpu().numpy(), experts, gpus), device=device)

    order = torch.argsort(dest, stable=True)
    in_splits = torch.bincount(dest[order], minlength=gpus)   # (gpus,) 측정값
    send_hidden = local_hidden[tok_idx[order]]
    send_expert = exp_flat[order]
    send_origin = torch.stack([tok_idx[order], slot_idx[order],
                               torch.full_like(tok_idx[order], rank)], dim=1)

    out_splits = torch.empty_like(in_splits)
    dist.all_to_all_single(out_splits, in_splits, group=group)

    recv_hidden = _a2a(send_hidden, in_splits, out_splits, group)
    recv_expert = _a2a(send_expert, in_splits, out_splits, group)
    recv_origin = _a2a(send_origin, in_splits, out_splits, group)

    proc = recv_hidden.clone()
    if recv_hidden.numel() > 0:
        for e in torch.unique(recv_expert):
            m = recv_expert == e
            proc[m] = expert_fn(int(e), recv_hidden[m])

    back = _a2a(proc, out_splits, in_splits, group)
    back_origin = _a2a(recv_origin, out_splits, in_splits, group)

    out = torch.zeros((n_local, H), dtype=local_hidden.dtype, device=device)
    o_tok = back_origin[:, 0].long()
    o_slot = back_origin[:, 1].long()
    w = local_gate[o_tok, o_slot].unsqueeze(1).to(out.dtype)
    out.index_add_(0, o_tok, back * w)
    return out, in_splits
