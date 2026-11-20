import numpy as np

from moescope.placement import rank_of, dest_counts_full, dest_counts_feat


def test_rank_of_linear():
    # 64 experts, 4 gpus -> 16 experts/gpu
    assert rank_of(0, 64, 4) == 0
    assert rank_of(15, 64, 4) == 0
    assert rank_of(16, 64, 4) == 1
    assert rank_of(63, 64, 4) == 3


def test_dest_counts_full_counts_assignments():
    # 한 토큰, top-4: experts 0,16,17,48 -> ranks 0,1,1,3
    routing = np.array([[0, 16, 17, 48]])
    full = dest_counts_full(routing, 64, 4)
    assert full.tolist() == [[1, 2, 0, 1]]
    assert full.sum() == 4  # top-k 개수와 같음


def test_dest_counts_feat_drops_rank0():
    routing = np.array([[0, 16, 17, 48]])
    feat = dest_counts_feat(routing, 64, 4)
    assert feat.tolist() == [[2, 0, 1]]  # rank0 제외


def test_sum_equals_topk():
    rng = np.random.default_rng(0)
    routing = np.stack([rng.choice(64, 8, replace=False) for _ in range(50)])
    assert (dest_counts_full(routing, 64, 4).sum(axis=-1) == 8).all()
