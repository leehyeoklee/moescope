import numpy as np

from moetl.traffic import volume_features, prompt_features


def test_volume_features_shape():
    # N=10 steps, 16 layers, top-8, 64 experts, 4 gpus -> 16*(4-1)=48 dims
    rng = np.random.default_rng(0)
    routing = np.stack([np.stack([rng.choice(64, 8, replace=False) for _ in range(16)])
                        for _ in range(10)])
    x = volume_features(routing, 64, 4)
    assert x.shape == (10, 48)
    assert x.dtype == np.float32


def test_volume_features_values_match_manual():
    routing = np.array([[[0, 16, 17, 48, 1, 2, 3, 4]]])  # 1 step, 1 layer, k=8
    # ranks: 0,1,1,3,0,0,0,0 -> rank1:2, rank2:0, rank3:1
    x = volume_features(routing, 64, 4)
    assert x.tolist() == [[2.0, 0.0, 1.0]]


def test_prompt_features_normalized_by_length():
    # 2 prompts, lengths 2 and 1
    e1 = np.array([[[0, 16, 17, 48]], [[0, 16, 17, 48]], [[0, 16, 17, 48]]])  # (3 tok,1 layer,k4)
    x = prompt_features(e1, [2, 1], 64, 4)
    assert x.shape == (2, 3)  # 1 layer * (4-1)
    # prompt0: 두 토큰 합 [2,0,1]*2 /2 = [2,0,1]
    assert np.allclose(x[0], [2, 0, 1])
