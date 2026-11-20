"""통신량 → 입력 분야 / 출력 토큰 분류기. 수집한 트레이스로 학습·저장.

출력 토큰: StandardScaler + LogisticRegression, 사전 = 학습빈도 ≥ MIN_COUNT.
입력 분야: 같은 파이프라인, 라벨 = 분야.
창(window): 'none'(스텝 단독) / 'causal'(과거 -W..0, 실시간 복원용).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .traffic import volume_features, prompt_features, measured_features, measured_prompt_features

SEED = 0
MIN_COUNT = 5
TEST_SHARE = 0.2


def _pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def prompt_split(prompt_ids, seed=SEED, test_share=TEST_SHARE):
    """프롬프트 단위 분할(토큰 누출 방지)."""
    uniq = np.unique(prompt_ids)
    shuffled = np.random.default_rng(seed).permutation(uniq)
    test_set = set(shuffled[:int(len(shuffled) * test_share)].tolist())
    test = np.array([p in test_set for p in prompt_ids])
    return ~test, test


def causal_window(x, prompt_ids, step, W=2):
    """과거 -W..0 스텝 특징을 이어붙임(미래 없음, 실시간용)."""
    index = {(p, s): n for n, (p, s) in enumerate(zip(prompt_ids, step))}
    parts = []
    for off in range(-W, 1):
        part = np.zeros_like(x)
        for n, (p, s) in enumerate(zip(prompt_ids, step)):
            m = index.get((p, s + off))
            if m is not None:
                part[n] = x[m]
        parts.append(part)
    return np.concatenate(parts, axis=1)


def fit_output(x_train, y_train):
    values, counts = np.unique(y_train, return_counts=True)
    vocab = values[counts >= MIN_COUNT]
    keep = np.isin(y_train, vocab)
    return _pipe().fit(x_train[keep], y_train[keep]), vocab


def train_output(trace_path, experts, gpus, window="none", seed=SEED, save=None, source="computed"):
    """출력 토큰 분류기 학습·(선택)저장. 트레이스 키: tokens/routing/prompt_ids/step.

    source="measured"이면 EP 서빙에서 측정한 통신량(키 measured)을 그대로 쓴다.
    """
    d = np.load(trace_path)
    tokens, routing = d["tokens"], d["routing"]
    prompt_ids, step = d["prompt_ids"], d["step"]
    train, test = prompt_split(prompt_ids, seed)
    x = measured_features(d["measured"]) if source == "measured" else volume_features(routing, experts, gpus)
    if window == "causal":
        x = causal_window(x, prompt_ids, step)
    model, vocab = fit_output(x[train], tokens[train])
    acc = float((model.predict(x[test]) == tokens[test]).mean())
    values, counts = np.unique(tokens[train], return_counts=True)
    baseline = float((tokens[test] == values[counts.argmax()]).mean())
    meta = {"task": "output", "source": source, "window": window, "experts": int(experts), "gpus": int(gpus),
            "seed": seed, "dictionary_size": int(len(vocab)), "accuracy": acc,
            "baseline": baseline, "n_test": int(test.sum())}
    if save:
        import joblib
        joblib.dump({"model": model, "vocab": vocab, "meta": meta}, save)
    return meta


def train_domain(trace_path, experts, gpus, seed=SEED, save=None, source="computed"):
    """입력 분야 분류기. 트레이스 키: 고정길이 routing(P,L,T,k)+labels, 또는 prefill_routing+prefill_lengths+labels."""
    d = np.load(trace_path)
    labels = np.asarray(d["labels"])

    def key(name):                      # 예전 파일은 prefill_ 대신 e1_ 로 저장돼 있다
        return f"prefill_{name}" if f"prefill_{name}" in d.files else f"e1_{name}"

    if source == "measured":
        x = measured_prompt_features(d[key("measured")], d[key("lengths")])
    elif key("routing") in d.files:
        x = prompt_features(d[key("routing")], d[key("lengths")], experts, gpus)
    else:  # 고정길이 포맷 (P, layers, tokens, k)
        from .placement import dest_counts_feat
        r = d["routing"]
        dc = dest_counts_feat(r, experts, gpus)            # (P, layers, tokens, g-1)
        x = (dc.sum(axis=2) / r.shape[2]).reshape(len(r), -1).astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x)); cut = int(len(x) * (1 - TEST_SHARE))
    tr, te = idx[:cut], idx[cut:]
    model = _pipe().fit(x[tr], labels[tr])
    acc = float((model.predict(x[te]) == labels[te]).mean())
    meta = {"task": "domain", "source": source, "experts": int(experts), "gpus": int(gpus), "seed": seed,
            "accuracy": acc, "n_test": int(len(te)), "classes": sorted(set(labels.tolist()))}
    if save:
        import joblib
        joblib.dump({"model": model, "meta": meta}, save)
    return meta
