"""demo가 바로 쓰는 분류기(moescope/pretrained/)를 수집해 둔 라우팅 기록에서 만든다.

사용: python tools/build_pretrained.py <수집 폴더> <inputs.json> [GPU 수 ...]
  <수집 폴더>/output_<model>/v11_<model>.npz  입력 1,000개의 prefill 라우팅과 프롬프트 300개의 decode 라우팅
  <inputs.json>                                입력의 분야 라벨(e1[].label)과 decode 프롬프트의 학습/시험 구분(e2_split)

라우팅에서 계산한 통신량은 EP 서빙에서 측정한 통신량과 같으므로(serve-ep가 layer 호출마다 확인한다),
여기서 만든 분류기를 측정 통신량에 그대로 쓸 수 있다. GPU 수마다 통신량의 형태가 달라 따로 학습한다.
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn

from moescope import config
from moescope.classifiers import SEED, TEST_SHARE, _pipe, causal_window, fit_output
from moescope.traffic import prompt_features, volume_features

TOPICS = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet",
          "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(key, npz, inp, gpus, out_dir, base):
    spec, d = config.get(key), np.load(npz)
    info = dict(base, model=key, checkpoint=spec.repo, revision=spec.revision, gpus=gpus, experts=spec.experts,
                layers=spec.layers, seed=SEED, traces_sha256=sha(npz))

    # 입력 분야: prefill 통신량을 입력 토큰 수로 나눈 값
    labels = np.array([TOPICS[r["label"]] for r in inp["e1"]])[d["e1_ids"]]
    x = prompt_features(d["e1_routing"], d["e1_lengths"], spec.experts, gpus)
    idx = np.random.default_rng(SEED).permutation(len(x)); cut = int(len(x) * (1 - TEST_SHARE))
    dom = _pipe().fit(x[idx[:cut]], labels[idx[:cut]])
    acc_d = float((dom.predict(x[idx[cut:]]) == labels[idx[cut:]]).mean())
    joblib.dump({"model": dom, "meta": dict(info, task="domain", accuracy=acc_d, n_test=int(len(x) - cut))},
                out_dir / f"{key}_g{gpus}_domain.joblib", compress=3)

    # 응답 토큰: decode 단계별 통신량, 앞의 두 단계를 이어 붙인다
    tokens, pid, step = d["e2_tokens"], d["e2_prompt_ids"], d["e2_step"]
    x = causal_window(volume_features(d["e2_routing"], spec.experts, gpus), pid, step)
    test = np.isin(pid, inp["e2_split"]["test"])
    out, vocab = fit_output(x[~test], tokens[~test])
    acc_o = float((out.predict(x[test]) == tokens[test]).mean())
    joblib.dump({"model": out, "vocab": vocab,
                 "meta": dict(info, task="output", window="causal", accuracy=acc_o, n_test=int(test.sum()),
                              dictionary_size=int(len(vocab)))},
                out_dir / f"{key}_g{gpus}_output.joblib", compress=3)
    return acc_d, acc_o


def main(folder, inputs, gpus_list):
    inp = json.loads(Path(inputs).read_text(encoding="utf-8"))
    out_dir = Path(__file__).resolve().parent.parent / "moescope" / "pretrained"
    out_dir.mkdir(exist_ok=True)
    base = {"inputs_sha256": sha(inputs), "sklearn": sklearn.__version__, "numpy": np.__version__}
    rows = []
    for key in ("olmoe", "granite", "qwen"):
        npz = Path(folder) / f"output_{key}" / f"v11_{key}.npz"
        if not npz.exists():
            print(f"skip {key}: {npz} 없음"); continue
        for g in gpus_list:
            acc_d, acc_o = build(key, npz, inp, g, out_dir, base)
            rows.append({"model": key, "gpus": g, "domain_accuracy": round(acc_d, 4), "output_accuracy": round(acc_o, 4)})
            print(rows[-1], flush=True)
    (out_dir / "index.json").write_text(json.dumps(dict(base, classifiers=rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(g) for g in sys.argv[3:]] or [2, 4])
