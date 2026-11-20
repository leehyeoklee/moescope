"""EP 서빙 시험(gloo, CPU): 작은 임의 MoE 모델을 프로세스 여러 개에 나누어 실행한다.

확인하는 것
  1) EP로 실행한 결과가 한 프로세스에서 원래 모델로 실행한 결과와 같다.
  2) layer마다 실제 보낸 행 수(측정 통신량)가 라우팅에서 계산한 값과 같다.
transformers가 없으면 건너뛴다.
"""
import numpy as np
import pytest

pytest.importorskip("transformers")

from moescope import serve_ep                      # noqa: E402
from moescope.placement import dest_counts_full    # noqa: E402
from moescope.traffic import measured_features, volume_features    # noqa: E402

LAYERS, EXPERTS, TOP_K, HIDDEN = 3, 8, 2, 32
INPUTS = [[5, 17, 42, 8, 63, 2, 30], [9, 9, 71, 4], [11, 3, 55, 20, 6, 77, 1, 13, 48]]


def _launch(task, gpus, port, dtype="float32", family="olmoe"):
    loader = ("tiny", 0, LAYERS, EXPERTS, TOP_K, HIDDEN, dtype, family)
    job = {"task": task, "inputs": INPUTS, "max_new_tokens": 6, "check": True}
    return serve_ep.launch(loader, (LAYERS, EXPERTS, TOP_K), job, gpus, backend="gloo", port=port)


@pytest.mark.parametrize("gpus,port", [(2, 29711), (4, 29712)])
def test_decode_matches_single_process(gpus, port):
    data, report = _launch("decode", gpus, port)
    assert report["layer_calls_mismatch"] == 0, "측정 통신량 != 라우팅 계산값"
    assert report["token_agreement"] == 1.0, "EP로 생성한 토큰이 원래 모델과 다르다"
    assert report["forced_token_agreement"] == 1.0 and report["forced_max_logit_diff"] < 1e-5
    # 단계별 측정 통신량은 토큰 하나 몫이다: layer마다 합이 top-k
    assert data["measured"].shape == (len(data["tokens"]), LAYERS, gpus)
    assert (data["measured"].sum(-1) == TOP_K).all()
    computed = dest_counts_full(data["routing"], EXPERTS, gpus)
    assert np.array_equal(data["measured"], computed)
    # 분류기에 들어가는 값도 측정값과 계산값이 같다
    assert np.array_equal(measured_features(data["measured"]), volume_features(data["routing"], EXPERTS, gpus))


@pytest.mark.parametrize("gpus,port", [(2, 29721), (4, 29722)])
def test_prefill_matches_single_process(gpus, port):
    data, report = _launch("prefill", gpus, port)
    assert report["layer_calls_mismatch"] == 0
    assert report["max_logit_diff"] < 1e-5, "EP로 계산한 logits가 원래 모델과 다르다"
    assert data["prefill_measured"].shape == (len(INPUTS), LAYERS, gpus)
    lengths = np.array([len(x) for x in INPUTS])
    assert np.array_equal(data["prefill_lengths"], lengths)
    assert (data["prefill_measured"].sum(-1) == lengths[:, None] * TOP_K).all()


def test_half_precision_transport():
    """bfloat16 hidden state도 all_to_all로 오가며 결과가 유지된다."""
    data, report = _launch("decode", 2, 29731, dtype="bfloat16")
    assert report["layer_calls_mismatch"] == 0
    assert report["token_agreement"] == 1.0


@pytest.mark.parametrize("family,gpus,port", [("granite", 2, 29741), ("granite", 4, 29742),
                                              ("qwen", 2, 29743), ("qwen", 4, 29744)])
def test_other_model_families(family, gpus, port):
    """Granite, Qwen1.5-MoE와 같은 구조의 모델도 EP 결과가 원래 모델과 같다."""
    data, report = _launch("decode", gpus, port, family=family)
    assert report["layer_calls_mismatch"] == 0
    assert report["token_agreement"] == 1.0
    assert report["forced_token_agreement"] == 1.0 and report["forced_max_logit_diff"] < 1e-5
    assert np.array_equal(data["measured"], dest_counts_full(data["routing"], EXPERTS, gpus))


def test_single_gpu_has_no_cross_traffic():
    """GPU가 하나면 모든 Expert가 같은 GPU에 있어 다른 GPU로 가는 전송이 없다."""
    data, report = _launch("decode", 1, 29751)
    assert report["token_agreement"] == 1.0 and data["measured"].shape[-1] == 1


def test_gpu_count_must_divide_experts():
    with pytest.raises(ValueError):
        _launch("decode", 3, 29761)


def test_live_screen_keeps_results():
    """실시간 화면(chat)을 켜도 생성 토큰과 측정 통신량은 그대로다."""
    loader = ("tiny", 0, LAYERS, EXPERTS, TOP_K, HIDDEN, "float32", "olmoe")
    job = {"task": "decode", "inputs": INPUTS, "max_new_tokens": 6}
    plain, _ = serve_ep.launch(loader, (LAYERS, EXPERTS, TOP_K), job, 2, backend="gloo", port=29761)
    live, report = serve_ep.launch(loader, (LAYERS, EXPERTS, TOP_K), dict(job, live={"model": "tiny"}), 2,
                                   backend="gloo", port=29762)
    assert np.array_equal(plain["tokens"], live["tokens"]) and np.array_equal(plain["measured"], live["measured"])
    assert report["layer_calls_mismatch"] == 0 and report["layer_calls"] > 0
    assert [r["tokens"] for r in report["live"]] == [6, 6, 6]


def test_chat_takes_typed_prompts():
    """입력창 방식: 모델을 한 번 올린 채 문장을 차례로 받아 처리하고, exit로 끝난다."""
    from moescope import chat
    typed = iter(["hello there", "second question", "exit", "never read"])
    report = chat.run(gpus=2, tiny=True, max_new_tokens=5, port=29771, read=lambda: next(typed))
    assert report["layer_calls_mismatch"] == 0
    assert [(r["prompt"], r["tokens"]) for r in report["live"]] == [(0, 5), (1, 5)]
    assert next(typed) == "never read"


def test_text_span_keeps_tokens():
    """입력 가운데 문장 구간을 따로 넣어 통신량을 재도 생성 토큰은 한 번에 넣은 것과 같다."""
    loader = ("tiny", 0, LAYERS, EXPERTS, TOP_K, HIDDEN, "float32", "olmoe")
    job = {"task": "decode", "inputs": INPUTS, "max_new_tokens": 6}
    plain, _ = serve_ep.launch(loader, (LAYERS, EXPERTS, TOP_K), job, 2, backend="gloo", port=29781)
    split, report = serve_ep.launch(loader, (LAYERS, EXPERTS, TOP_K), dict(job, spans=[(2, 5), (0, 3), (3, 8)]), 2,
                                    backend="gloo", port=29782)
    assert report["layer_calls_mismatch"] == 0
    assert np.array_equal(plain["tokens"], split["tokens"])
    assert np.array_equal(plain["prefill_measured"], split["prefill_measured"])
    assert serve_ep.encode(None, "abc", "decode") == ([ord(c) % 97 for c in "abc"], (0, 3))


def test_prompt_file_with_labels(tmp_path):
    from moescope.cli import read_prompts
    f = tmp_path / "p.json"
    f.write_text('[{"text": "a", "label": "X"}, {"text": "b", "label": "Y"}]', encoding="utf-8")
    assert read_prompts(f) == (["a", "b"], ["X", "Y"])
    f.write_text('["a", "b"]', encoding="utf-8")
    assert read_prompts(f) == (["a", "b"], None)
