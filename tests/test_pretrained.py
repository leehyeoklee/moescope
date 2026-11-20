"""저장소에 들어 있는 분류기가 chat 화면에 바로 쓰일 수 있는지 확인한다(모델 가중치 없이)."""
import numpy as np
import pytest

from moescope import chat, config


class _Rec:
    calls, mismatch = 0, 0


@pytest.mark.parametrize("model", ["olmoe", "granite", "qwen"])
@pytest.mark.parametrize("gpus", [2, 4])
def test_pretrained_classifiers_fit_the_traffic_shape(model, gpus, capsys):
    spec = config.get(model)
    out, dom = chat.find_clf(model, gpus, "output"), chat.find_clf(model, gpus, "domain")
    assert out and dom, "분류기 파일이 없다"
    chat._check_clf(out, gpus, spec.layers, "output"); chat._check_clf(dom, gpus, spec.layers, "domain")

    # 측정 통신량 형태의 값을 넣어 화면 한 장이 그려지는지 본다
    job = {"prompts": ["hello"], "live": {"output_clf": out, "domain_clf": dom, "model": model}}
    live = chat.Live(None, spec.layers, spec.experts, gpus, "gloo", job)
    rng = np.random.default_rng(0)
    sends = rng.multinomial(spec.top_k, [1 / gpus] * gpus, size=spec.layers)      # (layers, gpus), layer마다 합 top-k
    live.start(0)
    live.step(7, sends, (sends * 5, 5), _Rec())
    live.step(8, sends, None, _Rec())
    live.end()
    screen = capsys.readouterr().out
    assert "recovered stream" in screen and "GPU%d" % (gpus - 1) in screen
    assert live.summary[0]["tokens"] == 2 and "domain" in live.summary[0]
    top3 = live.summary[0]["domain_top3"]
    assert len(top3) == 3 and top3[0][1] >= top3[1][1] >= top3[2][1] and top3[1][0] in screen


def test_no_pretrained_for_other_gpu_counts():
    assert chat.pretrained("olmoe", 8, "output") is None


def test_own_classifier_comes_first(tmp_path, monkeypatch):
    """직접 학습해 둔 것(models/)이 있으면 저장소에 들어 있는 것보다 먼저 쓴다."""
    monkeypatch.chdir(tmp_path)
    assert chat.find_clf("olmoe", 4, "output") == chat.pretrained("olmoe", 4, "output")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "olmoe_g4_output.joblib").write_bytes(b"x")
    assert chat.find_clf("olmoe", 4, "output").startswith("models")
