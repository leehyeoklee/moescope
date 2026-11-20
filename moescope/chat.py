"""EP 서빙 입력창과 실시간 화면.

모델을 GPU별 프로세스에 나누어 EP로 올려 둔 채 입력창에서 문장을 받는다. 문장을 처리하는 동안 토큰을
하나 생성할 때마다 화면을 갱신한다.
  한쪽   rank 0이 생성하는 응답
  다른쪽 그 토큰을 처리하며 각 GPU로 실제 보낸 행 수(측정 통신량)와, 통신량으로 추정한 입력 분야와 응답 토큰

  moescope chat --model olmoe --gpus 4                 입력창을 띄운다. exit 또는 빈 줄로 끝낸다.
  moescope chat --model olmoe --gpus 4 --prompt "..."  그 문장만 처리하고 끝낸다.
  moescope chat --model olmoe --gpus 4 --no-clf        분류기 없이 응답과 통신량만 본다.
  moescope chat --tiny --gpus 4                        가중치 없이 작은 임의 모델로 화면과 EP 경로만 확인(CPU)

분류기는 --output-clf/--domain-clf로 준 파일, models/ 폴더에 직접 학습해 둔 것(moescope train),
저장소에 들어 있는 것(moescope/pretrained/) 순서로 찾는다.
"""
import os
import sys
import time

import numpy as np

from . import config
from .traffic import measured_features, measured_prompt_features

R = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"; YEL = "\033[93m"
GREY = "\033[90m"; MAG = "\033[95m"; WHITE = "\033[97m"; CLEAR = "\033[2J\033[H"
PANEL = 46
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)


def _strip(s):
    out, i = "", 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
        else:
            out += s[i]; i += 1
    return out


def _pad(s, w=PANEL):
    return s + " " * max(0, w - len(_strip(s)))


def _wrap(lines, pieces, first="  "):
    """색이 입혀진 조각들을 PANEL 폭에 맞춰 줄로 나눈다."""
    cur = first
    for w in pieces:
        if len(_strip(cur)) + len(_strip(w)) + 1 > PANEL:
            lines.append(cur); cur = "  "
        cur += w + " "
    lines.append(cur)


def render(prompt, gen_tokens, recon, domain, trow, gpus, info=None):
    """화면 한 장을 그린다.

    recon: [(복원 토큰, 맞았는지)] 또는 None(응답 토큰 분류기 없음).  domain: [(분야, 확률), ...] 확률이 높은 순서, 또는 None.
    trow: 이 토큰을 처리하며 GPU별로 보낸 행 수.  info: EP 서빙 상태(title, per, calls, mismatch, layers, clf).
    """
    info = info or {}
    L, Rr = [], []
    L.append(f"{BOLD}{GREEN} {info.get('title', 'model ❯ generate')}{R}")
    L.append(f"{GREY}{'─'*PANEL}{R}")
    L.append(f"{CYAN}user>{R} {prompt[:PANEL-7]}")
    L.append("")
    _wrap(L, [WHITE + t + R for t in gen_tokens] + [f"{GREY}▌{R}"], first=f"{MAG}gen>{R} ")

    Rr.append(f"{BOLD}{YEL} ep-traffic-mon{R}{DIM}  measured · {gpus} GPUs{R}")
    Rr.append(f"{GREY}{'─'*PANEL}{R}")
    if domain:
        if not isinstance(domain[0], (tuple, list)):
            domain = [domain]
        Rr.append(f"{DIM}domain≈{R} {YEL}{BOLD}{domain[0][0]}{R} {DIM}{domain[0][1]:.2f}{R}")
        for name, prob in domain[1:]:
            Rr.append(f"        {name} {DIM}{prob:.2f}{R}")
    else:
        Rr.append(f"{DIM}domain≈{R} {GREY}{'inferring…' if info.get('clf', (1, 1))[1] else 'no classifier'}{R}")
    Rr.append("")
    if recon is not None:
        Rr.append(f"{DIM}recovered stream:{R}")
        if recon:
            _wrap(Rr, [(GREEN if ok else RED) + tok + R for tok, ok in recon])
            c = sum(1 for _, ok in recon if ok); acc = c / len(recon) * 100
            col = GREEN if acc >= 50 else (YEL if acc >= 35 else RED)
            Rr.append("")
            Rr.append(f"{DIM}accuracy{R} {col}{c}/{len(recon)} ({acc:.0f}%){R}")
        else:
            Rr.append(f"{GREY}  …{R}")
        Rr.append("")
    if trow is not None:
        layers = f", {info['layers']} layers" if "layers" in info else ""
        Rr.append(f"{DIM}rows sent to each GPU, this token{layers}{R}")
        mx = max(1, max(trow)); cols = [CYAN, MAG, YEL, GREEN, "\033[94m"]
        per = info.get("per")
        for g, v in enumerate(trow):
            n = int(round(v / mx * 14))
            own = f"{DIM}  E{g*per}-{(g+1)*per-1}{R}" if per else ""
            Rr.append(f"  GPU{g} {cols[g%5]}{'█'*n}{GREY}{'░'*(14-n)}{R} {int(v):>3}{own}")
    if "calls" in info:
        ok = info["mismatch"] == 0
        Rr.append("")
        Rr.append(f"{DIM}measured = computed{R} {(GREEN if ok else RED)}{info['calls'] - info['mismatch']}/{info['calls']}"
                  f"{R}{DIM} layer calls{R}")
    while len(Rr) < max(14, len(L)):
        Rr.append("")
    while len(L) < len(Rr):
        L.append("")
    buf = [CLEAR, f"{GREY}  {'═'*(PANEL*2+3)}{R}"]
    for la, rb in zip(L, Rr):
        buf.append("  " + _pad(la) + f" {GREY}│{R} " + _pad(rb))
    buf.append(f"{GREY}  {'═'*(PANEL*2+3)}{R}")
    sys.stdout.write("\n".join(buf) + "\n"); sys.stdout.flush()


def _causal(feats, W=2):
    cur = feats[-1]; parts = []
    for off in range(-W, 1):
        j = len(feats) - 1 + off
        parts.append(feats[j] if j >= 0 else np.zeros_like(cur))
    return np.concatenate(parts)[None, :]


class Live:
    """EP 서빙의 rank 0이 토큰을 하나 만들 때마다 부른다. 측정한 통신량으로 화면을 갱신한다."""

    def __init__(self, tok, layers, experts, gpus, backend, job):
        opt = job["live"]
        self.tok, self.layers, self.experts, self.gpus = tok, layers, experts, gpus
        self.prompts = job.get("prompts") or [" ".join(map(str, x)) for x in job.get("inputs") or []]
        self.offset = 0
        self.delay = opt.get("delay", 0.0)
        self.title = f"EP serving ❯ {opt.get('model', 'model')} · {gpus} proc · {backend}"[:PANEL - 2]
        self.out = self.dom = None
        if opt.get("output_clf") or opt.get("domain_clf"):
            import joblib
            self.out = joblib.load(opt["output_clf"]) if opt.get("output_clf") else None
            self.dom = joblib.load(opt["domain_clf"]) if opt.get("domain_clf") else None
        self.summary = []

    def _text(self, tok_id):
        if self.tok is None:
            return f"t{tok_id}"
        return self.tok.decode([tok_id], skip_special_tokens=True).strip() or "·"

    def start(self, pid):
        self.pid = pid
        self.gen, self.feats, self.domain = [], [], None
        self.recon = [] if self.out is not None else None

    def step(self, tok_id, sends, prefill, rec):
        """sends (layers, gpus): 이 토큰의 측정 통신량.
        prefill: (입력 문장의 측정 통신량, 그 토큰 수). 첫 단계에만 온다."""
        self.gen.append(self._text(tok_id))
        if prefill is not None and self.dom is not None:
            x = measured_prompt_features(prefill[0][None], [prefill[1]])
            p = self.dom["model"].predict_proba(x)[0]
            top = np.argsort(p)[::-1][:3]                      # 확률이 높은 분야 3개
            self.domain = [(str(self.dom["model"].classes_[i]), float(p[i])) for i in top]
        if self.out is not None:
            self.feats.append(measured_features(sends[None])[0])
            causal = self.out.get("meta", {}).get("window") == "causal"
            pred = int(self.out["model"].predict(_causal(self.feats) if causal else self.feats[-1][None])[0])
            self.recon.append((self._text(pred), pred == tok_id))
        info = {"title": self.title, "per": self.experts // self.gpus, "layers": self.layers,
                "calls": rec.calls, "mismatch": rec.mismatch, "clf": (self.out is not None, self.dom is not None)}
        render(self.prompts[self.pid], self.gen, self.recon, self.domain, sends.sum(0), self.gpus, info)
        if self.delay:
            time.sleep(self.delay)

    def end(self):
        row = {"prompt": self.pid + self.offset, "tokens": len(self.gen)}
        if self.recon:
            row["recovered"] = sum(ok for _, ok in self.recon)
        if self.domain:
            row["domain"] = self.domain[0][0]
            row["domain_top3"] = [(n, round(pr, 3)) for n, pr in self.domain]
        self.summary.append(row)
        if self.delay:
            time.sleep(self.delay * 6)


def pretrained(model, gpus, task):
    """저장소에 들어 있는 분류기의 경로. 그 모델·GPU 수의 것이 없으면 None."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained", f"{model}_g{gpus}_{task}.joblib")
    return p if os.path.exists(p) else None


def _check_clf(path, gpus, layers, kind):
    """분류기가 이 GPU 수의 통신량으로 학습된 것인지 서빙 전에 확인한다."""
    import joblib
    obj = joblib.load(path)
    want = layers * (gpus - 1) * (3 if obj.get("meta", {}).get("window") == "causal" else 1)
    got = obj["model"].n_features_in_
    if got != want:
        raise ValueError(f"{kind} 분류기 {path}의 입력 크기 {got}이 GPU {gpus}개·layer {layers}개의 통신량({want})과 다르다. "
                         f"같은 --gpus로 학습한 분류기를 써야 한다.")


def find_clf(model, gpus, task):
    """직접 학습해 둔 분류기(models/)를 먼저, 없으면 저장소에 들어 있는 것을 찾는다."""
    own = os.path.join("models", f"{model}_g{gpus}_{task}.joblib")
    return own if os.path.exists(own) else pretrained(model, gpus, task)


def _wait(resp, procs):
    """서빙 프로세스가 다음 입력을 받을 준비가 될 때까지 기다린다. 프로세스가 죽으면 그 오류를 올린다."""
    import queue
    while True:
        try:
            return resp.get(timeout=0.5)
        except queue.Empty:
            if procs.join(timeout=0.01):
                raise RuntimeError("서빙 프로세스가 끝났다")


def _prompt_loop(read):
    """입력창: 준비되면 한 줄을 받아 서빙 프로세스로 보낸다. 빈 줄, exit, quit, Ctrl+D면 끝낸다."""
    def loop(req, resp, procs):
        first = True
        while True:
            _wait(resp, procs)
            if first:
                print(f"\n  {DIM}type a prompt and press Enter. empty line or 'exit' to stop.{R}", flush=True)
                first = False
            try:
                text = read()
            except (EOFError, KeyboardInterrupt):
                text = None
            if text is None or text.strip().lower() in ("", "exit", "quit", "/bye"):
                req.put(None)
                return
            req.put(text.strip())
    return loop


def run(model="olmoe", gpus=4, prompts=None, output_clf=None, domain_clf=None, max_new_tokens=48, delay=0.0,
        dtype="bfloat16", backend=None, tiny=False, out=None, port=29561, use_clf=True, read=None):
    """prompts가 없으면 입력창을 띄우고, 있으면 그 문장들만 처리하고 끝낸다.

    use_clf=False면 분류기 없이 통신량만 본다. read는 입력 한 줄을 돌려주는 함수(기본: 터미널 입력).
    """
    from . import serve_ep
    name = "tiny" if tiny else model
    if use_clf:
        output_clf = output_clf or find_clf(name, gpus, "output")
        domain_clf = domain_clf or find_clf(name, gpus, "domain")
    else:
        output_clf = domain_clf = None
    job = {"task": "decode", "max_new_tokens": max_new_tokens, "verbose": True,
           "live": {"output_clf": output_clf, "domain_clf": domain_clf, "delay": delay, "model": name}}
    if tiny:                                   # 가중치 없이 화면과 EP 경로만 확인
        spec3, loader = (4, 16, 4), ("tiny", 0, 4, 16, 4, 32)
        backend = backend or "gloo"
    else:
        spec = config.get(model)
        spec3, loader = (spec.layers, spec.experts, spec.top_k), ("hf", spec.repo, spec.revision, dtype)
    for path, kind in ((output_clf, "응답 토큰"), (domain_clf, "입력 분야")):
        if path:
            _check_clf(path, gpus, spec3[0], kind)
    for path, kind in ((output_clf, "output-token"), (domain_clf, "input-topic")):
        print(f"  {kind} classifier: {path or 'none (traffic only)'}", flush=True)
    interact = None
    if prompts:
        job["prompts"] = list(prompts)
    else:
        interact = _prompt_loop(read or (lambda: input(f"\n  {CYAN}user>{R} ")))
    data, report = serve_ep.launch(loader, spec3, job, gpus, backend, port, interact=interact)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.savez_compressed(out, **data)
    print(f"\n  {DIM}EP serving on {report['gpus']} processes ({report['backend']}), {len(data['tokens'])} tokens generated, "
          f"layer calls with measured != computed: {report['layer_calls_mismatch']}{R}")
    for row in report.get("live", []):
        print(f"  {DIM}{row}{R}")
    print()
    return report
