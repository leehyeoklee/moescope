"""실시간 서빙 + 통신량 복원 CLI.

왼쪽: 모델이 실제로 답 생성(토큰 스트리밍).
오른쪽: 각 스텝 라우팅 → 통신량(측정=계산, measure-ep로 검증) → 학습된 분류기로 복원.

  moescope demo --model olmoe --output-clf models/olmoe_output_causal.joblib
  moescope demo --model olmoe --mock-model     # 모델 없이 UI/파이프라인 확인
"""
import os
import sys
import time

import numpy as np

from . import config
from .traffic import volume_features
from .placement import dest_counts_full

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


def render(prompt, gen_tokens, recon, domain, trow, gpus):
    L, Rr = [], []
    L.append(f"{BOLD}{GREEN} model ❯ generate{R}")
    L.append(f"{GREY}{'─'*PANEL}{R}")
    L.append(f"{CYAN}user>{R} {prompt[:PANEL-7]}")
    L.append("")
    body = f"{MAG}gen>{R} " + " ".join(WHITE + t + R for t in gen_tokens) + f"{GREY}▌{R}"
    cur = ""
    for w in body.split(" "):
        if len(_strip(cur)) + len(_strip(w)) + 1 > PANEL:
            L.append(cur); cur = "  "
        cur += w + " "
    L.append(cur)
    while len(L) < 14:
        L.append("")

    Rr.append(f"{BOLD}{YEL} ep-traffic-mon{R}{DIM}  measured · ep={gpus}{R}")
    Rr.append(f"{GREY}{'─'*PANEL}{R}")
    Rr.append(f"{DIM}domain≈{R} " + (f"{YEL}{BOLD}{domain[0]}{R} {DIM}p={domain[1]:.2f}{R}"
                                      if domain else f"{GREY}inferring…{R}"))
    Rr.append("")
    Rr.append(f"{DIM}recovered stream:{R}")
    line = "  "
    for tok, ok in recon:
        piece = (GREEN if ok else RED) + tok + R
        if len(_strip(line)) + len(tok) + 1 > PANEL:
            Rr.append(line); line = "  "
        line += piece + " "
    Rr.append(line if recon else f"{GREY}  …{R}")
    Rr.append("")
    if recon:
        c = sum(1 for _, ok in recon if ok); acc = c / len(recon) * 100
        col = GREEN if acc >= 50 else (YEL if acc >= 35 else RED)
        Rr.append(f"{DIM}accuracy{R} {col}{c}/{len(recon)} ({acc:.0f}%){R}")
    if trow is not None:
        Rr.append("")
        Rr.append(f"{DIM}this step traffic (per dest GPU){R}")
        mx = max(1, trow.max()); cols = [CYAN, MAG, YEL, GREEN, "\033[94m"]
        for g, v in enumerate(trow):
            n = int(round(v / mx * 14))
            Rr.append(f"  GPU{g+1} {cols[g%5]}{'█'*n}{GREY}{'░'*(14-n)}{R} {int(v)}")
    while len(Rr) < 14:
        Rr.append("")
    buf = [CLEAR, f"{GREY}  {'═'*(PANEL*2+3)}{R}"]
    for la, rb in zip(L, Rr):
        buf.append("  " + _pad(la) + f" {GREY}│{R} " + _pad(rb))
    buf.append(f"{GREY}  {'═'*(PANEL*2+3)}{R}")
    sys.stdout.write("\n".join(buf) + "\n"); sys.stdout.flush()


def _mock_stream(spec, steps, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(steps):
        routing = np.stack([rng.choice(spec.experts, spec.top_k, replace=False)
                            for _ in range(spec.layers)])
        yield f"t{i}", int(rng.integers(0, 5000)), routing


def _real_stream(model, tok, prompt, spec, max_new):
    import torch
    from .collect import routing_from_logits
    ids = tok(prompt, return_tensors="pt").to(model.device)
    cur, past = ids.input_ids, None
    with torch.no_grad():
        for _ in range(max_new):
            out = model(input_ids=cur if past is None else cur[:, -1:],
                        past_key_values=past, use_cache=True, output_router_logits=True)
            past = out.past_key_values
            nxt = int(out.logits[:, -1, :].argmax(-1))
            rl = [lg[-1:] for lg in out.router_logits]
            routing = routing_from_logits(rl, spec.top_k, spec.experts)[0]
            cur = torch.cat([cur, torch.tensor([[nxt]], device=model.device)], 1)
            yield tok.decode([nxt]), nxt, routing
            if nxt == tok.eos_token_id:
                break


def _causal(vols, W=2):
    cur = vols[-1]; parts = []
    for off in range(-W, 1):
        j = len(vols) - 1 + off
        parts.append(vols[j] if j >= 0 else np.zeros_like(cur))
    return np.concatenate(parts)[None, :]


def run(model="olmoe", gpus=4, output_clf=None, domain_clf=None, mock_model=False,
        steps=17, max_new_tokens=48, delay=0.25):
    import joblib
    spec = config.get(model)
    out_clf = joblib.load(output_clf)
    dom_clf = joblib.load(domain_clf) if domain_clf and os.path.exists(domain_clf) else None

    if mock_model:
        prompt = "How do I treat a mild fever at home?"
        stream = _mock_stream(spec, steps)
        decode = lambda i: f"w{i%97}"
    else:
        from .collect import load_model
        m, tok = load_model(spec)
        prompt = input("  user> ").strip() or "How do I treat a mild fever at home?"
        stream = _real_stream(m, tok, prompt, spec, max_new_tokens)
        decode = lambda i: tok.decode([i])

    gen, recon, vols, domain = [], [], [], None
    mo = out_clf["model"]
    for true_text, true_id, routing in stream:
        gen.append((true_text.strip() or true_text))
        vols.append(volume_features(routing[None], spec.experts, gpus)[0])
        pred_id = int(mo.predict(_causal(vols))[0])
        recon.append((decode(pred_id), pred_id == true_id))
        if dom_clf is not None and domain is None and len(vols) >= 3:
            dx = np.mean(vols, axis=0)[None, :]
            try:
                p = dom_clf["model"].predict_proba(dx)[0]
                domain = (str(dom_clf["model"].classes_[p.argmax()]), float(p.max()))
            except Exception:
                domain = None
        trow = dest_counts_full(routing, spec.experts, gpus).sum(0)
        render(prompt, gen, recon, domain, trow, gpus)
        time.sleep(delay)
    print(f"\n  {DIM}traffic computed from routing (= measured, verified by measure-ep). single request.{R}\n")
