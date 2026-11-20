"""EP 서빙: MoE 모델의 Expert를 GPU별 프로세스에 나누어 올리고 실제로 추론한다.

프로세스 하나가 GPU 하나를 맡는다(rank r ↔ GPU r). Expert는 번호 순서대로 E/G개씩 나뉜다.
  rank 0   요청을 받는다. 모델의 나머지 부분(임베딩·attention·라우터)과 자기 몫의 Expert를 가진다.
  rank r>0 자기 몫의 Expert만 가진다.
각 MoE layer에서 rank 0은 라우터가 고른 Expert에 따라 토큰을 all_to_all로 보내고(dispatch),
받은 rank가 Expert를 계산해 되돌려 보내며(combine), rank 0이 가중합한다. 이때 목적지 rank별로
실제 보낸 행 수가 측정 통신량이다(ep.ep_layer의 input_split_sizes).

모델 쪽 연결은 transformers의 `*Experts` 모듈(OLMoE·Granite·Qwen1.5-MoE 공통: 3차원 가중치
gate_up_proj / down_proj, forward(hidden, top_k_index, top_k_weights)) 하나만 바꾼다. 라우터와
그 밖의 계산은 원래 모델 코드를 그대로 쓴다.

실행
  moescope collect --model olmoe --task decode --prompts examples/prompts.json --gpus 4 --out traces/olmoe_decode.npz
  moescope collect --model olmoe --task prefill --prompts examples/topic_prompts.json --gpus 4 --out traces/olmoe_prefill.npz
      프롬프트 파일을 EP로 처리하며 통신량을 저장한다. --check 를 주면 같은 입력을 원래 모델로 실행한 결과와 비교한다.
  moescope chat --model olmoe --gpus 4
      모델을 EP로 올려 둔 채 입력창에서 문장을 받아 처리한다(chat.py).
"""
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import config
from .ep import ep_layer
from .placement import dest_counts_full

OP_STOP, OP_LAYER = 0, 1

_LOG = {"on": False, "rank": 0}


def log(msg):
    """진행 상황을 시각·rank와 함께 바로 출력한다(--quiet이면 끈다)."""
    if _LOG["on"]:
        print(f"[{time.strftime('%H:%M:%S')} rank{_LOG['rank']}] {msg}", flush=True)


def _every(i, n, parts=20):
    """n개 가운데 i번째(1부터)를 출력할 차례인지. 길어도 20줄 안팎만 나오게 한다."""
    return i == n or i == 1 or i % max(1, n // parts) == 0


# ---------------------------------------------------------------- 모델 쪽 연결
def find_experts(model):
    """MoE layer 순서대로 Expert 묶음 모듈을 돌려준다."""
    mods = [m for m in model.modules()
            if type(m).__name__.endswith("Experts") and hasattr(m, "gate_up_proj") and hasattr(m, "down_proj")]
    if not mods:
        raise RuntimeError("Expert 모듈(gate_up_proj/down_proj)을 찾지 못했다")
    return mods


def shard(mod, lo, hi, device):
    """이 rank가 맡은 Expert [lo, hi)의 가중치만 남겨 device에 올린다."""
    for name in ("gate_up_proj", "down_proj"):
        full = getattr(mod, name)
        setattr(mod, name, torch.nn.Parameter(full.data[lo:hi].to(device).clone(), requires_grad=False))
        del full
    mod._ep_lo, mod._ep_hi = lo, hi


def expert_fn(mod):
    """전역 Expert 번호와 입력 행을 받아 이 rank에 있는 Expert를 계산한다(원래 모델의 식과 같다)."""
    def fn(e, rows):
        i = e - mod._ep_lo
        assert 0 <= i < mod._ep_hi - mod._ep_lo, f"Expert {e}는 이 rank의 것이 아니다"
        gate, up = F.linear(rows, mod.gate_up_proj[i]).chunk(2, dim=-1)
        return F.linear(mod.act_fn(gate) * up, mod.down_proj[i])
    return fn


class Recorder:
    """rank 0이 forward 한 번 동안 layer별 측정 통신량과 라우팅을 모은다."""

    def __init__(self, layers, experts, gpus):
        self.layers, self.experts, self.gpus = layers, experts, gpus
        self.mismatch = self.calls = 0
        self.reset()

    def reset(self):
        self.sends = [None] * self.layers
        self.routing = [None] * self.layers

    def put(self, layer, sends, routing):
        sends = sends.detach().cpu().numpy().astype(np.int64)
        routing = routing.detach().cpu().numpy()
        computed = dest_counts_full(routing, self.experts, self.gpus).sum(0)
        self.mismatch += int(not np.array_equal(sends, computed)); self.calls += 1
        self.sends[layer], self.routing[layer] = sends, routing

    def take(self):
        """(layers, gpus) 측정 통신량, (tokens, layers, k) 라우팅."""
        sends = np.stack(self.sends); routing = np.stack(self.routing, axis=1)
        self.reset()
        return sends, routing


def attach(model, experts, gpus, rank, device, rec):
    """rank 0: Expert 묶음의 forward를 EP 버전으로 바꾼다."""
    for layer, mod in enumerate(find_experts(model)):
        fn = expert_fn(mod)

        def forward(hidden_states, top_k_index, top_k_weights, layer=layer, fn=fn):
            ctrl = torch.tensor([OP_LAYER, layer], dtype=torch.int64, device=device)
            dist.broadcast(ctrl, src=0)
            out, sends = ep_layer(hidden_states, top_k_index.to(torch.int64), top_k_weights,
                                  experts, gpus, fn, rank)
            rec.put(layer, sends, top_k_index)
            return out.to(hidden_states.dtype)

        mod.forward = forward


def worker_loop(mods, hidden, top_k, dtype, experts, gpus, rank, device):
    """rank r>0: rank 0의 신호에 맞춰 각 MoE layer의 통신에 참여하고 받은 행의 Expert를 계산한다."""
    fns = [expert_fn(m) for m in mods]
    empty_h = torch.empty((0, hidden), dtype=dtype, device=device)
    empty_r = torch.empty((0, top_k), dtype=torch.int64, device=device)
    while True:
        ctrl = torch.zeros(2, dtype=torch.int64, device=device)
        dist.broadcast(ctrl, src=0)
        if int(ctrl[0]) == OP_STOP:
            return
        ep_layer(empty_h, empty_r, None, experts, gpus, fns[int(ctrl[1])], rank)


def stop_workers(device):
    dist.broadcast(torch.tensor([OP_STOP, 0], dtype=torch.int64, device=device), src=0)


# ---------------------------------------------------------------- 모델 올리기
def load_tokenizer(repo, revision):
    """토크나이저를 불러온다. Granite처럼 기본 방식으로 열리지 않는 체크포인트는 GPT2TokenizerFast로 연다."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(repo, revision=revision)
    except ValueError:
        from transformers import GPT2TokenizerFast
        return GPT2TokenizerFast.from_pretrained(repo, revision=revision)


def encode(tok, text, task, raw=False, max_len=32):
    """문장을 토큰 번호로 바꾼다. (ids, span)을 돌려주며 span은 ids 안에서 문장 자체가 놓인 구간이다.

    prefill  문장만 넣는다. 앞에서 max_len개까지 쓴다.
    decode   모델의 대화 형식(chat template)을 씌워 넣는다. 형식이 없는 토크나이저는 문장만 넣는다.
    raw      토크나이저의 기본 동작 그대로 넣는다(대화 형식을 쓰지 않는다).
    """
    if tok is None:                       # 작은 임의 모델(동작 확인용): 글자를 토큰 번호로 쓴다
        ids = [ord(c) % 97 for c in text][:24] or [1]
        return ids, (0, len(ids))
    if task == "prefill":
        kw = {} if raw else {"add_special_tokens": False}
        ids = list(tok(text, truncation=True, max_length=max_len, **kw)["input_ids"])
        return ids, (0, len(ids))
    if raw or not getattr(tok, "chat_template", None):
        ids = list(tok(text)["input_ids"])
        return ids, (0, len(ids))
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = list(ids[0]) if len(ids) and isinstance(ids[0], (list, tuple)) else list(ids)
    body = list(tok(text, add_special_tokens=False)["input_ids"])
    for a in range(len(ids) - len(body) + 1):
        if body and ids[a:a + len(body)] == body:
            return ids, (a, a + len(body))
    return ids, (0, len(ids))


def load(loader):
    """loader: ("hf", repo, revision, dtype) 또는 ("tiny", seed, layers, experts, top_k, hidden[, dtype[, family]])."""
    if loader[0] == "hf":
        from transformers import AutoModelForCausalLM
        _, repo, revision, dtype = loader
        tok = load_tokenizer(repo, revision)
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True,
            experts_implementation="eager").eval()
        return model, tok
    if loader[0] == "tiny":            # 가중치를 내려받지 않는 작은 임의 모델(동작 확인용)
        import transformers as tf
        _, seed, layers, experts, top_k, hidden = loader[:6]
        family = loader[7] if len(loader) > 7 else "olmoe"
        torch.manual_seed(seed)
        common = dict(vocab_size=97, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                      num_experts_per_tok=top_k)
        if family == "olmoe":
            cfg, cls = tf.OlmoeConfig(num_experts=experts, **common), tf.OlmoeForCausalLM
        elif family == "granite":
            cfg, cls = tf.GraniteMoeConfig(num_local_experts=experts, **common), tf.GraniteMoeForCausalLM
        elif family == "qwen":         # 공유 Expert가 따로 있는 구조
            cfg, cls = tf.Qwen2MoeConfig(num_experts=experts, moe_intermediate_size=2 * hidden,
                                         shared_expert_intermediate_size=2 * hidden, **common), tf.Qwen2MoeForCausalLM
        else:
            raise ValueError(family)
        cfg._experts_implementation = "eager"
        model = cls(cfg).eval()
        for m in find_experts(model):   # 임의 초기화가 0에 가까우면 비교가 무의미하므로 값을 키운다
            torch.nn.init.normal_(m.gate_up_proj, std=0.2); torch.nn.init.normal_(m.down_proj, std=0.2)
        if len(loader) > 6:
            model = model.to(getattr(torch, loader[6]))
        return model, None
    raise ValueError(loader)


def _device(rank, backend):
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("cpu")


# ---------------------------------------------------------------- 추론(rank 0)
def _forward(model, ids, past, rec):
    out = model(input_ids=ids, past_key_values=past, use_cache=True)
    sends, routing = rec.take()
    return out, sends, routing


@torch.no_grad()
def run_decode(model, rec, inputs, max_new_tokens, eos=None, watch=None, spans=None):
    """응답 생성(decode): 토큰을 하나씩 만들며 단계별 측정 통신량·라우팅을 모은다.

    프롬프트의 마지막 토큰은 따로 한 번에 넣는다. 그래야 첫 생성 단계의 통신량도 토큰 하나 몫으로
    측정된다(KV cache를 쓰므로 결과는 프롬프트를 한 번에 넣은 것과 같다).
    watch(chat.Live)가 있으면 토큰을 하나 만들 때마다 그 단계의 측정 통신량을 넘겨 화면을 갱신한다.
    spans가 있으면 입력 가운데 문장 자체의 구간(대화 형식의 앞뒤 토큰 제외)을 따로 넣어 그 구간의
    통신량을 잰다. 입력 분야 분류기는 문장만 넣어 잰 통신량으로 학습하기 때문이다.
    """
    device = next(model.parameters()).device
    toks, routes, vols, pids, steps, pre_vol = [], [], [], [], [], []
    t_all = time.time()
    for pid, ids in enumerate(inputs):
        t0, n0 = time.time(), len(toks)
        ids = torch.as_tensor(ids, dtype=torch.long, device=device)[None]
        past, pv = None, np.zeros((rec.layers, rec.gpus), np.int64)
        n, text = ids.shape[1], None
        a, b = spans[pid] if spans else (0, n)
        if watch is not None:
            watch.start(pid)
        if 0 <= a < b <= n - 1 and (a, b) != (0, n - 1):      # 앞부분, 문장, 뒷부분을 차례로 넣는다
            for lo, hi in ((0, a), (a, b), (b, n - 1)):
                if hi > lo:
                    out, s, _ = _forward(model, ids[:, lo:hi], past, rec)
                    past, pv = out.past_key_values, pv + s
                    if (lo, hi) == (a, b):
                        text = (s, b - a)
        elif n > 1:
            out, s, _ = _forward(model, ids[:, :-1], None, rec)
            past, pv = out.past_key_values, pv + s
        cur = ids[:, -1:]
        for step in range(max_new_tokens):
            out, s, r = _forward(model, cur, past, rec)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            toks.append(nxt); routes.append(r[0]); vols.append(s); pids.append(pid); steps.append(step)
            if step == 0:
                pv = pv + s
            if watch is not None:
                watch.step(nxt, s, (text or (pv, n)) if step == 0 else None, rec)
            cur = torch.tensor([[nxt]], device=device)
            if eos is not None and nxt == eos:
                break
        pre_vol.append(pv)
        if watch is not None:
            watch.end()
        log(f"decode {pid + 1}/{len(inputs)}: {ids.shape[1]} input tokens, {len(toks) - n0} generated, "
            f"{time.time() - t0:.1f}s, traffic mismatches so far {rec.mismatch}")
    log(f"decode done: {len(toks)} tokens in {time.time() - t_all:.0f}s")
    return {"tokens": np.array(toks, np.int64), "routing": np.stack(routes).astype(np.int32),
            "measured": np.stack(vols), "prompt_ids": np.array(pids, np.int64),
            "step": np.array(steps, np.int64), "prefill_measured": np.stack(pre_vol)}


def run_chat(model, rec, tok, job, eos, watch):
    """입력창 방식: 모델을 올려 둔 채 문장을 하나씩 받아 처리한다. 문장마다 독립된 요청이다.

    부모 프로세스가 입력을 받아 req로 보내고, 처리가 끝나 다음 입력을 받을 수 있으면 resp로 알린다.
    """
    req, resp = job["chan"]
    parts, inputs = [], []
    while True:
        resp.put("ready")
        text = req.get()
        if text is None:
            break
        ids, span = encode(tok, text, "decode", job.get("raw", False))
        if watch is not None:
            watch.prompts, watch.offset = [text], len(parts)
        d = run_decode(model, rec, [ids], job.get("max_new_tokens", 64), eos, watch, [span])
        d["prompt_ids"] = d["prompt_ids"] + len(parts)
        parts.append(d); inputs.append(ids)
    if not parts:
        z = np.zeros((0, rec.layers, rec.gpus), np.int64)
        return {"tokens": np.zeros(0, np.int64), "routing": np.zeros((0, rec.layers, 1), np.int32), "measured": z,
                "prompt_ids": np.zeros(0, np.int64), "step": np.zeros(0, np.int64), "prefill_measured": z}, inputs
    return {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}, inputs


@torch.no_grad()
def run_prefill(model, rec, inputs):
    """입력 처리(prefill): 프롬프트를 한 번에 넣고 프롬프트 단위 측정 통신량·라우팅을 모은다."""
    device = next(model.parameters()).device
    routes, lengths, vols = [], [], []
    t_all = time.time()
    for i, ids in enumerate(inputs, 1):
        ids = torch.as_tensor(ids, dtype=torch.long, device=device)[None]
        _, s, r = _forward(model, ids, None, rec)
        routes.append(r); lengths.append(len(r)); vols.append(s)
        if _every(i, len(inputs)):
            log(f"prefill {i}/{len(inputs)}: {sum(lengths)} input tokens so far, {time.time() - t_all:.0f}s, "
                f"traffic mismatches so far {rec.mismatch}")
    return {"prefill_routing": np.concatenate(routes).astype(np.int32), "prefill_lengths": np.array(lengths, np.int32),
            "prefill_measured": np.stack(vols), "prefill_ids": np.arange(len(inputs), dtype=np.int32)}


@torch.no_grad()
def run_forced(model, rec, inputs, ref):
    """기준 실행이 만든 토큰을 그대로 넣어 주며 단계마다 EP의 출력과 비교한다.

    토큰을 이어서 생성하면 한 번의 차이가 뒤 전체로 번지므로, 단계별로 같은 입력을 준 상태에서 비교한다.
    """
    device = next(model.parameters()).device
    same = total = 0; diffs = []
    for i, (ids, r) in enumerate(zip(inputs, ref), 1):
        ids = torch.as_tensor(ids, dtype=torch.long, device=device)[None]
        past = None
        if ids.shape[1] > 1:
            out, _, _ = _forward(model, ids[:, :-1], None, rec); past = out.past_key_values
        cur = ids[:, -1:]
        for step, tok_id in enumerate(r["tokens"]):
            out, _, _ = _forward(model, cur, past, rec); past = out.past_key_values
            lg = out.logits[0, -1].float().cpu()
            same += int(int(lg.argmax()) == tok_id); total += 1
            diffs.append(float((lg - r["logits"][step]).abs().max()))
            cur = torch.tensor([[tok_id]], device=device)
        if _every(i, len(inputs)):
            log(f"step-by-step comparison {i}/{len(inputs)}: {same}/{total} steps equal, "
                f"max logit diff {max(diffs):.4g}")
    return {"forced_token_agreement": same / max(total, 1), "forced_steps": total,
            "forced_max_logit_diff": max(diffs), "forced_mean_logit_diff": float(np.mean(diffs))}


@torch.no_grad()
def reference(model, inputs, task, max_new_tokens, eos=None):
    """한 장치에서 원래 모델 그대로 실행한 결과(비교용)."""
    device = model.get_input_embeddings().weight.device
    res = []
    t_all = time.time()
    for i, ids in enumerate(inputs, 1):
        if _every(i, len(inputs)):
            log(f"unmodified model {i}/{len(inputs)}, {time.time() - t_all:.0f}s")
        ids = torch.as_tensor(ids, dtype=torch.long, device=device)[None]
        if task == "prefill":
            res.append(model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu()); continue
        past, cur, gen, lg = None, ids, [], []
        if ids.shape[1] > 1:
            past = model(input_ids=ids[:, :-1], use_cache=True).past_key_values
            cur = ids[:, -1:]
        for _ in range(max_new_tokens):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            lg.append(out.logits[0, -1].float().cpu())
            nxt = int(lg[-1].argmax()); gen.append(nxt)
            cur = torch.tensor([[nxt]], device=device)
            if eos is not None and nxt == eos:
                break
        res.append({"tokens": gen, "logits": torch.stack(lg)})
    return res


def _reference_run(model, loader, backend, device, inputs, job, eos):
    """비교용 기준 실행. GPU가 여러 개면 모델을 GPU들에 나누어 올려 실행한다(한 GPU에 다 안 들어갈 때 대비)."""
    args = (inputs, job["task"], job.get("max_new_tokens", 64), eos)
    if loader[0] == "hf" and backend == "nccl" and torch.cuda.device_count() > 1:
        import gc
        from transformers import AutoModelForCausalLM
        _, repo, revision, dtype = loader
        log(f"loading the unmodified model across {torch.cuda.device_count()} GPUs for comparison")
        whole = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, torch_dtype=getattr(torch, dtype), device_map="auto",
            experts_implementation="eager").eval()   # EP와 같은 계산 방식(Expert마다 따로 계산)으로 비교
        ref = reference(whole, *args)
        del whole; gc.collect()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        return ref
    model.to(device)
    return reference(model, *args)


# ---------------------------------------------------------------- 프로세스 진입점
def run_rank(rank, world, backend, loader, spec, job, ret, port=29541):
    """spec: (layers, experts, top_k).  job: dict(task, inputs|prompts, max_new_tokens, max_len, check)."""
    os.environ.setdefault("USE_LIBUV", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    if not dist.is_initialized():
        dist.init_process_group(backend, rank=rank, world_size=world)
    device = _device(rank, backend)
    layers, experts, top_k = spec
    per = experts // world
    assert per * world == experts, f"Expert {experts}개를 GPU {world}개로 똑같이 나눌 수 없다"
    lo, hi = rank * per, (rank + 1) * per
    _LOG.update(on=bool(job.get("verbose")), rank=rank)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    log(f"started on {device} ({name}), backend {backend}, experts {lo}..{hi - 1} of {experts}")

    model = tok = ref = inputs = spans = None
    for turn in range(world):                     # 한 번에 한 rank씩 올린다(CPU 메모리 최대치를 낮춘다)
        if turn == rank:
            t0 = time.time()
            log("loading the checkpoint on CPU")
            model, tok = load(loader)
            log(f"checkpoint loaded in {time.time() - t0:.0f}s")
            mods = find_experts(model)
            assert len(mods) == layers, f"MoE layer 수 {len(mods)} != {layers}"
            hidden, dtype = mods[0].gate_up_proj.shape[-1], mods[0].gate_up_proj.dtype
            if rank == 0:
                inputs, spans = job.get("inputs"), job.get("spans")
                if inputs is None:
                    enc = [encode(tok, p, job["task"], job.get("raw", False), job.get("max_len", 32))
                           for p in job.get("prompts") or []]
                    inputs = [e[0] for e in enc]
                    spans = [e[1] for e in enc] if job.get("live") else None
                if not job.get("chan"):
                    log(f"task {job['task']}: {len(inputs)} prompts, {sum(len(x) for x in inputs)} input tokens")
                if job.get("check"):
                    ref = _reference_run(model, loader, backend, device, inputs, job,
                                         getattr(tok, "eos_token_id", None))
                for m in mods:
                    shard(m, lo, hi, device)
                model.to(device)
            else:
                for m in mods:
                    shard(m, lo, hi, device)
                del model; model = None
            if backend == "nccl":
                torch.cuda.empty_cache()
                log(f"experts {lo}..{hi - 1} on {device}, {torch.cuda.memory_allocated(device) / 2**30:.1f} GiB allocated")
            else:
                log(f"experts {lo}..{hi - 1} ready")
        dist.barrier()

    if rank > 0:
        log("waiting for layer calls from rank 0")
        worker_loop(mods, hidden, top_k, dtype, experts, world, rank, device)
        if not job.get("live"):
            log("done")
        dist.barrier(); dist.destroy_process_group()
        return

    rec = Recorder(layers, experts, world)
    attach(model, experts, world, rank, device, rec)
    eos = getattr(tok, "eos_token_id", None)
    log(f"all {world} processes ready, serving in EP form")
    watch = None
    if job.get("live") and job["task"] == "decode":   # 실시간 화면: 토큰마다 측정 통신량을 그린다
        from .chat import Live
        watch = Live(tok, layers, experts, world, backend, job)
        _LOG["on"] = False                            # 화면을 지우며 그리므로 진행 로그는 끈다
    if job.get("chan"):
        data, inputs = run_chat(model, rec, tok, job, eos, watch)
    else:
        data = (run_prefill(model, rec, inputs) if job["task"] == "prefill"
                else run_decode(model, rec, inputs, job.get("max_new_tokens", 64), eos, watch, spans))
    report = {"gpus": world, "backend": backend, "layer_calls_mismatch": rec.mismatch, "layer_calls": rec.calls}
    if watch is not None:
        report["live"] = watch.summary
    if ref is not None:
        log("comparing with the unmodified model")
        if job["task"] == "prefill":                   # 마지막 위치의 logits를 다시 계산해 비교
            diffs = []
            for ids, r in zip(inputs, ref):
                ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)[None]
                with torch.no_grad():
                    out = model(input_ids=ids_t, use_cache=False)
                rec.reset()
                diffs.append(float((out.logits[0, -1].float().cpu() - r).abs().max()))
            report["max_logit_diff"] = max(diffs)
        else:
            got, n = {}, 0
            for t, p in zip(data["tokens"].tolist(), data["prompt_ids"].tolist()):
                got.setdefault(p, []).append(t)
            same = sum(int(a == b) for p, r in enumerate(ref) for a, b in zip(got.get(p, []), r["tokens"]))
            n = sum(len(r["tokens"]) for r in ref)
            report["token_agreement"] = same / max(n, 1); report["tokens_compared"] = n
            report.update(run_forced(model, rec, inputs, ref))
    stop_workers(device)
    dist.barrier(); dist.destroy_process_group()
    if tok is not None and job["task"] == "decode":
        texts, start = [], 0
        for p in range(len(inputs)):
            n = int((data["prompt_ids"] == p).sum())
            texts.append(tok.decode(data["tokens"][start:start + n].tolist())); start += n
        data["generated"] = np.array("\u0001".join(texts))
    ret["data"] = {k: v for k, v in data.items()}
    ret["report"] = report


def launch(loader, spec, job, gpus, backend=None, port=29541, interact=None):
    """프로세스 gpus개를 띄워 EP 서빙을 실행하고 (자료, 보고)를 돌려준다.

    interact(req, resp, procs)를 주면 프로세스들이 떠 있는 동안 부모에서 그 함수를 실행한다(입력창 방식).
    """
    import torch.multiprocessing as mp
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else "gloo"
    experts = spec[1]
    if gpus < 1 or experts % gpus != 0:
        ok = [g for g in range(1, experts + 1) if experts % g == 0]
        raise ValueError(f"Expert {experts}개는 GPU {gpus}개로 똑같이 나눌 수 없다. 가능한 GPU 수: {ok}")
    if backend == "nccl" and gpus > torch.cuda.device_count():
        raise ValueError(f"GPU {gpus}개를 요청했지만 이 장비에는 {torch.cuda.device_count()}개만 있다")
    mgr = mp.Manager(); ret = mgr.dict()
    if interact is None:
        mp.spawn(run_rank, args=(gpus, backend, loader, spec, job, ret, port), nprocs=gpus, join=True)
    else:
        req, resp = mgr.Queue(), mgr.Queue()
        job = dict(job, chan=(req, resp))
        procs = mp.spawn(run_rank, args=(gpus, backend, loader, spec, job, ret, port), nprocs=gpus, join=False)
        try:
            interact(req, resp, procs)
        finally:
            req.put(None)                              # 어떤 경우에도 서빙 프로세스를 끝낸다
            while not procs.join():
                pass
    return dict(ret["data"]), dict(ret["report"])


def run(model_key, task, prompts, gpus, out, max_new_tokens=64, check=False, dtype="bfloat16", backend=None,
        verbose=True, raw=False, labels=None):
    """프롬프트 목록을 EP로 처리해 통신량을 out(.npz)에 저장한다. labels가 있으면 함께 저장한다(입력 분야 학습용)."""
    spec = config.get(model_key)
    loader = ("hf", spec.repo, spec.revision, dtype)
    job = {"task": task, "prompts": prompts, "max_new_tokens": max_new_tokens, "check": check, "verbose": verbose,
           "raw": raw}
    data, report = launch(loader, (spec.layers, spec.experts, spec.top_k), job, gpus, backend)
    if labels is not None:
        data["labels"] = np.array(labels)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **data)
    return report
