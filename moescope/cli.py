"""moescope — MoE 모델의 EP 서빙과 GPU 간 통신량 측정 CLI.

  moescope collect   프롬프트 파일을 EP로 처리하며 통신량을 저장한다          [멀티 GPU]
  moescope train     저장한 통신량으로 분류기를 학습한다
  moescope chat      모델을 EP로 올려 두고 입력창에서 문장을 받아 응답·통신량·추정 결과를 보여 준다  [멀티 GPU]
  moescope verify    임의 라우팅으로 all_to_all 통신을 해 보고 측정값과 계산값이 같은지 확인한다(설치 확인)
"""
import argparse
import json
import sys
from pathlib import Path

from . import config


def _add(sub, name, help):
    return sub.add_parser(name, help=help, description=help)


def _verify_worker(rank, world, N, E, k, ret):
    import os
    import numpy as np
    import torch
    import torch.distributed as dist
    from .ep import ep_layer
    from .placement import dest_counts_full
    os.environ.update(USE_LIBUV="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29517")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    rng = np.random.default_rng(0)
    routing = np.stack([rng.choice(E, k, replace=False) for _ in range(N)]).astype(np.int64)
    hidden = np.tile(np.arange(N, dtype=np.float32)[:, None], (1, 8)) + 1.0
    b = np.linspace(0, N, world + 1).astype(int); s, e = b[rank], b[rank + 1]
    out, sends = ep_layer(torch.from_numpy(hidden[s:e]), torch.from_numpy(routing[s:e]),
                          None, E, world, lambda _e, x: x, rank)
    g = [torch.zeros_like(sends) for _ in range(world)]
    dist.all_gather(g, sends)
    if rank == 0:
        ret["measured"] = torch.stack(g).sum(0).numpy().tolist()
        ret["computed"] = dest_counts_full(routing, E, world).sum(0).tolist()
    ret[f"rt_{rank}"] = bool(torch.allclose(out, torch.from_numpy(hidden[s:e]) * k))
    dist.barrier(); dist.destroy_process_group()


def cmd_verify(a):
    import torch.multiprocessing as mp
    mgr = mp.Manager(); ret = mgr.dict()
    mp.spawn(_verify_worker, args=(a.gpus, a.tokens, a.experts, a.top_k, ret),
             nprocs=a.gpus, join=True)
    ok = ret["measured"] == ret["computed"] and all(ret[f"rt_{r}"] for r in range(a.gpus))
    print(f"[verify] gpus={a.gpus} tokens={a.tokens} experts={a.experts} k={a.top_k}")
    print(f"  measured : {ret['measured']}")
    print(f"  computed : {ret['computed']}")
    print(f"  measured==computed : {ret['measured']==ret['computed']}")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def read_prompts(path):
    """프롬프트 파일을 읽어 (문장 목록, 라벨 목록 또는 None)을 돌려준다.

    형식: ["문장", ...] 또는 [{"text": "문장", "label": "분야"}, ...] 또는 {"prompts": [...]}.
    라벨은 입력 분야 분류기를 학습할 때 쓴다. 모든 항목에 있을 때만 쓴다.
    """
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(items, dict):
        items = items.get("prompts", list(items.values()))
    texts = [it["text"] if isinstance(it, dict) else it for it in items]
    labels = [it.get("label") if isinstance(it, dict) else None for it in items]
    return texts, (labels if texts and all(x is not None for x in labels) else None)


def cmd_collect(a):
    from . import serve_ep
    prompts, labels = read_prompts(a.prompts)
    print(f"[collect] {a.model} task={a.task} prompts={len(prompts)} gpus={a.gpus}"
          f"{' labels=' + str(len(set(labels))) + ' classes' if labels else ''}")
    report = serve_ep.run(a.model, a.task, prompts, a.gpus, a.out, max_new_tokens=a.max_new_tokens,
                          check=a.check, dtype=a.dtype, backend=a.backend, verbose=not a.quiet, raw=a.raw,
                          labels=labels)
    print("  " + json.dumps(report, ensure_ascii=False))
    print(f"  saved {a.out}")
    ok = report["layer_calls_mismatch"] == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_train(a):
    from . import classifiers
    spec = config.get(a.model)
    out = a.out or f"models/{a.model}_g{a.gpus}_{a.task}.joblib"       # chat이 이 이름으로 찾는다
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if a.task == "output":
        m = classifiers.train_output(a.traces, spec.experts, a.gpus, window=a.window, save=out, source=a.source)
    else:
        m = classifiers.train_domain(a.traces, spec.experts, a.gpus, save=out, source=a.source)
    print(f"[train] {a.model} {a.task} gpus={a.gpus} acc={m['accuracy']:.4f} -> {out}")
    print("  " + json.dumps(m, ensure_ascii=False))


def cmd_chat(a):
    from . import chat
    prompts = list(a.prompt or [])
    if a.prompts:
        prompts += read_prompts(a.prompts)[0]
    report = chat.run(a.model, a.gpus, prompts or None, a.output_clf, a.domain_clf, a.max_new_tokens, a.delay,
                      dtype=a.dtype, backend=a.backend, tiny=a.tiny, out=a.out, use_clf=not a.no_clf)
    sys.exit(0 if report["layer_calls_mismatch"] == 0 else 1)


def main():
    p = argparse.ArgumentParser(prog="moescope", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = _add(sub, "collect", "프롬프트 파일을 EP로 처리하며 통신량을 저장한다 [멀티 GPU]")
    s.add_argument("--model", default="olmoe"); s.add_argument("--task", choices=["prefill", "decode"], default="decode")
    s.add_argument("--prompts", required=True, help='JSON: ["문장", ...] 또는 [{"text": ..., "label": ...}, ...]')
    s.add_argument("--out", required=True)
    s.add_argument("--gpus", type=int, default=4); s.add_argument("--max-new-tokens", type=int, default=64)
    s.add_argument("--check", action="store_true", help="같은 입력을 원래 모델로 실행한 결과와 비교")
    s.add_argument("--raw", action="store_true", help="대화 형식을 씌우지 않고 토크나이저 기본 동작 그대로 넣는다")
    s.add_argument("--dtype", default="bfloat16"); s.add_argument("--backend", choices=["nccl", "gloo"])
    s.add_argument("--quiet", action="store_true", help="진행 로그를 끈다")
    s.set_defaults(func=cmd_collect)

    s = _add(sub, "train", "저장한 통신량으로 분류기를 학습한다")
    s.add_argument("--model", default="olmoe"); s.add_argument("--task", choices=["output", "domain"], default="output")
    s.add_argument("--traces", required=True); s.add_argument("--gpus", type=int, default=4)
    s.add_argument("--window", choices=["none", "causal"], default="causal",
                   help="응답 토큰: causal은 앞의 두 단계 통신량을 함께 쓴다")
    s.add_argument("--out", help="기본: models/<model>_g<gpus>_<task>.joblib (chat이 자동으로 찾는다)")
    s.add_argument("--source", choices=["auto", "measured", "computed"], default="auto",
                   help="measured: collect로 측정한 통신량, computed: 라우팅에서 계산한 통신량")
    s.set_defaults(func=cmd_train)

    s = _add(sub, "chat", "모델을 EP로 올려 두고 입력창에서 문장을 받는다: 응답, GPU별 측정 통신량, 추정 결과 [멀티 GPU]")
    s.add_argument("--model", default="olmoe"); s.add_argument("--gpus", type=int, default=4)
    s.add_argument("--prompt", action="append", help="이 문장만 처리하고 끝낸다(여러 번 줄 수 있다). 없으면 입력창을 띄운다")
    s.add_argument("--prompts", help="문장 목록 JSON 파일(입력창 없이 처리)")
    s.add_argument("--output-clf", help="응답 토큰 분류기(.joblib). 없으면 models/, 저장소에 들어 있는 것 순으로 찾는다")
    s.add_argument("--domain-clf", help="입력 분야 분류기(.joblib). 없으면 models/, 저장소에 들어 있는 것 순으로 찾는다")
    s.add_argument("--no-clf", action="store_true", help="분류기 없이 응답과 통신량만 본다")
    s.add_argument("--max-new-tokens", type=int, default=48); s.add_argument("--delay", type=float, default=0.0)
    s.add_argument("--dtype", default="bfloat16"); s.add_argument("--backend", choices=["nccl", "gloo"])
    s.add_argument("--tiny", action="store_true", help="가중치 없이 작은 임의 모델로 화면과 EP 경로만 확인(CPU)")
    s.add_argument("--out", help="측정 통신량과 생성 토큰을 저장할 .npz")
    s.set_defaults(func=cmd_chat)

    s = _add(sub, "verify", "설치 확인: 임의 라우팅으로 측정값 = 계산값 확인 (CPU)")
    s.add_argument("--gpus", type=int, default=4); s.add_argument("--tokens", type=int, default=300)
    s.add_argument("--experts", type=int, default=64); s.add_argument("--top-k", type=int, default=8)
    s.set_defaults(func=cmd_verify)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
