"""moescope — MoE 모델의 EP 서빙과 GPU 간 통신량 측정 CLI.

  moescope verify-ep   측정 통신량 = 라우팅 계산 검증 (gloo CPU / NCCL GPU)
  moescope collect     실제 모델에서 라우팅 추출 → 트레이스(.npz)   [GPU]
  moescope train       통신량 분류기 학습·저장
  moescope measure-ep  실제 EP로 통신량 실측·계산 일치 확인          [멀티 GPU]
  moescope serve-ep    모델을 EP로 서빙하며 통신량 측정              [멀티 GPU]
  moescope demo        실시간 서빙 + 통신량 복원                     [GPU]
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


def cmd_verify_ep(a):
    import torch.multiprocessing as mp
    mgr = mp.Manager(); ret = mgr.dict()
    mp.spawn(_verify_worker, args=(a.gpus, a.tokens, a.experts, a.top_k, ret),
             nprocs=a.gpus, join=True)
    ok = ret["measured"] == ret["computed"] and all(ret[f"rt_{r}"] for r in range(a.gpus))
    print(f"[verify-ep] gpus={a.gpus} tokens={a.tokens} experts={a.experts} k={a.top_k}")
    print(f"  measured : {ret['measured']}")
    print(f"  computed : {ret['computed']}")
    print(f"  measured==computed : {ret['measured']==ret['computed']}")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_collect(a):
    import numpy as np
    from . import collect
    spec = config.get(a.model)
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    if isinstance(prompts, dict):
        prompts = prompts.get("prompts", list(prompts.values()))
    print(f"[collect] {a.model} task={a.task} prompts={len(prompts)}")
    model, tok = collect.load_model(spec)
    data = (collect.collect_e2(model, tok, prompts, spec, max_new_tokens=a.max_new_tokens)
            if a.task == "e2" else collect.collect_e1(model, tok, prompts, spec))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **data)
    print(f"  saved {a.out}")


def cmd_train(a):
    from . import classifiers
    spec = config.get(a.model)
    out = a.out or f"models/{a.model}_{a.task}{'_causal' if getattr(a,'window','none')=='causal' else ''}.joblib"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if a.task == "output":
        m = classifiers.train_output(a.traces, spec.experts, a.gpus, window=a.window, save=out, source=a.source)
    else:
        m = classifiers.train_domain(a.traces, spec.experts, a.gpus, save=out, source=a.source)
    print(f"[train] {a.model} {a.task} acc={m['accuracy']:.4f} -> {out}")
    print("  " + json.dumps(m, ensure_ascii=False))


def cmd_measure_ep(a):
    from . import measure_ep
    sys.exit(0 if measure_ep.run(a.model, a.routing, a.max_steps, a.gpus) else 1)


def cmd_serve_ep(a):
    from . import serve_ep
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    if isinstance(prompts, dict):
        prompts = prompts.get("prompts", list(prompts.values()))
    print(f"[serve-ep] {a.model} task={a.task} prompts={len(prompts)} gpus={a.gpus}")
    report = serve_ep.run(a.model, a.task, prompts, a.gpus, a.out, max_new_tokens=a.max_new_tokens,
                          check=a.check, dtype=a.dtype, backend=a.backend)
    print("  " + json.dumps(report, ensure_ascii=False))
    print(f"  saved {a.out}")
    ok = report["layer_calls_mismatch"] == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_demo(a):
    from . import demo
    demo.run(a.model, a.gpus, a.output_clf, a.domain_clf, a.mock_model,
             a.steps, a.max_new_tokens, a.delay)


def main():
    p = argparse.ArgumentParser(prog="moescope", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = _add(sub, "verify-ep", "측정=계산 검증 (gloo)")
    s.add_argument("--gpus", type=int, default=4); s.add_argument("--tokens", type=int, default=300)
    s.add_argument("--experts", type=int, default=64); s.add_argument("--top-k", type=int, default=8)
    s.set_defaults(func=cmd_verify_ep)

    s = _add(sub, "collect", "실제 모델 라우팅 추출 [GPU]")
    s.add_argument("--model", default="olmoe"); s.add_argument("--task", choices=["e1", "e2"], default="e2")
    s.add_argument("--prompts", required=True); s.add_argument("--out", required=True)
    s.add_argument("--max-new-tokens", type=int, default=64); s.set_defaults(func=cmd_collect)

    s = _add(sub, "train", "통신량 분류기 학습")
    s.add_argument("--model", default="olmoe"); s.add_argument("--task", choices=["output", "domain"], default="output")
    s.add_argument("--traces", required=True); s.add_argument("--gpus", type=int, default=4)
    s.add_argument("--window", choices=["none", "causal"], default="none"); s.add_argument("--out")
    s.add_argument("--source", choices=["computed", "measured"], default="computed",
                   help="measured: serve-ep로 측정한 통신량으로 학습")
    s.set_defaults(func=cmd_train)

    s = _add(sub, "serve-ep", "모델을 EP로 서빙하며 통신량 측정 [멀티 GPU]")
    s.add_argument("--model", default="olmoe"); s.add_argument("--task", choices=["e1", "e2"], default="e2")
    s.add_argument("--prompts", required=True); s.add_argument("--out", required=True)
    s.add_argument("--gpus", type=int, default=4); s.add_argument("--max-new-tokens", type=int, default=64)
    s.add_argument("--check", action="store_true", help="한 GPU에서 실행한 결과와 비교")
    s.add_argument("--dtype", default="bfloat16"); s.add_argument("--backend", choices=["nccl", "gloo"])
    s.set_defaults(func=cmd_serve_ep)

    s = _add(sub, "measure-ep", "실제 EP 측정=계산 [멀티 GPU]")
    s.add_argument("--model", default="olmoe"); s.add_argument("--routing", required=True)
    s.add_argument("--max-steps", type=int, default=500); s.add_argument("--gpus", type=int, default=4)
    s.set_defaults(func=cmd_measure_ep)

    s = _add(sub, "demo", "실시간 서빙 + 복원 [GPU]")
    s.add_argument("--model", default="olmoe"); s.add_argument("--gpus", type=int, default=4)
    s.add_argument("--output-clf"); s.add_argument("--domain-clf")
    s.add_argument("--mock-model", action="store_true"); s.add_argument("--steps", type=int, default=17)
    s.add_argument("--max-new-tokens", type=int, default=48); s.add_argument("--delay", type=float, default=0.25)
    s.set_defaults(func=cmd_demo)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
