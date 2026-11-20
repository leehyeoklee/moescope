"""실제 NCCL(또는 gloo)에서 '측정 통신량 = 라우팅 계산' 확인.

검증된 ep.ep_layer에 실제 라우팅(트레이스)을 넣어, 실제 all_to_all 송신 수를 측정하고
계산값과 비교한다 → "실제 EP 배포에서도 통신량 = 계산값" 입증.

실행:
  moescope measure-ep --model olmoe --routing TRACE.npz --gpus 4      # 로컬 self-spawn
  torchrun --nproc_per_node=4 -m moescope.measure_ep --model olmoe --routing TRACE.npz  # 멀티 GPU
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

from . import config
from .ep import ep_layer
from .placement import dest_counts_full

H = 8


def run_rank(rank, world, routing, spec, backend, ret):
    os.environ.setdefault("USE_LIBUV", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    if not dist.is_initialized():
        dist.init_process_group(backend, rank=rank, world_size=world)
    use_cuda = backend == "nccl" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if use_cuda else "cpu")

    N, layers, k = routing.shape
    b = np.linspace(0, N, world + 1).astype(int)
    s, e = b[rank], b[rank + 1]
    mism, rt = 0, 0
    for L in range(layers):
        lr = torch.from_numpy(routing[s:e, L, :]).to(device)
        lh = torch.arange(s, e, device=device, dtype=torch.float32)[:, None].repeat(1, H) + 1.0
        out, sends = ep_layer(lh, lr, None, spec.experts, world, lambda _e, x: x, rank)
        gathered = [torch.zeros_like(sends) for _ in range(world)]
        dist.all_gather(gathered, sends)
        if rank == 0:
            measured = torch.stack(gathered).sum(0).cpu().numpy()
            computed = dest_counts_full(routing[:, L, :], spec.experts, world).sum(0)
            mism += int(not np.array_equal(measured, computed))
        rt += int(not torch.allclose(out, lh * k))
    if rank == 0:
        ret["mismatches"], ret["layers"] = mism, layers
    ret[f"rt_{rank}"] = rt
    dist.barrier()
    dist.destroy_process_group()


def _load(path, spec, max_steps):
    d = np.load(path)
    r = d["routing"] if "routing" in d.files else d["e2_routing"]
    r = r[:max_steps].astype(np.int64)
    assert r.shape[1] == spec.layers and r.shape[2] == spec.top_k, f"{r.shape} vs {spec}"
    return r


def _report(model, backend, world, routing, ret):
    mism = ret.get("mismatches", -1); layers = ret.get("layers", routing.shape[1])
    rt = sum(ret.get(f"rt_{r}", 0) for r in range(world))
    ok = mism == 0 and rt == 0
    print(f"[measure-ep] model={model} backend={backend} world={world} "
          f"steps={routing.shape[0]} layers={layers}")
    print(f"  measured==computed mismatch: {mism}/{layers}")
    print(f"  roundtrip fail: {rt}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def run(model="olmoe", routing_path=None, max_steps=500, gpus=4, backend=None):
    spec = config.get(model)
    routing = _load(routing_path, spec, max_steps)
    if "RANK" in os.environ:  # torchrun
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        bk = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        ret = {}
        run_rank(rank, world, routing, spec, bk, ret)
        return _report(model, bk, world, routing, ret) if rank == 0 else True
    import torch.multiprocessing as mp
    bk = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    mgr = mp.Manager(); ret = mgr.dict()
    mp.spawn(run_rank, args=(gpus, routing, spec, bk, ret), nprocs=gpus, join=True)
    return _report(model, bk, gpus, routing, ret)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmoe")
    ap.add_argument("--routing", required=True)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()
    sys.exit(0 if run(a.model, a.routing, a.max_steps, a.gpus, a.backend) else 1)


if __name__ == "__main__":
    main()
