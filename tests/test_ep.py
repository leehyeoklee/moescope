"""핵심 테스트: gloo(CPU) 실제 all_to_all 송신 수 == 라우팅 계산값, 왕복 정확."""
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from moetl.ep import ep_layer
from moetl.placement import dest_counts_full


def _worker(rank, world, N, E, k, port, ret):
    os.environ.update(USE_LIBUV="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
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


@pytest.mark.parametrize("world,N,E,k,port", [
    (4, 200, 64, 8, 29601),   # olmoe
    (4, 200, 40, 8, 29602),   # granite
    (4, 200, 60, 4, 29603),   # qwen
    (8, 240, 64, 8, 29604),   # 8-gpu
])
def test_measured_equals_computed(world, N, E, k, port):
    mgr = mp.Manager(); ret = mgr.dict()
    mp.spawn(_worker, args=(world, N, E, k, port, ret), nprocs=world, join=True)
    assert ret["measured"] == ret["computed"], "측정 송신 수 != 라우팅 계산값"
    assert all(ret[f"rt_{r}"] for r in range(world)), "왕복(dispatch→combine) 불일치"
