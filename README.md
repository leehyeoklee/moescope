# moe-traffic-leak

**Does the volume of GPU-to-GPU communication leak what a Mixture-of-Experts model is doing?**

In expert-parallel (EP) MoE inference, each layer routes every token to the GPUs that hold its
selected experts via an `all_to_all`. The *number of tokens* sent to each GPU depends on the
routing — which depends on the input. This project is a small toolkit to study that channel:
measure the real inter-GPU traffic, and show that **the traffic volume alone is enough to
recover the input topic and the generated tokens.**

The central, testable claim:

> The number of rows an EP `all_to_all` actually sends to each destination GPU **equals** the
> count computed directly from the routing table. So an observer who sees only per-destination
> traffic volume sees the routing — and routing leaks content.

This is verified in code on both CPU (`gloo`) and real GPUs (`NCCL`).

## Key result

`measured == computed` — the real `all_to_all` send counts equal the routing-derived counts,
with **zero** mismatch:

| backend | hardware | model(s) | steps | mismatched layers |
|---|---|---|---|---|
| gloo | CPU | olmoe, granite, qwen (G=4, 8) | — | **0** |
| NCCL | 2× Tesla T4 | olmoe, real routing | 240 | **0 / 16** |

On the T4 run, routing was also fully deterministic across repeats (`repeat_same_routing = 1.0`).

Because the equality is exact integer counting over the *same* linear expert→GPU placement, it
holds for any EP deployment using that placement — including larger GPU counts — so results
obtained by computing traffic from routing transfer to a real multi-GPU deployment.

## Install

```bash
pip install -e .            # core: numpy, scikit-learn, torch (CPU is enough for verify-ep / train / tests)
pip install -e ".[model]"   # + transformers etc. for `collect` / `demo` (GPU recommended)
```

## Quickstart

```bash
# 1) Verify the core claim locally (no GPU, no model): real all_to_all vs computed
moetl verify-ep --gpus 4

# 2) Collect routing from a real model (GPU)
moetl collect --model olmoe --task e2 --prompts examples/prompts.json --out traces/olmoe.npz

# 3) Train a traffic→token classifier from the trace
moetl train --model olmoe --task output --traces traces/olmoe.npz --out models/olmoe_output.joblib
moetl train --model olmoe --task output --traces traces/olmoe.npz --window causal \
            --out models/olmoe_output_causal.joblib        # for live demo

# 4) Confirm measured==computed on real multi-GPU NCCL
torchrun --nproc_per_node=4 -m moetl.measure_ep --model olmoe --routing traces/olmoe.npz

# 5) Live reconstruction demo (left: model generating; right: recovery from traffic)
moetl demo --model olmoe --output-clf models/olmoe_output_causal.joblib
moetl demo --model olmoe --mock-model            # UI/pipeline preview without a model
```

## How it works

```
prompt ─▶ MoE model (EP across G GPUs)
             │  each layer: route tokens to experts' GPUs via all_to_all
             ▼
     ep.ep_layer  ── measures rows sent per destination GPU  ◀── the only thing an observer sees
             │
             ▼
   traffic.volume_features  (per-layer, per-destination counts)
             │
             ▼
   classifiers  ── input topic / output token recovery
```

- `ep.ep_layer` performs a dropless, per-`(token, expert)` `all_to_all` and returns the measured
  per-destination send counts. The same function runs under `gloo` (CPU) and `NCCL` (GPU).
- `placement.dest_counts_full` computes the expected counts from the routing table.
- `collect` extracts routing from a real model via `output_router_logits` (top-k indices).
- `classifiers` trains logistic-regression recovers of input topic and output token from the
  traffic features (dictionary = tokens seen ≥ 5× in training).

## Supported models

`olmoe` (OLMoE-1B-7B), `granite` (Granite-3.0-3B-A800M), `qwen` (Qwen1.5-MoE-A2.7B). Checkpoints
and revisions are pinned in `moetl/config.py`; the serving/measuring model must match the one used
to collect the training traces.

## Limitations

- **Interconnect.** Routing-dependent `all_to_all` of this kind needs an EP backend that sends
  only to the needed GPUs. Production engines use it on NVLink/RDMA; on PCIe-only setups their
  default backends send fixed-size traffic, so nothing leaks there. This toolkit's custom
  `all_to_all` works on any interconnect but is not a production serving engine.
- **Single request.** All recovery assumes one request at a time; mixing concurrent requests sums
  the traffic and degrades recovery quickly.
- **Precision.** Low-precision (bf16) routing can flip near-tied top-k choices across runs on some
  hardware; use an fp32 router / report the repeat-match rate. (It was 1.0 on T4.)
- **Upper bound.** Traffic computed from routing is a noise-free upper bound on what a real
  side-channel observer could measure.

## Project layout

```
moetl/        config · placement · traffic · ep · collect · classifiers · measure_ep · demo · cli
tests/        test_ep (gloo measured==computed) · test_placement · test_traffic
examples/     sample prompts
```

## Tests

```bash
pip install pytest && pytest -q
```

CI runs the CPU tests (including the `gloo` measured==computed check) on every push.

## Disclaimer

This is a research tool for studying and defending against a privacy side-channel in MoE serving.
Use it on systems you own or are authorized to analyze.

## License

MIT — see [LICENSE](LICENSE).
