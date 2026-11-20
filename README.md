# moescope

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
git clone https://github.com/leehyeoklee/moescope.git
cd moescope
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -e .                   # core: numpy, scikit-learn, torch (CPU is enough for verify-ep / train / tests)
pip install -e ".[model]"          # + transformers etc. for `collect` / `demo` (GPU recommended)
```

On Windows, `torchrun` has a libuv issue, so use the equivalent self-launching command instead:
`moescope measure-ep --model olmoe --routing traces/olmoe.npz --gpus 4` (Linux can use either).

## Quickstart

```bash
# 1) Verify the core claim locally (no GPU, no model): real all_to_all vs computed
moescope verify-ep --gpus 4

# 2) Collect routing from a real model (GPU)
moescope collect --model olmoe --task e2 --prompts examples/prompts.json --out traces/olmoe.npz

# 3) Train a traffic→token classifier from the trace
moescope train --model olmoe --task output --traces traces/olmoe.npz --out models/olmoe_output.joblib
moescope train --model olmoe --task output --traces traces/olmoe.npz --window causal \
            --out models/olmoe_output_causal.joblib        # for live demo

# 4) Confirm measured==computed on real multi-GPU NCCL
torchrun --nproc_per_node=4 -m moescope.measure_ep --model olmoe --routing traces/olmoe.npz

# 5) Live reconstruction demo (left: model generating; right: recovery from traffic)
moescope demo --model olmoe --output-clf models/olmoe_output_causal.joblib
moescope demo --model olmoe --mock-model            # UI/pipeline preview without a model
```

## EP serving on multiple GPUs

`moescope serve-ep` runs the model itself in expert-parallel form. One process owns one GPU. Experts
are split in index order (`E/G` per GPU); rank 0 holds the request, the shared parts of the model
(embeddings, attention, routers) and its own experts, and every other rank holds only its experts.
In each MoE layer rank 0 dispatches tokens to the GPUs of the selected experts with a real
`all_to_all`, those GPUs run the expert computation, and the results are combined on rank 0. The
number of rows actually sent to each GPU is recorded per layer — that is the measured traffic.

```bash
# decode: generate token by token, record per-step traffic       (keys: tokens, routing, measured, ...)
moescope serve-ep --model olmoe --task e2 --prompts examples/prompts.json --gpus 4 --out traces/olmoe_ep.npz --check

# prefill: feed each prompt at once, record per-prompt traffic   (keys: e1_routing, e1_lengths, e1_measured)
moescope serve-ep --model olmoe --task e1 --prompts examples/prompts.json --gpus 4 --out traces/olmoe_ep_e1.npz --check

# train the classifiers on the measured traffic
moescope train --model olmoe --task output --traces traces/olmoe_ep.npz    --gpus 4 --source measured
moescope train --model olmoe --task domain --traces traces/olmoe_ep_e1.npz --gpus 4 --source measured
```

`serve-ep` prints a report and exits non-zero if any layer's measured traffic differs from the
count computed from the routing (`layer_calls_mismatch`). With `--check` the same inputs are first
run on a single GPU with the unmodified model, and the report adds `token_agreement` (decode) or
`max_logit_diff` (prefill). `--check` needs the whole model on GPU 0 once (about 14 GB for OLMoE).

Running on a 4-GPU AWS instance (e.g. `g6.12xlarge`, 4× NVIDIA L4, with a PyTorch/CUDA image):

```bash
git clone https://github.com/leehyeoklee/moescope.git && cd moescope
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[model,dev]"
nvidia-smi --query-gpu=index,name,memory.total --format=csv     # expect 4 GPUs
pytest -q                                                        # CPU tests, including a small EP-served model
moescope serve-ep --model olmoe --task e2 --prompts examples/prompts.json --gpus 4 --out traces/olmoe_ep.npz --check
```

Each process loads the checkpoint on CPU in turn before moving its share to its GPU, so the host
needs roughly one model's worth of RAM free (about 14 GB for OLMoE in bfloat16).

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
and revisions are pinned in `moescope/config.py`; the serving/measuring model must match the one used
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
moescope/        config · placement · traffic · ep · collect · serve_ep · classifiers · measure_ep · demo · cli
tests/        test_ep (gloo measured==computed) · test_serve_ep (EP-served model == single process) · test_placement · test_traffic
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
