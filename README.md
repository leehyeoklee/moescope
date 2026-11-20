# moescope

Serve open Mixture-of-Experts (MoE) language models in expert-parallel (EP) form across several
GPUs, measure the GPU-to-GPU traffic of every MoE layer while the model runs, and estimate the
input topic and the generated tokens from that traffic alone.

In EP serving the experts of each MoE layer are spread over the GPUs. A router picks the top-k
experts for every token, the token is sent to the GPUs that hold those experts with an
`all_to_all`, the experts are computed there, and the results are sent back and combined. How many
rows go to each GPU depends on which experts were picked, and that depends on the input. moescope
runs this for real, records the per-layer, per-GPU row counts, and trains classifiers on them.

## What it does

- **EP serving.** One process per GPU. Experts are split in index order (`E/G` per GPU). The
  model produces the same output as when it runs on a single GPU.
- **Traffic measurement.** For every MoE layer, the number of rows actually sent to each GPU is
  recorded: per prompt in the prefill stage, per generated token in the decode stage.
- **Feature extraction and classifiers.** The measured traffic is turned into classifier inputs
  (prefill traffic divided by the number of input tokens; decode traffic per generated token). An
  input-topic classifier and an output-token classifier (standardization followed by logistic
  regression) are trained and evaluated on them.

## Verified results

EP-served output compared with the unmodified model, and measured traffic compared with the count
computed from the routing:

| model | hardware | processes | generated tokens equal | max logit difference | layers with traffic mismatch |
|---|---|---|---|---|---|
| OLMoE-1B-7B Instruct | 2× Tesla T4 (NCCL) | 2 (32 experts each) | 160 / 160 | 0 | 0 |
| Granite-3.0-3B-A800M | 2× Tesla T4 (NCCL) | 2 (20 experts each) | 160 / 160 | 0 | 0 |

Both runs used 5 prompts with 32 generated tokens each, plus a prefill run of the same prompts
(max logit difference 0). The CPU test suite repeats the same comparison on small randomly
initialized models with the OLMoE, Granite and Qwen1.5-MoE layouts, using 1, 2 and 4 processes.

## Install

```bash
git clone https://github.com/leehyeoklee/moescope.git
cd moescope
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[model,dev]"      # torch, transformers, accelerate, scikit-learn, pytest
```

## Quickstart

```bash
# 1) Serve the model in EP form and record the traffic of the decode stage (token by token)
moescope serve-ep --model olmoe --task decode --prompts examples/prompts.json --gpus 4 --out traces/olmoe_ep_decode.npz --check

# 2) Same for the prefill stage (each prompt fed at once)
moescope serve-ep --model olmoe --task prefill --prompts examples/prompts.json --gpus 4 --out traces/olmoe_ep_prefill.npz --check

# 3) Train the classifiers on the measured traffic
moescope train --model olmoe --task output --traces traces/olmoe_ep_decode.npz    --gpus 4 --source measured
moescope train --model olmoe --task domain --traces traces/olmoe_ep_prefill.npz --gpus 4 --source measured
```

`serve-ep` prints a report and ends with `RESULT: PASS` when the measured traffic of every layer
equals the count computed from the routing (`layer_calls_mismatch: 0`).

While it runs, every process prints timestamped progress lines (`--quiet` turns them off):

```
[12:01:03 rank0] started on cuda:0 (NVIDIA L4), backend nccl, experts 0..15 of 64
[12:01:03 rank0] loading the checkpoint on CPU
[12:02:10 rank0] experts 0..15 on cuda:0, 5.1 GiB allocated
[12:03:40 rank0] all 4 processes ready, serving in EP form
[12:03:44 rank0] decode 1/30: 21 input tokens, 32 generated, 3.9s, traffic mismatches so far 0
```

With `--check`, the same inputs are first run with the unmodified model and the report adds:

| field | stage | meaning |
|---|---|---|
| `token_agreement` | decode | share of freely generated tokens equal to the unmodified model |
| `forced_token_agreement` | decode | share of steps with the same next token when both are fed the same tokens |
| `forced_max_logit_diff` | decode | largest logit difference over those steps |
| `max_logit_diff` | prefill | largest logit difference at the last position |

For this comparison the unmodified model is spread over the available GPUs and uses per-expert
computation (`experts_implementation="eager"`), the same arithmetic as the EP path.

### Output files

| task | key | shape | content |
|---|---|---|---|
| `decode` | `tokens` | (N,) | generated token ids |
| | `routing` | (N, layers, k) | top-k expert ids per step |
| | `measured` | (N, layers, G) | rows sent to each GPU per step |
| | `prefill_measured` | (P, layers, G) | rows sent to each GPU while reading each prompt |
| | `prompt_ids`, `step`, `generated` | | prompt index, step index, decoded text |
| `prefill` | `prefill_routing` | (tokens, layers, k) | top-k expert ids per input token |
| | `prefill_lengths` | (P,) | number of input tokens per prompt |
| | `prefill_measured` | (P, layers, G) | rows sent to each GPU per prompt |

## Supported models and GPU counts

Checkpoints and revisions are pinned in `moescope/config.py`.

| `--model` | checkpoint | MoE layers | experts | top-k | usable `--gpus` |
|---|---|---|---|---|---|
| `olmoe` | allenai/OLMoE-1B-7B-0125-Instruct | 16 | 64 | 8 | 1, 2, 4, 8, 16, 32, 64 |
| `granite` | ibm-granite/granite-3.0-3b-a800m-instruct | 32 | 40 | 8 | 1, 2, 4, 5, 8, 10, 20, 40 |
| `qwen` | Qwen/Qwen1.5-MoE-A2.7B-Chat | 24 | 60 | 4 | 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60 |

`--gpus` must divide the number of experts and must not exceed the GPUs of the machine;
`serve-ep` stops before loading the model otherwise.

## Running on a multi-GPU instance

Example for a 4-GPU AWS instance (`g6.12xlarge`, 4× NVIDIA L4) with a PyTorch/CUDA image:

```bash
git clone https://github.com/leehyeoklee/moescope.git && cd moescope
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[model,dev]"
nvidia-smi --query-gpu=index,name,memory.total --format=csv     # expect 4 GPUs
pytest -q                                                        # CPU tests, including small EP-served models
moescope serve-ep --model olmoe --task decode --prompts examples/prompts.json --gpus 4 --out traces/olmoe_ep_decode.npz --check
```

Each process loads the checkpoint on CPU in turn and then moves its share to its GPU, so the host
needs about one model's worth of free RAM (about 14 GB for OLMoE in bfloat16).

## How it works

```
prompt ─▶ rank 0 (GPU 0): embeddings, attention, routers, experts 0 … E/G-1
             │
             │  every MoE layer:
             │    router picks top-k experts per token
             │    all_to_all dispatch ─▶ rank r (GPU r): experts r·E/G … (r+1)·E/G-1 compute
             │    all_to_all combine  ◀─ results, weighted sum on rank 0
             │    rows sent per destination GPU are recorded            ─▶ measured traffic
             ▼
        next layer … logits

measured traffic ─▶ traffic.measured_features / measured_prompt_features ─▶ classifiers
```

- `serve_ep` loads the model in every process, keeps only that process's experts on its GPU, and
  replaces the forward of the model's experts module with the EP path. Routers and all other
  computation are the model's own code.
- `ep.ep_layer` performs the `all_to_all` dispatch, the local expert computation and the combine
  for one MoE layer and returns the rows sent to each GPU. The same code runs under `NCCL` (GPU)
  and `gloo` (CPU). Half-precision hidden states are carried as bytes, and expert contributions
  are added in ascending expert order so the result equals the unmodified model.
- `placement.dest_counts_full` computes the same counts from the routing; `serve-ep` compares the
  two for every layer call.
- `traffic` turns traffic into classifier inputs; `classifiers` trains and evaluates the
  input-topic and output-token classifiers (token dictionary = tokens seen at least 5 times in
  training).

## Other commands

```bash
moescope verify-ep --gpus 4                      # all_to_all row counts vs computed counts, random routing, CPU
moescope collect --model olmoe --task decode --prompts examples/prompts.json --out traces/olmoe.npz    # routing only, single device
moescope measure-ep --model olmoe --routing traces/olmoe.npz --gpus 4                              # replay a routing trace through all_to_all
moescope train --model olmoe --task output --traces traces/olmoe.npz --window causal --out models/olmoe_output_causal.joblib
moescope demo --model olmoe --output-clf models/olmoe_output_causal.joblib                         # live generation next to the estimated tokens
```

## Project layout

```
moescope/     config · placement · ep · serve_ep · collect · traffic · classifiers · measure_ep · demo · cli
tests/        test_serve_ep (EP-served model == single process) · test_ep · test_placement · test_traffic
examples/     sample prompts
```

## Tests

```bash
pytest -q
```

22 tests, all on CPU. The 11 EP-serving tests need `transformers` and are skipped without it.

## License

MIT — see [LICENSE](LICENSE).
