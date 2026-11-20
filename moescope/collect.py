"""실제 모델에서 라우팅(top-k Expert)을 뽑아 트레이스(.npz)로 저장.

transformers의 output_router_logits를 쓴다(모델 공용). 층별 router_logits
(num_tokens, num_experts)의 top-k 인덱스 = 선택 Expert id.

스키마
  E2(출력): tokens(N,), routing(N, layers, k)uint8, prompt_ids(N,), step(N,), generated(str)
  E1(입력): e1_routing(sumTok, layers, k), e1_lengths(P,), e1_ids(P,)

routing_from_logits만 모델 없이 단위 테스트된다. 나머지는 GPU(실제 모델) 필요.
"""
import numpy as np


def routing_from_logits(router_logits, top_k, experts=None):
    """층별 router_logits 튜플 → (num_tokens, num_layers, top_k) 선택 Expert id."""
    import torch
    layers = []
    for lg in router_logits:
        t = (lg if isinstance(lg, torch.Tensor) else torch.as_tensor(lg)).float()
        layers.append(torch.topk(t, k=top_k, dim=-1).indices)
    arr = torch.stack(layers, dim=1).cpu().numpy()
    return arr.astype(np.uint8 if (experts is None or experts <= 256) else np.int32)


def load_tokenizer(repo, revision):
    """토크나이저를 불러온다. Granite처럼 기본 방식으로 열리지 않는 체크포인트는 GPT2TokenizerFast로 연다."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(repo, revision=revision)
    except ValueError:
        from transformers import GPT2TokenizerFast
        return GPT2TokenizerFast.from_pretrained(repo, revision=revision)


def load_model(spec, dtype="bfloat16", device_map="auto"):
    import torch
    from transformers import AutoModelForCausalLM
    tok = load_tokenizer(spec.repo, spec.revision)
    model = AutoModelForCausalLM.from_pretrained(
        spec.repo, revision=spec.revision, torch_dtype=getattr(torch, dtype),
        device_map=device_map, output_router_logits=True).eval()
    return model, tok


def collect_e2(model, tok, prompts, spec, max_new_tokens=64, greedy=True):
    """출력 토큰 복원용: 생성하며 스텝별 라우팅 수집 → v7 스키마 dict."""
    import torch
    toks, routes, pids, steps, gen_texts = [], [], [], [], []
    for pid, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").to(model.device)
        cur, past, produced = ids.input_ids, None, []
        with torch.no_grad():
            for s in range(max_new_tokens):
                out = model(input_ids=cur if past is None else cur[:, -1:],
                            past_key_values=past, use_cache=True, output_router_logits=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                nxt = int(logits.argmax(-1)) if greedy else int(torch.multinomial(logits.softmax(-1), 1))
                rl = [lg[-1:] for lg in out.router_logits]
                routes.append(routing_from_logits(rl, spec.top_k, spec.experts)[0])
                toks.append(nxt); pids.append(pid); steps.append(s); produced.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], device=model.device)], 1)
                if nxt == tok.eos_token_id:
                    break
        gen_texts.append(tok.decode(produced))
    return {"tokens": np.array(toks, np.int64), "routing": np.stack(routes).astype(np.uint8),
            "prompt_ids": np.array(pids, np.int64), "step": np.array(steps, np.int64),
            "generated": np.array("\u0001".join(gen_texts))}


def collect_e1(model, tok, prompts, spec, max_len=32):
    """입력 분야용: 프롬프트 prefill 라우팅(토큰별) → e1 스키마 dict."""
    import torch
    routings, lengths = [], []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            out = model(**ids, output_router_logits=True, use_cache=False)
        r = routing_from_logits(out.router_logits, spec.top_k, spec.experts)
        routings.append(r); lengths.append(len(r))
    return {"e1_routing": np.concatenate(routings).astype(np.uint8),
            "e1_lengths": np.array(lengths, np.int32),
            "e1_ids": np.arange(len(prompts), np.int32)}
