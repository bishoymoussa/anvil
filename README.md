# Anvil

A research-first, evaluation-first inference library.

> **Status: pre-alpha (v0 in progress).** Building toward v0 per the design manuscript in [`docs/design.md`](docs/design.md). The current build is at **Milestone 0** ("hello, eval"): smallest end-to-end slice with the HuggingFace slow path.

## What this is

Anvil is **not** trying to be the fastest inference engine. vLLM and SGLang win throughput. Anvil's identity is correctness, reproducibility, and research ergonomics:

- Every run produces a content-hashed [`Manifest`](src/anvil/manifest/schema.py): two runs with the same manifest must produce identical numbers.
- Every chat template, tokenization, and sampler is a versioned, hashed object — not a string loaded from a file at runtime.
- Day-zero new-model coverage via a transformers slow path; popular architectures graduate to a fast path.
- Per-request logits processors and hidden-state extraction are stable public APIs (the V0-vLLM API, restored).
- A preflight CaaS agent runs a smoke test + quality sentinel before every major run, catches the silent failures (missing chat template, EOS misconfigured, OOM-from-bad-config), and either fixes them or refuses to publish a manifest that crossed a silent regression.

See [`docs/design.md`](docs/design.md) for the full design rationale.

## Install (dev)

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quickstart (M0)

```python
import anvil

result = anvil.eval(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tasks=["gsm8k"],
    n_fewshot=5,
    limit=50,
)
print(result.scores["gsm8k"]["accuracy"])
result.manifest.save("run.json")
```

### A note for M0 users

In M0 the only available backend is the HuggingFace slow path
(`transformers.AutoModelForCausalLM`). vLLM lands in M1.

CaaS lands in M3, so M0 will not catch (for example) the well-known
**Llama-3 EOT runaway**: Llama-3-Instruct emits `<|eot_id|>` (token id `128009`)
as the assistant turn terminator, but transformers' default generation will
not stop on it unless you set `stop_token_ids=[128009]` (Anvil exposes
`Sampler` for this). Until M3, set this manually for Llama-3-Instruct
families or your GSM8K score will be effectively meaningless.

## Milestones

The build proceeds milestone-by-milestone (`docs/design.md` §16.10):

- **M0** — HF slow path, GSM8K, manifest emitted (no signing yet).
- **M1** — vLLM wrapper + ChatTemplate canonicalization + MMLU/HumanEval+.
- **M2** — Manifest canonical JSON + sign/verify/diff/replay.
- **M3** — CaaS rule engine + 15 KB seed entries + test corpus.
- **M4** — Multimodal (Qwen2.5-VL fast path + MMMU).
- **M5** — lm-eval-harness shim + custom non-text modality.
- **M6** — uv wheels (cu121/cu128/cu130), 5 fast paths, OpenAI-compatible serve, `anvil doctor`.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
