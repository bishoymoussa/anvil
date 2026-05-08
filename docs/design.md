<p align="center">
  <img src="assets/anvil_logo_name.png" alt="anvil" width="360" />
</p>

# A Research-First, Evaluation-First Inference Library

**Design Manuscript — v1.0 (May 2026)**

> Working codename: **Anvil**. (Final name TBD; alternatives in §15.)

---

## 0. Executive Summary

vLLM has won production inference and is structurally hostile to the people doing the science: per-request logits processors removed in V1, hidden-state extraction unsupported, day-zero model support delayed and buggy, chat templates and sampling defaults that change silently between minor releases, and a torch/CUDA matrix that breaks on every other consumer GPU. The eval ecosystem on top of it is duct tape — lm-evaluation-harness has chat-template bugs that move scores by tens of points, runs 40× slower than it should against vLLM after API drift, and per-task YAMLs that are fragile by design. lighteval is ~2× slower than that. OpenCompass is heavyweight. Nobody owns the "fast, reproducible, research-first" niche.

This manuscript specifies a library that fills it. It is built on **three pillars**:

1. **Research as a first-class user.** Per-request logits processors, hidden-state extraction, structured output, and custom decoding strategies are stable, versioned APIs, not forks of the engine.
2. **Datasets and benchmarks integrate in five lines.** A versioned task spec compatible with lm-evaluation-harness migration, batched evaluation primitives that drive the engine at full throughput, and a built-in library of the benchmarks that actually matter. The `Task` abstraction is modality-agnostic: text, multimodal, and arbitrary inputs (RNA, audio, graphs, anything) all use the same five-line registration.
3. **Day-zero model support, by default.** New HuggingFace architectures load via the Transformers backend the day they drop. The top ~30 architectures have a fast path. Multimodal is first-class, not bolted on.

A fourth, transverse principle ties them together: **reproducibility by construction**. Every run produces a manifest with the model SHA, dataset SHA, chat-template hash, sampler params, library version, and tokenizer version. Two runs with the same manifest must produce identical numbers.

A fifth feature is the **CaaS preflight agent** (Coder-as-a-Service): a small open-source coder model that runs a 1+ε sample smoke test before any major run, catches silent failures (chat template missing, EOS misconfigured, OOM-from-bad-config, TP-divisibility, `mm_processor_kwargs` defaults, dtype mismatches), proposes a fix, and either applies it (CI mode) or surfaces it as a diff (research mode). Every CaaS-applied delta lands in the provenance manifest. This is the first system in this space designed to *prevent* the silent quality regressions that are the dominant failure mode in modern LLM evaluation.

We are not trying to be vLLM at scale. We are not trying to win throughput benchmarks. We are trying to make the question *"did this model actually score X on benchmark Y?"* answerable, reproducible, and not subject to the whims of last week's vLLM release.

---

## 1. Problem Statement

### 1.1 What goes wrong, concretely

A researcher wants to evaluate Qwen3-VL-8B on MMMU. They install vLLM. The wheel is built against CUDA 12.x, their host has CUDA 13.x — the import fails (vLLM #28669). They downgrade. They get the engine running, but the model OOMs expecting 256 GB of memory on a 2B variant because `mm_processor_kwargs` defaults reserve 163,840 tokens per image (#27706). They lower image budgets. The model loads. They wire up lm-evaluation-harness, run 14,000 samples, and discover at sample 12,000 that the chat template wasn't being applied — Llama-3-8B-Instruct fell back to base-model behavior and the GSM8K score is meaningless (lm-eval-harness #1841). They rerun with `--apply_chat_template`, but `fewshot_as_multiturn` was silently auto-set to a different value, the sampling defaults are now read from `generation_config.json` (vLLM v0.8.0 PR #12622), and the score has moved 12 points. They have no way to tell which delta caused which change.

This is not an edge case. This is the median experience.

### 1.2 vLLM's structural problems

- **V0/V1 transition** (Jan 2025 → V1 default v0.8.0 → V0 fully removed v0.11.0) deliberately removed per-request logits processors, `best_of`, GPU↔CPU KV swapping, and request-level structured output backends. LoRA in V1 is documented as slower than in V0. The transition introduced four silent V1-vs-V0 numerical mismatches that took a HuggingFace blog post to debug.
- **PyTorch lock-step.** vLLM wheels compile against a single torch+CUDA combination per release. The official docs admit: *"vLLM has to compile many cuda kernels [...] it is recommended to install vLLM with a fresh new environment."* This is a confession that the dependency story is fragile by design.
- **Default sampling source flipped in v0.8.0.** Same prompt, same flags, different output between v0.7 and v0.8 unless you explicitly set `--generation-config vllm`. This is a silent reproducibility break.
- **Per-request logits processors gone in V1** (RFC #13360, issue #21672). Custom decoding strategies — DoLa, contrastive decoding, classifier-free guidance, custom logit biases — now require forking the engine.
- **Hidden states unsupported officially.** RFC #33118 still open. Community workarounds (vllm-lens, speculators) require *"heavy modification and patching of vLLM"* per their authors.
- **New-model support is rapid in the press release, slow in real life.** Qwen3.5/3.6 broke between v0.17.1 and v0.18.0 (#37749). DeepSeek-V3.2 needed `--tokenizer-mode deepseek_v32` not registered in v0.12.0 (#30933). Llama 4 vision regressed after PR #21126 for weeks (#25888). Gemma 4 tool calling produces 4096 `<pad>` tokens under concurrency (#39392).
- **Tool-call parsers proliferate.** A separate `--tool-call-parser` exists for every model family (`hermes`, `llama3_json`, `mistral`, `qwen3_coder`, `glm47`, `gemma4`, `pythonic`, `deepseek_v3.2`, `holo2`, `internlm`, …) and many are buggy.

### 1.3 The eval ecosystem's brokenness

- **lm-evaluation-harness chat-template bugs are well-documented.** Issue #1841 shows accuracy gaps of tens of points on Llama-3-8B-Instruct GSM8K depending on whether the chat template is applied. Issue #3150 shows Qwen2.5-Math-7B-Instruct producing wildly different scores between HF and vLLM backends because of repeating tokens at the end. Issue #2278/#3214 show GSM8K's "flexible-extract" picking the first number, not the last.
- **Performance against vLLM is bad.** Issue #3152: 10-minute hang on startup. Issue #3292: 40× throughput drop after a vLLM API change. Issue #1625: scores differ between batch_size=1 and batch_size=4 even with deterministic algorithms. Issue #3148: GPU at 0–10% utilization because the harness serializes per-doc.
- **Reproducibility crisis confirmed by the harness team itself** (Biderman et al. 2024, "Lessons from the Trenches on Reproducible Evaluation of Language Models"): minor prompt and template variations move scores by single-digit percentages and the team itself struggles to reproduce paper numbers.
- **lighteval is ~2× slower than lm-eval-harness** on the same A100×4 (HF lighteval issue #179).
- **No widely-adopted eval framework treats chat templates as a versioned, lockable object.** The chat template lives in `tokenizer_config.json` *or* `chat_template.json`, can be overridden by the server's `--chat-template`, can be silently overridden by `generation_config.json` defaults, and reasoning parsers (`--reasoning-parser qwen3` etc.) may further mangle output before it reaches the metric function.

### 1.4 The opportunity

The competitor is not vLLM-as-server. The competitor is **"lm-evaluation-harness + vLLM" as a duct-tape pipeline**. We are replacing the duct tape, not the engine. We can wrap a fast engine (vLLM, SGLang, or our own thin path) and win on the layer above: correctness, reproducibility, research ergonomics, day-zero model coverage, and a single coherent surface that doesn't break every six weeks.

---

## 2. Design Principles

These are the principles every API decision is checked against. They are listed in priority order; when two conflict, the higher one wins.

### 2.1 Correctness over throughput

The number you publish must be the number the model actually scored. We will accept measurable throughput losses (10–30% vs vLLM bare-metal is acceptable) in exchange for "the same manifest produces the same number, today, tomorrow, and on someone else's machine." This is not a slogan; it is a constraint that shows up in concrete decisions:

- The default sampler is **neutral** (greedy, temperature=0, no penalties), independent of `generation_config.json`. Reading from the model's defaults is opt-in via `Sampler.from_generation_config(model_id)`.
- Chat templates are **content-hashed objects**, not strings loaded from a file at runtime. Two runs with the same manifest cannot produce different prompts.
- The smoke test runs a known-answer **quality sentinel** before every major run. Chat-template-missing-and-model-degraded-to-base-quality is detected and refused before sample 1.

### 2.2 Reproducibility by construction

Every run produces a `Manifest` (§8). The manifest is the *primary output* of an evaluation, not a sidecar. The library refuses to publish a leaderboard upload without a manifest, refuses to mix manifests of different library versions in the same comparison, and exposes a `manifest diff <a> <b>` command that explains exactly which deltas could account for any score difference.

### 2.3 Day-zero model support is the default

If a model loads in `transformers.AutoModelForCausalLM`, it loads in this library — same day, same hour. We achieve this by routing unknown architectures through the **Transformers backend** by default, accepting the throughput loss, and graduating popular architectures to the **fast path** as PRs land. The user never has to wait for a release to evaluate a model that dropped this morning.

### 2.4 Research is a first-class user

Per-request logits processors. Hidden-state extraction. Custom samplers. Activation hooks. Top-k logprobs at any layer. Streaming intermediate state. These are documented public APIs with semver guarantees, not undocumented hacks behind environment variables.

### 2.5 Stable, versioned, narrow APIs

Two API tiers:

- **Public** (`anvil.eval`, `anvil.serve`, `anvil.models`, `anvil.datasets`): semver. Breaking changes only at major releases, with a `anvil upgrade --doctor` that flags every code change required.
- **Internal** (`anvil._engine`): may break at any minor release.

We will resist API surface growth. New flags are a tax. New abstractions are a tax. The CLI will not have 47 mutually-exclusive `--enable-*` and `--reasoning-parser` and `--tool-call-parser` flags. Configuration is a typed Python object or a YAML; the CLI is a thin wrapper.

### 2.6 Self-healing preflight (CaaS)

Before every major run, the library runs a 3-sample smoke test plus a quality sentinel (§7). If anything fails, the CaaS agent attempts a bounded fix using a rule engine + curated known-issue database, escalating to a small coder LLM only when both miss. Every fix is recorded in the manifest. The user can audit, accept, reject, or pin the original config. This is the line of defense between researchers and the silent failures that the rest of this document is trying to prevent.

### 2.7 uv-installable, no torch lock-step

`uv pip install anvil` works on day one against any torch ≥ 2.4 + CUDA ≥ 12.1, with a wheel matrix (`cu121`, `cu124`, `cu126`, `cu128`, `cu130`, plus a CPU fallback) that mirrors PyTorch's. We refuse to lock the user's torch version. ABI is detected at install time. If the user has a torch we don't have a wheel for, we fall back to a pure-Python path with a clear warning, not a hard failure.

---

## 3. System Architecture

```
                        ┌──────────────────────────────┐
                        │   Public API (anvil.*)       │
                        │   - eval / serve / models    │
                        │   - datasets / metrics       │
                        └─────────────┬────────────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
┌──────▼──────┐               ┌───────▼────────┐             ┌───────▼────────┐
│  Eval       │               │  CaaS          │             │  Provenance    │
│  Runner     │               │  Preflight     │             │  Manifest      │
│  (batched   │               │  Agent (§7)    │             │  (§8)          │
│   primitives)│              │                │             │                │
└──────┬──────┘               └───────┬────────┘             └───────┬────────┘
       │                              │                              │
       └────────┬─────────────────────┴──────────────────────────────┘
                │
        ┌───────▼────────┐         ┌──────────────────┐
        │  Engine        │ ◀─────▶ │  ChatTemplate /  │
        │  (vLLM-wrap or │         │  Tokenization /  │
        │   SGLang-wrap  │         │  Sampler /       │
        │   or thin HF+  │         │  LogitsProcessor │
        │   FlashAttn)   │         │  (§4)            │
        └───────┬────────┘         └──────────────────┘
                │
        ┌───────▼────────┐
        │  Models        │
        │  - Transformers│
        │    backend     │
        │    (default)   │
        │  - Fast path   │
        │    (~30 archs) │
        └────────────────┘
```

### 3.1 Engine layer

We do not write attention kernels. The engine layer is a thin facade over a real engine. v0 ships **two backends** behind one interface:

- **`anvil-engine-vllm`** (default for fast-path architectures): a stable wrapper around vLLM that pins to the last-known-good vLLM minor version per Anvil release, normalizes the CLI surface, and exposes the abstractions in §4 *regardless of vLLM version* — including reimplementing per-request logits processors and hidden-state extraction in the wrapper layer where vLLM V1 dropped them.
- **`anvil-engine-hf`** (default for unknown architectures): `transformers.AutoModelForCausalLM` + FlashAttention-2/3 if available + a thin paged KV cache + continuous batching. Slower than vLLM but loads anything HF loads.

Optional backends as plugins (post-v0): `anvil-engine-sglang`, `anvil-engine-lmdeploy`. Same facade. Same manifest fields. A user can switch backends with `--engine sglang` and the manifest records which engine produced the run; if scores differ across engines, that is *itself* a finding the manifest makes visible.

### 3.2 Evaluation layer

A native, batched evaluation runner that drives the engine at full throughput. Implements the four primitives (`loglikelihood`, `loglikelihood_rolling`, `generate_until`, `generate_logprobs`) as batched calls, not per-doc Python loops. Reuses lm-evaluation-harness task definitions where possible (with a documented compatibility shim). See §6.

### 3.3 Models layer

Two paths into the same `Model` interface:

- **Fast path:** ~30 hand-tuned architectures (Llama 2/3/4, Qwen 2/2.5/3/3.5/3.6, DeepSeek V2/V3, Mistral, Mixtral, Gemma 2/3/4, Phi-3/4, Yi, Command-R, MiniCPM, InternLM, GLM-4, Qwen-VL, LLaVA, InternVL, Pixtral, Phi-3-Vision, Idefics, Molmo, CogVLM). Optimized kernels, custom attention, fused MLPs.
- **Slow path:** anything else, via `transformers.AutoModelForCausalLM` and `AutoModelForVision2Seq`. Loads day-zero. Throughput within 2–4× of the fast path on most architectures.

Architecture detection is automatic: `anvil.load(model_id)` inspects `config.json`, picks fast path if available, slow path otherwise, and emits a one-line log with the choice. Users can force `--model-impl=transformers` or `--model-impl=fast`.

### 3.4 CaaS layer

A same-process Python module invoked by `Engine.preflight()`. Loads no LLM in the steady state; loads a 7B coder model lazily on first failure if both rule engine and known-issue DB miss. Detailed in §7.

### 3.5 Provenance layer

Cross-cutting. Every component writes to a single immutable manifest object that is sealed at end-of-run. Detailed in §8.

---

## 4. Core Abstractions: Versioned First-Class Objects

This is where Anvil structurally differs from every existing library. Six objects, each with a content hash, each pinnable in a manifest.

### 4.1 `ChatTemplate`

```python
from anvil import ChatTemplate

ct = ChatTemplate.from_model("Qwen/Qwen2.5-7B-Instruct")
print(ct.hash)          # sha256 of the canonicalized Jinja source
print(ct.version)       # "qwen2.5-instruct@v1"
print(ct.source)        # "tokenizer_config.json" | "chat_template.json" | "user-supplied"
print(ct.fewshot_style) # "interleaved" | "concat-system" | "raw"
```

A `ChatTemplate` is the Jinja source plus an explicit declaration of how few-shot examples are structured (interleaved as separate turns, concatenated into the system message, or pre/postpended raw). This eliminates the lm-evaluation-harness ambiguity where `apply_chat_template + fewshot_as_multiturn` silently produces different prompts than `apply_chat_template` alone. The fewshot style is part of the hash.

`ChatTemplate.canonicalize()` strips whitespace, normalizes quoting, and sorts conditional branches by their canonical form so cosmetic edits don't change the hash. The hash *is* the identity of the template.

### 4.2 `Tokenization`

```python
from anvil import Tokenization

tok = Tokenization.from_model("Qwen/Qwen2.5-7B-Instruct")
tok.bos_token_handling   # "auto-prepend" | "never" | "from-template"
tok.eos_token_ids        # [151645, 151643]   <-- explicit, all candidates
tok.padding_side         # "left"
tok.add_special_tokens   # False (when chat_template is applied)
tok.assert_no_double_bos = True   # raises if a prompt would double-BOS
```

A `Tokenization` is the tokenizer plus *assertions* about how it should be used. Double-BOS, wrong padding side, missing EOS candidates — these are caught at construction, not at sample 12,000.

### 4.3 `Sampler`

```python
from anvil import Sampler

# Neutral default. Independent of generation_config.json.
s = Sampler.greedy()

# Explicit opt-in to model defaults.
s = Sampler.from_generation_config("Qwen/Qwen2.5-7B-Instruct")

# Explicit construction.
s = Sampler(temperature=0.7, top_p=0.9, top_k=40, seed=42, max_tokens=2048)

# Reasoning models need long max_tokens.
s = Sampler.for_reasoning_model("DeepSeek-R1", max_tokens=32768)
```

The default is greedy, no penalties, no `generation_config.json` involvement. This is the opposite of vLLM v0.8.0+ — and it is the only way to make scores reproducible across vLLM versions.

`Sampler.hash` includes every parameter that affects output. `Sampler.diff(other)` is a one-line method that explains why two runs produced different numbers.

### 4.4 `LogitsProcessor` (the V0-style API, restored)

```python
from anvil import LogitsProcessor

class DoLaProcessor(LogitsProcessor):
    """Decoding by Contrasting Layers — Chuang et al. 2023."""
    def __init__(self, mature_layer=-1, premature_layers=(0, 12, 24)):
        self.mature_layer = mature_layer
        self.premature_layers = premature_layers

    def process(self, request_id, token_ids, logits, hidden_states):
        # hidden_states is shape [num_layers, seq_len, hidden_dim]
        contrast = logits - hidden_states[self.premature_layers].mean(0) @ self.lm_head_T
        return contrast

# Per-request, exactly like vLLM V0.
engine.generate(
    prompts=["..."],
    sampler=Sampler.greedy(),
    logits_processors=[DoLaProcessor()],
)
```

The processor sees `hidden_states` if it asks for them via `requires_hidden_states = True`, and the engine wires that through automatically. Internally we implement this as a batch-level wrapper that respects argmax-invariance for greedy fast paths. No global registration. No engine restart. No fork.

### 4.5 `HiddenStates`

```python
from anvil import HiddenStateSpec

spec = HiddenStateSpec(layers=[0, 12, 24, -1], positions="all")

result = engine.generate(
    prompts=["The capital of France is"],
    sampler=Sampler.greedy(),
    capture=spec,
)

result.outputs[0].text                # "Paris"
result.outputs[0].hidden_states[12]   # shape [seq_len, hidden_dim]
```

This solves vLLM RFC #33118 in user space. Implementation: hooks into the model's forward pass, copies activations to pinned host memory async, attaches them to the output. Throughput cost: ~5–10% when active, zero when not. Memory cost: scales with `len(layers) * positions * hidden_dim * dtype`; refused if it would exceed configured budget.

### 4.6 `Manifest`

The cross-cutting reproducibility object. See §8.

---

## 5. Models: Day-Zero Integration

### 5.1 The default is the slow path, and the slow path is fine

```python
import anvil

m = anvil.load("Qwen/Qwen3.7-MoE-Instruct")    # released 2 hours ago
out = m.generate("Hello", max_tokens=20)
```

This works. Because if `Qwen3.7-MoE` loads in `transformers.AutoModelForCausalLM` (i.e., it's been merged to transformers main), it loads in Anvil. We do not gate on a vLLM PR being reviewed, merged, released, and the user upgrading.

Throughput on the slow path is 2–4× lower than the fast path on most architectures. For a researcher running an eval the day a model drops, this is the right trade-off. Fast-path support is added in subsequent releases.

### 5.2 The fast path: ~30 architectures, hand-tuned

Initial fast-path coverage (v0):

| Family | Variants |
|---|---|
| Llama | 2, 3.1, 3.2, 3.3, 4 (text + Maverick vision) |
| Qwen | 2, 2.5, 3, 3.5, 3.6 (text + VL) |
| DeepSeek | V2, V3, V3.2, R1 (and distilled variants) |
| Mistral | 7B, Small, Large, Mistral 3.1 |
| Mixtral | 8x7B, 8x22B |
| Gemma | 2, 3, 4 |
| Phi | 3, 3.5, 4, 4-mini |
| Yi | 34B, 1.5 series |
| Command-R | 35B, plus |
| GLM | 4, 4.5, 4.7 |
| Multimodal | Qwen-VL family, LLaVA-1.6, InternVL, Pixtral, Phi-Vision, Idefics, Molmo, CogVLM, MiniCPM-V |

A model is on the fast path if there is a `anvil/models/<family>.py` file with a registered `ModelImpl`. Adding one is the canonical "good first issue" for new contributors.

### 5.3 Plugin protocol for new architectures

For an architecture not in transformers and not on our fast path:

```python
# anvil_plugin_myarch/__init__.py
from anvil.plugins import register_model_impl, ModelImpl

@register_model_impl("MyArchForCausalLM")
class MyArchImpl(ModelImpl):
    def load_weights(self, hf_config, weights_iter): ...
    def forward(self, input_ids, kv_cache, attn_metadata, hidden_state_capture): ...
    # ~5 required methods, ~3 optional
```

Install with `pip install anvil-plugin-myarch`. Anvil discovers it via entry points. The plugin protocol is **versioned** (`anvil.plugins.v1`); breaking changes increment the version, and old plugins keep working until the version is sunset with one major release of warning. This is the contract vLLM RFCs #19161 and #19376 admit they don't have.

### 5.4 Vision-language and multimodal models

Multimodal is where vLLM hurts the most: Qwen2-VL multi-image slicing errors (#23371), Qwen3-VL FP8 async-scheduling crashes (#31679), Qwen2.5-VL 256GB-expected-on-3B-model OOMs (#27706), Llama 4 Maverick vision regressions for weeks after PR #21126 (#25888), Pixtral / InternVL / MiniCPM-V each with their own preprocessor surprises, and a constant churn of `--limit-mm-per-prompt`, `--mm-encoder-tp-mode`, `--mm-processor-kwargs` flags. We treat VLMs as a first-class concern, not a bolted-on capability.

**The user-facing API is uniform across modalities:**

```python
from PIL import Image
import anvil

m = anvil.load("Qwen/Qwen2.5-VL-7B-Instruct")

out = m.generate(
    messages=[{"role": "user", "content": [
        {"type": "image", "image": Image.open("cat.png")},
        {"type": "image", "image": Image.open("dog.png")},
        {"type": "text",  "text": "Compare these two animals."}
    ]}],
    sampler=anvil.Sampler.greedy(),
)
out.text                       # "The first image shows a..."
out.input_token_count          # 2847 (text + vision tokens)
out.image_token_counts         # [1280, 1280] per image
out.hidden_states              # if requested, with image-token spans labeled
```

The same shape works for video (`{"type": "video", "video": ...}` accepting a path, an iterable of frames, or a `decord.VideoReader`), audio (`{"type": "audio", "audio": ...}` accepting a path, a numpy array, or a `(waveform, sample_rate)` tuple), and any combination thereof.

**What this gets right that vLLM gets wrong:**

- **No preprocessor flag confusion.** `mm_processor_kwargs`, `max_pixels`, `min_pixels`, `limit_mm_per_prompt`, image budget per request — all derived automatically from the model's config plus the actual input dimensions. If your images are ≤ 1280×768, we set `max_pixels=1280×28×28` automatically; we do not reserve KV cache for an imaginary 12-megapixel input. CaaS preflight catches the residual cases.
- **Image format normalization is automatic.** RGBA → RGB, grayscale → RGB, EXIF orientation honored, large images downscaled to the model's sweet spot before tokenization (with the original dimensions recorded in the manifest). The "image preprocessor expects RGB but gets RGBA" class of bug is impossible by construction.
- **`image_grid_thw` and equivalent indexing is the engine's problem, not yours.** Qwen2-VL/2.5-VL/3-VL families get this wrong in vLLM under multi-image prompts (#23371). We thread the grid metadata correctly per-request and assert the slice indices match the produced token count before forward.
- **Multi-image and interleaved prompts are first-class.** A message can contain any sequence of images and text segments; we pack them in the order given, record per-image token spans in the output, and don't silently merge or reorder.
- **Video is handled at the frame level, not magicked.** You provide an FPS budget or an explicit frame iterator; we sample, resize, and tokenize deterministically. The manifest records the exact frame indices used so a video eval reproduces.
- **Audio sample rates are checked, not assumed.** The `Tokenization` object asserts the input matches the model's expected rate (typically 16 kHz) and resamples explicitly with the resampler choice recorded in the manifest.

**Vision-token hidden states for interpretability:**

```python
from anvil import HiddenStateSpec

out = m.generate(
    messages=[...],
    capture=HiddenStateSpec(layers=[12, 24, -1], positions="image_tokens"),
)
out.hidden_states[12]          # shape [num_image_tokens, hidden_dim]
out.hidden_states_image_spans  # [(start, end)] per image in the prompt
```

`positions="image_tokens"` returns activations at vision-token positions only (with per-image spans), `positions="text_tokens"` at text positions, `positions="all"` for everything. This is the API vLLM RFC #33118 is still trying to land; ours works on day one for any VLM on the slow path, with fast-path optimizations as we add them.

**VLM fast-path coverage at v0:**

Qwen-VL family (Qwen-VL, Qwen2-VL, Qwen2.5-VL, Qwen3-VL), LLaVA-1.5/1.6, InternVL 2/2.5/3, Pixtral, Phi-3-Vision / Phi-3.5-Vision / Phi-4-Vision, Idefics 2/3, Molmo, CogVLM, MiniCPM-V 2.6 / 3.0. Anything not on this list — including the next VLM that drops tomorrow — runs via the slow path (`transformers.AutoModelForVision2Seq` or `AutoModelForImageTextToText`), the same day it's available in transformers main.

**VLM-aware CaaS preflight:**

The smoke test (§7.2) for VLM tasks runs three image samples spanning the dimension distribution: smallest, largest, and median by pixel count. This catches the "OOM at sample N because that one had a 4K image" class of failure that pure text smoke tests miss. The CaaS KB has dedicated VLM entries:

- `mm_processor_kwargs_default_too_high` (Qwen-VL family expects 12.8M-pixel budget) → set to actual max image dimensions
- `image_grid_thw_slice_mismatch` (multi-image Qwen2-VL) → ensure correct `image_grid_thw` metadata threading
- `vision_encoder_dtype_mismatch` (FP8 main model + BF16 vision encoder) → align dtypes per-component
- `limit_mm_per_prompt_zero` (silent default of 1 image when prompt has more) → raise to actual prompt's max
- `video_frame_count_explosion` (default 32 fps × 60s video = 1920 frames) → sane defaults with explicit override

**Multimodal evaluation tasks at v0 (Tier 1):**

MMMU, MathVista, ChartQA, DocVQA, MMBench, RealWorldQA. Each is manifest-locked with a per-model quality sentinel. Tier 2 imports of lm-evaluation-harness multimodal tasks work via the shim where the upstream task supports VLMs; bug-for-bug compatibility includes inheriting whatever bugs the upstream YAML has, marked `unverified` in the manifest as usual.

**Custom multimodal tasks (Tier 3):**

The same five-line registration as text, with whatever modality your model accepts:

```python
@anvil.register_task
class MyMedicalVQA(anvil.Task):
    name = "med_vqa_v1"
    dataset = "myorg/medical-vqa"

    def doc_to_request(self, doc):
        return anvil.Generate(messages=[{"role": "user", "content": [
            {"type": "image", "image": doc["xray"]},
            {"type": "text",  "text": doc["question"]},
        ]}])

    def request_to_prediction(self, response, doc):
        return response.text.strip().lower()

    metric = anvil.metrics.exact_match
```

That's it. The library handles tokenization, vision-encoder forward, KV cache, batching, manifest, preflight. You write the task, not the plumbing.

### 5.5 Quantization

Default fast paths supported: FP8, AWQ-INT4, GPTQ-INT4, compressed-tensors (the "datacenter" formats vLLM is hardening around). GGUF and bitsandbytes via the slow path only — vLLM is deprecating both (RFC #39583), and we follow that lead because the maintenance cost is real. For consumer-GPU GGUF workflows we explicitly recommend `llama.cpp` and provide an `anvil-engine-llamacpp` plugin spec for a future contributor.

---

## 6. Datasets and Benchmarks: Quick Integration

### 6.1 The user story

```python
import anvil

result = anvil.eval(
    model="Qwen/Qwen2.5-7B-Instruct",
    tasks=["mmlu", "gsm8k", "humaneval"],
    n_fewshot=5,
)
print(result.scores)          # {"mmlu": 0.745, "gsm8k": 0.832, "humaneval": 0.711}
result.manifest.save("run.json")
```

That's it. No 14 flags. No YAML files. No "is the chat template being applied?". The defaults are the right defaults.

For a custom dataset, five lines:

```python
from anvil import Task, register_task

register_task(Task(
    name="my_internal_eval",
    dataset="myorg/my-private-set",
    template="Question: {question}\nAnswer:",
    metric="exact_match",
))
result = anvil.eval(model="...", tasks=["my_internal_eval"])
```

### 6.2 Engine primitives, batched

The engine exposes a small set of primitives. All are batched. None have a per-doc Python loop.

For text models — the four lm-evaluation-harness-compatible primitives:

```python
engine.loglikelihood(requests: list[(context, continuation)]) -> list[(logprob, is_greedy)]
engine.loglikelihood_rolling(requests: list[str]) -> list[float]
engine.generate_until(requests: list[(prompt, until_strings)]) -> list[str]
engine.generate_logprobs(requests: list[prompt], top_k: int) -> list[Generation]
```

For non-generative or non-text models — the universal primitives:

```python
engine.embed(inputs: list[Any]) -> list[Tensor]
engine.classify(inputs: list[Any], label_set: list[str]) -> list[ClassificationResult]
engine.custom(fn: Callable[[Batch], Batch], inputs: list[Any]) -> list[Any]
```

`engine.custom` is the universal escape hatch: pass any callable that operates on a batch, get batched execution with the engine's host-device transfers, instrumentation, manifest integration, and CaaS preflight. This is what makes Anvil work for anything from RNA sequences to audio to graph inputs (§6.7).

Implementation details that matter:

- `loglikelihood` runs all (context, continuation) pairs in a single batched prefill, extracting logprobs at the right offsets in one shot. Our target on H100 80GB with Llama-3-8B: 50,000 MMLU pairs in <30 seconds. lm-evaluation-harness baseline: ~10 minutes.
- `generate_until` uses streaming logprob extraction so stop strings are detected mid-generation, not by post-hoc string search. No more "model rambled past the answer" silent failures.
- `embed` works against any encoder loaded via `transformers.AutoModel` and any custom encoder registered as a fast-path model.
- All primitives accept (where applicable) a `Sampler`, `LogitsProcessor` list, and `HiddenStateSpec`. The full research API is available inside an eval — including for non-text tasks.
- Manifest fields are conditional on request type: `Generate` requests pull in `Sampler` and `ChatTemplate`; `Embed` requests don't (they record pooling strategy and layer index instead); `Custom` requests record the callable's source hash and signature.

### 6.3 Task spec

A task is a Python class. YAML is a convenience layer that compiles to one.

```python
from anvil.tasks import Task, MultipleChoice, Generative
from anvil.metrics import exact_match, pass_at_k

class MMLU(MultipleChoice):
    name = "mmlu"
    dataset = "cais/mmlu"
    subjects = "all"  # or a list

    def doc_to_text(self, doc):
        return f"{doc['question']}\nA. {doc['choices'][0]}\nB. {doc['choices'][1]}\n..."

    def doc_to_choices(self, doc):
        return [" A", " B", " C", " D"]

    def doc_to_target(self, doc):
        return doc["answer"]   # 0-3

    fewshot_style = "interleaved"   # explicit, part of manifest
    metric = "accuracy"
```

### 6.4 Benchmark coverage strategy: three tiers

We do not promise to be a drop-in replacement for lm-evaluation-harness's full task catalog (~400 tasks). We promise something more useful: a curated tier we vouch for, a shim for everything else, and a clean path for anything you invent.

**Tier 1 — Anvil-curated** (~20 benchmarks at v0, growing). Each task is manifest-locked, has known-good baselines for ~10 reference models, ships with a quality sentinel, and has been audited end-to-end (chat template, fewshot style, extraction strategy, metric correctness). When you run an Anvil-curated benchmark and the score diverges from our published baseline by more than the configured threshold, the run aborts and CaaS engages — by design. The list is the table in §6.5.

**Tier 2 — lm-evaluation-harness import** (~400 tasks). The compatibility shim accepts existing `lm-eval` task YAMLs unchanged. The manifest tags imported tasks as `tier: unverified` and we make no claims about reproducibility — we also fix the obvious things (wrong fewshot, wrong filter, missing chat template) where we can detect them, and we record every such delta. This is a migration path, not a destination. Users are expected to graduate the tasks they care about to Tier 1, ideally by upstreaming a curated version.

**Tier 3 — user-defined** (any number, any modality). Five-line registration of any task that fits the `Task` abstraction. This is the path for non-text and non-VQA evaluations — RNA, protein, audio, graphs, embeddings, anything (§6.7).

The CLI exposes the tier explicitly:

```bash
anvil eval --model X --tasks mmlu,gsm8k                          # Tier 1
anvil eval --model X --lm-eval-tasks mmlu_pro,leaderboard_math   # Tier 2
anvil eval --model X --tasks myorg.rna_function_v1               # Tier 3
```

Even Tier 2 tasks run at engine speed — the shim compiles each YAML to an Anvil `Task` at load time and runs through the batched primitives, giving 10–40× speedups over lm-evaluation-harness on loglikelihood tasks. For migration validation, `anvil eval --compare-with-lm-eval` runs both engines against a task and produces a delta report so you can see exactly which deltas (chat template, sampler, fewshot style, metric) account for any score difference before trusting the migration.

### 6.5 Built-in benchmarks (v0)

Curated, tested, manifest-locked. We commit to reproducing the published numbers from the original benchmark paper for each model in our fast path.

| Category | Tasks |
|---|---|
| General reasoning | MMLU, MMLU-Pro, MMLU-Redux, BBH, ARC, AGIEval |
| Math | GSM8K, MATH, MATH-500, AIME 2024/2025, HMMT |
| Code | HumanEval, HumanEval+, MBPP, MBPP+, BigCodeBench, LiveCodeBench |
| Long-context | RULER, LongBench, NIAH variants |
| Instruction following | IFEval, MT-Bench (auto-judge optional), AlpacaEval 2.0 |
| Multimodal | MMMU, MathVista, ChartQA, DocVQA, MMBench, RealWorldQA |
| Reasoning models | GPQA Diamond, MMLU-Pro-Reasoning, ARC-AGI |
| Tool use | BFCL v3, ToolBench |
| Truthfulness / safety | TruthfulQA, BBQ, SimpleQA |

Each task ships with: a deterministic dataset SHA, a default `ChatTemplate`-compatible prompt, an extraction strategy (boxed-answer parsers for math, AST-based for code, etc.), a metric, and a known-good baseline number for ~10 models for sanity-checking.

### 6.6 Scoring sentinels

Before any major run, the smoke test runs a **quality sentinel**: a fixed-answer prompt where the right answer is a deterministic short string ("The capital of France is Paris."). Score divergence from the model's published baseline triggers CaaS engagement *even when the engine technically didn't crash*. This is the line of defense against "instruct model degraded to base-model quality because the chat template is missing." The sentinel is per-model and lives next to the model's fast-path implementation.

### 6.7 Custom tasks and arbitrary modalities

Anvil's `Task` abstraction is intentionally not tied to text. The four required methods accept and return arbitrary types. If you can write a function that takes a doc and produces a model input, and another that takes a model output and produces a score, you have an Anvil task — whether the doc is a chat prompt, an RNA sequence, an audio clip, a graph, or a row of tabular data with a tensor attached.

**Data sources, three modes:**

- **HuggingFace dataset:** `dataset="myorg/my-set"`, one line.
- **Local files:** `dataset=Path("./eval.jsonl")`, supports `.jsonl`, `.parquet`, `.csv`, `.arrow`.
- **Programmatic:** `dataset=lambda: yield from my_iterator()`. The dataset SHA is computed by hashing the materialized iteration, so custom iterators are reproducible if they are deterministic.

All three feed the same `Task` machinery.

**Beyond text — the RNA example:**

```python
import anvil
from transformers import AutoModel, AutoTokenizer
from scipy.stats import spearmanr

# Load a non-causal model. The slow path supports any AutoModel class.
model = anvil.load_custom(
    model_id="multimolecule/rnafm",
    model_class=AutoModel,
    tokenizer_class=AutoTokenizer,
)

@anvil.register_task
class RNAFunctionRegression(anvil.Task):
    name = "rna_function_v1"
    dataset = "myorg/rna-function-set"

    def doc_to_request(self, doc):
        return anvil.Embed(input=doc["sequence"], pool="mean", layer=-1)

    def request_to_prediction(self, response, doc):
        return response.embedding         # shape [hidden_dim]

    def aggregate(self, predictions, docs):
        targets = [d["activity"] for d in docs]
        preds   = [float(p @ self.regression_head) for p in predictions]
        return {"spearman": spearmanr(preds, targets).statistic}

result = anvil.eval(model=model, tasks=["rna_function_v1"])
result.scores      # {"rna_function_v1": {"spearman": 0.81}}
result.manifest.save("run.json")
```

**The full surface for arbitrary modalities:**

- **`doc`** is `Any`. A row in your dataset can be a sequence, a tensor, a graph, a dict, anything serializable.
- **Request types** are extensible: `Generate`, `LogLikelihood`, `Embed`, `Classify`, `Custom(callable)`. New request types are a plugin extension point.
- **Predictions** are `Any`. Vectors, classifications, structured outputs, dataframes.
- **Metrics** are `Callable[[predictions, targets], float | dict]`. We ship a baseline set — `exact_match`, `accuracy`, `pass_at_k`, `bleu`, `rouge`, `spearman`, `pearson`, `rmse`, `f1`, `auroc`, `mcc` — and yours plugs in with no wrapper.
- **Aggregation** is per-task. Default is mean; override `aggregate()` for rank correlations, top-k, weighted averages, anything.
- **Multi-input tasks** are supported: `doc_to_request` can return a `MultiInput` containing several inputs that the model batches together (sequence + structure + ligand, image + audio, text + table).

**What this enables, beyond chat:**

- Protein / RNA / DNA function prediction (ESM-2/3, RNA-FM, NT, ProGen)
- Audio classification (Whisper-encoder embeddings, AST, EnCodec)
- Custom-encoder image classification (DINOv2, CLIP, SigLIP) with linear probes
- Embedding-similarity benchmarks (MTEB-style with your own embeddings)
- Linear probes / regression heads on hidden states (interpretability)
- Multi-modal fusion (sequence + structure + image)
- Anything else where you have a model, an iterator over docs, and a metric

**Caveats and honest limits:**

- Anvil's *fast path* is text-LLM-shaped. Non-text models go through the slow path (`transformers.AutoModel` + FlashAttention if applicable + batched execution + manifest + CaaS preflight). Throughput tracks the underlying model; Anvil's overhead is minimal but we don't accelerate kernels we didn't write.
- Some abstractions become irrelevant for non-chat tasks. A `ChatTemplate` for RNA makes no sense, so `Embed` requests don't carry one and the manifest reflects that. The `Sampler` doesn't apply to embeddings.
- The CaaS rule engine and KB are LLM-focused at v0. For non-text tasks, CaaS still catches the universal failures (dtype, OOM, dependency mismatches, model loading) but the modality-specific KB grows over time, contributor-driven.

The goal is simple: **anyone with a model and a dataset should not have to glue together their own batching, host-device transfers, manifest, and reproducibility envelope.** Anvil handles all of that for any task that fits the Task interface, regardless of what the data is.

---

## 7. CaaS: Preflight Self-Healing Agent

### 7.1 Why preflight, not reactive

Most failure modes in inference evaluation are silent. They don't crash; they produce wrong numbers. A reactive agent that only fires on tracebacks misses the worst class of bug. CaaS runs a smoke test *before* the major run, comparing both `did it crash?` and `did it produce sensible output?`, and only releases the run when both pass.

### 7.2 The smoke test design

Three samples, run before any major eval:

1. **Canonical short.** Fixed prompt, fixed expected token count. ~500 ms. Detects chat-template, EOS, config-load failures, and identifies the model.
2. **Longest input in the eval set.** Discovered by tokenizing the dataset upfront. Catches long-context OOM, KV-cache overflow, max-model-len truncation. This is the difference between "OOM at sample 12,000" and "OOM at sample 0".
3. **Random middle.** One full-generation sample from the eval set's middle. Catches task-specific surprises.

Plus a **quality sentinel** (§6.6): a known-answer prompt. If the score diverges from the model's published baseline by more than a configured threshold, CaaS engages even without an exception.

### 7.3 v0: rules + curated knowledge base, no LLM

The dominant value comes from a hand-curated **known-issue database** — a YAML file with ~150 entries, each keyed on a regex error signature, mapping to a deterministic fix.

```yaml
- id: vllm_chat_template_required_v044
  engines: [vllm>=0.5,<0.20; transformers>=4.44]
  signatures:
    - "default chat template is no longer allowed"
  category: chat_template
  severity: blocking
  fix:
    type: set_engine_flag
    flag: "--chat-template"
    value_strategy: lookup_from_model_card
  references: [HF paligemma-3b discussion #5]
  human_message: |
    Your model does not provide a chat template and transformers v4.44+
    rejects the default. Anvil will look up a compatible template from
    examples/chat_templates/ based on the model family.

- id: tp_attn_heads_divisibility
  engines: [vllm>=0.4]
  signatures:
    - "Total number of attention heads (\\d+) must be divisible by tensor parallel size (\\d+)"
  category: parallelism
  fix:
    type: set_engine_flag
    flag: "--tensor-parallel-size"
    value_strategy: largest_divisor_of_attn_heads
  references: [vllm #414, #596, #1041, #2652, #4232, #5003, #11797]

- id: llama3_eot_runaway
  signatures:
    - "model_id_matches: meta-llama/Meta-Llama-3-.*-Instruct"
    - "no eos_token_id triggered"
  fix:
    type: set_sampling_param
    param: stop_token_ids
    value: [128009]
  references: [vllm #4180, #4297, #5395]
```

The KB ships in the repo at `anvil/caas/known_issues/*.yaml`, organized by category. Initial v0 size: ~150 entries covering the categories tabulated in the gap analysis (install ~30, model loading ~25, memory ~30, tokenization ~25, sampler ~10, harness ~20, multimodal ~10, tool calling ~10). Each entry has an `expires-after` field so the KB rots gracefully as engines evolve.

A **rule engine** (~50 LOC of pure Python) handles the dead-deterministic cases — TP divisibility, dtype-on-Volta, `trust_remote_code` detection, `mm_processor_kwargs` defaults — without consulting the KB at all.

### 7.4 v1: small coder LLM as fallback

When rule engine and KB both miss, CaaS escalates to a small coder model:

- **Default:** Qwen2.5-Coder-7B-Instruct INT4 (~5 GB VRAM, Apache 2.0, HumanEval 88.4, Aider edit benchmark dominant).
- **Alternative:** Granite-4.1-8B FP8 (~5 GB, Apache 2.0, BFCL v3 68.27, production-tuned).
- **Tight VRAM:** Granite-4.1-3B-Instruct or Qwen2.5-Coder-3B-Instruct INT4 (~2 GB), with documented degradation in fix quality.
- User-overridable.

The model is loaded **lazily** — never present unless rule + KB both miss. On a clean preflight, CaaS adds zero VRAM and ~50 ms latency. On an engaged preflight, ~3.5 s for a 7B INT4 fix proposal, ~5 s for re-smoke-test, total ~10–15 s per retry, hard cap 3 retries.

### 7.5 The action space is tiny and typed

The coder model emits one of these via grammar-constrained decoding (xgrammar):

```
set_engine_flag(name: enum[<allow-list>], value: str)
unset_engine_flag(name: enum[<allow-list>])
set_env_var(name: enum[<allow-list>], value: str)
edit_yaml(path: str <within configs/>, key: str, value: scalar)
install_package(name: str, version: str)   # gated behind --ci --allow-install
restart_engine()
give_up(reason: str)
```

The flag allow-list is generated at runtime by parsing `vllm serve --help` (or the equivalent for the active engine), so the LLM cannot hallucinate flags that don't exist. Flags not on the parsed list are rejected without retry; the LLM is re-prompted with the actual help output.

### 7.6 Modes

- `--caas=off`: disabled.
- `--caas=advisory`: print the proposed diff, never apply.
- `--caas=research` (default): present a unified diff, ask `[y/n/edit/explain]`.
- `--caas=ci`: auto-apply if `(source ∈ {rule_engine, kb}) AND (severity ≠ "review-required") AND (action ∈ ci-allow-list) AND (confidence > 0.7)`. Otherwise abort the run with exit code != 0.

Any `install_package` action requires `--ci --allow-install`. Any `set_engine_flag` involving `--trust-remote-code` requires explicit human confirmation in any mode, because it is a code-execution surface.

### 7.7 Safety rails — what CaaS must never modify

- The eval task YAML or any dataset file.
- Ground-truth labels, prompts, fewshot examples.
- Code outside `configs/` and `engine/launch_args.json`.
- Model weights or tokenizer files.
- `requirements.txt` / `pyproject.toml` outside `--ci` mode.
- `.git/`.
- System files (`~/.bashrc`, system `pip`, `LD_PRELOAD`, `PATH`).

The env-var allow-list is fixed: `CUDA_VISIBLE_DEVICES`, `HF_HOME`, `HF_TOKEN`, `NCCL_*`, `VLLM_*`, `TORCH_*`, `TOKENIZERS_PARALLELISM`, `OMP_NUM_THREADS`. Anything else is rejected.

### 7.8 What CaaS will *not* try to fix

These get surfaced verbatim with diagnostics, never auto-applied:

- Driver-too-old errors (`libcudart` / NCCL init / sm_120 on consumer cards): hardware-side, requires user action.
- Bumping `gpu_memory_utilization > 0.92`: trades silent OOM at long contexts.
- Lowering `max_model_len` below `max(input_token_lengths) + max_output_tokens`: silently truncates the long inputs.
- Auto-applying `--enforce-eager` to silence a CUDA-graph crash: 3× throughput regression, surfaces instead.
- Pinning `transformers` to git main to load a new architecture: masks an upstream incompatibility.
- Downgrading any major dep to silence an import error: surfaces as a versioning conflict.

### 7.9 Audit log → manifest

Every CaaS action emits an immutable JSONL record:

```json
{
  "ts": "2026-05-08T14:23:11.482Z", "step": 2,
  "trigger": "ValueError: Total number of attention heads...",
  "match_source": "rule_engine",
  "action": "set_engine_flag",
  "args": {"name": "--tensor-parallel-size", "value": "4"},
  "rationale": "num_attention_heads=64 not divisible by tp_size=3; largest divisor ≤ available_gpus=4 is 4",
  "confidence": 1.0,
  "kb_entry_ids": ["tp_attn_heads_divisibility"],
  "previous_value": "3",
  "validator": "smoke_test_3sample",
  "validator_result": "pass",
  "llm_tokens_in": 0, "llm_tokens_out": 0
}
```

The full log is embedded in the run's `Manifest` (§8). A reviewer can see every CaaS-applied delta and either accept or reject the run. `anvil manifest reject-caas <run.json>` produces a frozen-config rerun.

### 7.10 Acceptance criterion

On a curated test corpus of 50 reproduced GitHub issues from the vLLM and lm-eval issue trackers, CaaS v0 must:

- Auto-resolve ≥ 60%.
- Surface clearly with correct guidance on ≥ 30%.
- False-positive rate ≤ 2% (a "fix" that introduces a new error or masks a real bug).
- p95 wall-clock overhead on engaged preflight ≤ 60 s.
- Zero VRAM overhead on clean preflight.

---

## 8. The Reproducibility Manifest

The manifest is the primary output of an evaluation run. It is sealed at end-of-run and signed (sha256 of canonical JSON).

### 8.1 Fields

```json
{
  "anvil_version": "0.4.2",
  "engine": {"name": "vllm", "version": "0.20.1", "backend_hash": "sha256:..."},
  "model": {
    "id": "Qwen/Qwen2.5-7B-Instruct",
    "revision": "sha256:abc...",
    "dtype": "bfloat16",
    "quantization": null,
    "config_hash": "sha256:..."
  },
  "tokenization": {
    "hash": "sha256:...",
    "bos_handling": "auto-prepend",
    "eos_token_ids": [151645, 151643],
    "padding_side": "left"
  },
  "chat_template": {
    "hash": "sha256:...",
    "version": "qwen2.5-instruct@v1",
    "source": "tokenizer_config.json",
    "fewshot_style": "interleaved"
  },
  "sampler": {
    "hash": "sha256:...",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 2048,
    "seed": 42,
    "source": "anvil.Sampler.greedy"
  },
  "tasks": [
    {
      "name": "mmlu",
      "version": "anvil@v3",
      "dataset_revision": "sha256:...",
      "n_fewshot": 5,
      "metric": "accuracy"
    }
  ],
  "scores": {"mmlu": 0.7451},
  "smoke_test": {
    "samples": 3,
    "sentinel_score": 1.0,
    "outcome": "pass"
  },
  "caas_log": [...],
  "hardware": {
    "gpus": ["NVIDIA H100 80GB HBM3 x4"],
    "cuda": "13.0", "driver": "580.82.07",
    "torch": "2.8.0+cu130"
  },
  "started_at": "2026-05-08T14:23:11.482Z",
  "ended_at": "2026-05-08T14:31:42.193Z",
  "manifest_signature": "sha256:..."
}
```

### 8.2 Modes

- **Frozen mode** (default for benchmark uploads): if CaaS would have to modify any user-supplied config, the run aborts. The manifest is bit-identical between machines.
- **Self-healing mode** (default for research iteration): CaaS may apply fixes; every fix is logged. The manifest records the original *and* effective configs.

### 8.3 Tooling

```bash
anvil manifest verify run.json              # checks signature, recomputes hashes
anvil manifest diff a.json b.json           # explains every delta that could affect scores
anvil manifest replay run.json              # deterministic rerun from a manifest
anvil manifest strip-caas run.json          # produces a frozen-config rerun spec
```

`anvil manifest diff` is the tool that, when two runs disagree by 12 points, points at the chat-template hash difference and tells you which one was the bug. This is the answer to the lm-evaluation-harness reproducibility crisis.

---

## 9. Installation and Compatibility

### 9.1 The `uv` story

```bash
uv pip install anvil
```

Works against any torch ≥ 2.4 with CUDA ≥ 12.1, on Linux x86_64 and ARM64. Wheels are built per (torch_minor × cuda_major) combination, mirroring PyTorch's wheel index:

```
anvil-0.4.0-cp311-linux_x86_64-cu121.whl
anvil-0.4.0-cp311-linux_x86_64-cu124.whl
anvil-0.4.0-cp311-linux_x86_64-cu126.whl
anvil-0.4.0-cp311-linux_x86_64-cu128.whl
anvil-0.4.0-cp311-linux_x86_64-cu130.whl
anvil-0.4.0-cp311-linux_aarch64-cu130.whl
anvil-0.4.0-cp311-cpu.whl                   # CPU fallback
```

ABI is detected at install time. If the user's torch combination is not covered, install falls back to a **pure-Python build** that uses the slow path only — slower but always works. We refuse to fail-closed on installation.

### 9.2 Hardware support

- **NVIDIA:** Volta (limited, FP16 only), Turing, Ampere, Ada, Hopper, Blackwell. GH200/DGX Spark first-class via ARM64 wheels.
- **AMD:** ROCm 6.x via `anvil[rocm]` extra. Slow-path always works; fast-path coverage tracks vLLM ROCm support.
- **Apple Silicon:** MLX backend via `anvil-engine-mlx` plugin (M1–M5). Slow path works on Metal via PyTorch MPS.
- **Intel:** XPU via `anvil[xpu]`, slow path only in v0.

### 9.3 Container

`ghcr.io/anvil/anvil:<version>-cu130` ships a known-good environment. Image size target: < 4 GB compressed. Multi-arch (linux/amd64, linux/arm64).

---

## 10. Public API Surface

### 10.1 Python (`anvil.eval`)

```python
import anvil

# Simplest case
result = anvil.eval(model="Qwen/Qwen2.5-7B-Instruct", tasks=["mmlu"])

# Research-mode with custom decoding and hidden-state capture
result = anvil.eval(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tasks=["gsm8k", "math"],
    sampler=anvil.Sampler(temperature=0.0, max_tokens=4096),
    logits_processors=[anvil.research.DoLa()],
    capture=anvil.HiddenStateSpec(layers=[-1], positions="last"),
    n_fewshot=4,
    chat_template="auto",         # or a ChatTemplate object
    caas="research",
    output_dir="./runs/exp_001/",
)
```

### 10.2 Python (`anvil.serve`)

```python
import anvil

server = anvil.serve(
    model="Qwen/Qwen2.5-7B-Instruct",
    port=8000,
    tool_calling="auto",          # constrained-decoding-based, no per-model parser flag
    structured_output="xgrammar", # or "outlines", "lmfe"
)
```

The serve API is OpenAI-compatible. Tool calling uses constrained decoding under `tool_choice="auto"`, with `strict=true` honored by construction. There are no `--tool-call-parser` flags. The 14 model-specific parsers vLLM ships with are replaced by one grammar.

### 10.3 CLI

```bash
# Eval
anvil eval --model Qwen/Qwen2.5-7B-Instruct --tasks mmlu,gsm8k --output ./run.json

# Serve
anvil serve --model Qwen/Qwen2.5-7B-Instruct --port 8000

# Migrate from lm-evaluation-harness
anvil eval --model ... --lm-eval-tasks mmlu_pro,arc_challenge --compare-with-lm-eval

# Manifest tools
anvil manifest verify run.json
anvil manifest diff a.json b.json
anvil manifest replay run.json

# CaaS tools
anvil caas test ./failing-config.yaml          # dry-run CaaS against a known failure
anvil caas list-known-issues                   # browse the KB

# Plumbing
anvil doctor                                   # diagnose install / env / GPU issues
anvil cache prune                              # clean weight cache
```

The CLI is a thin wrapper over the Python API. Every flag has a Python-API equivalent. No flag exists *only* in the CLI.

---

## 11. Roadmap

### 11.1 v0 — 12–16 weeks. The Smallest Viable Library

Goal: a single researcher can run MMLU on Qwen2.5-7B-Instruct in 5 minutes, reproduce it on a colleague's machine, and trust the number.

- Engine layer: `anvil-engine-vllm` wrapper + `anvil-engine-hf` slow path
- Six core abstractions (`ChatTemplate`, `Tokenization`, `Sampler`, `LogitsProcessor`, `HiddenStateSpec`, `Manifest`)
- Fast path: 12 architectures (Llama 3, Qwen 2.5, Qwen 3, DeepSeek V3, Mistral, Mixtral, Gemma 3, Phi 4, Yi, Command-R, GLM-4, Qwen-VL)
- Slow path: anything in transformers main
- 4 batched eval primitives
- 6 built-in benchmarks: MMLU, GSM8K, MATH-500, HumanEval, IFEval, MMMU
- lm-evaluation-harness compatibility shim
- Manifest + signing + verify/diff/replay
- CaaS v0: rules + 150-entry KB, no LLM, smoke-test design
- uv-installable wheels: cu121/124/128/130
- OpenAI-compatible serve with constrained-decoding tool calls
- Documentation written for a researcher

**Acceptance:** reproduce Open LLM Leaderboard 2 numbers for 5 reference models within 0.5 percentage points; 60% CaaS auto-resolve rate on the 50-issue test corpus; install succeeds on RTX 5090, A100, H100, MI300X, GH200.

### 11.2 v0.5 — +2 months. Eval Correctness

- Built-in benchmarks expanded to 20 (full table from §6.5)
- Quality sentinel per fast-path model
- `anvil manifest compare-with-lm-eval` and `compare-with-lighteval`
- Reasoning-model support (DeepSeek-R1, QwQ, o1-style): `<think>` extraction, tag-aware metrics
- Long-context-aware preflight (5 samples: short / median / P95 / max / random)

### 11.3 v1 — +6 months. CaaS LLM Tier and Engine Plurality

- CaaS coder LLM tier with Qwen2.5-Coder-7B INT4 default; Granite-4.1-8B alternative
- CaaS modes: ci, advisory, full audit
- KB auto-update workflow (user opts in; PR-style flow)
- `anvil-engine-sglang` plugin (engine-portable manifest: same Sampler+ChatTemplate produces same scores across engines, or the diff is recorded)
- Multimodal-first: clean `list[PIL.Image]` API across all VLM fast-path architectures
- Apple Silicon MLX plugin
- Full plugin protocol v1 (semver-stable)

### 11.4 v2 — +12 months. Deeper Research Hooks

- Activation hooks on arbitrary modules (interpretability researcher's kit)
- Custom attention pattern injection
- Streaming intermediate state inspection
- Active learning: anonymized error→fix→outcome traces (opt-in) fine-tune a CaaS-specialized 7B
- **InferenceFix-Bench:** a published benchmark of ~500 reproduced inference-pipeline failures
- AST-aware light-touch user-code edits in `--caas=expand-scope`

### 11.5 What we will not build

- Distributed serving / autoscaling. Use vLLM or SGLang; we have a serve API for development and small-scale eval, not for production traffic.
- Custom attention kernels. We use the engine's.
- A new quantization format. We support what the ecosystem ships.
- A model registry / catalog UI. HuggingFace exists.

---

## 12. Non-Goals and Anti-Goals

We will be tempted, repeatedly, to expand. These are the lines we hold:

- **We are not vLLM at scale.** No P/D disaggregation, no expert parallelism, no large-scale serving optimizations. vLLM is better at this and has the engineers to keep being better.
- **We are not a serving framework.** The serve API exists for development and small-scale evaluation. If you need to serve 10,000 users, deploy vLLM or SGLang behind it.
- **We are not a model zoo.** We don't host weights. HuggingFace does.
- **We are not a fine-tuning library.** Use TRL, axolotl, unsloth.
- **We are not a "single command does everything" agent.** CaaS fixes silly errors; it does not write your eval task or interpret your results.
- **We do not chase throughput benchmark wins.** Anvil's identity is correctness and reproducibility. If we are 25% slower than vLLM bare-metal on H100 8B Llama-3, that is acceptable. If we are 250% slower, that is a bug.

---

## 13. Risks and Mitigations

### 13.1 vLLM closes the gap

vLLM is moving fast. RFCs #33118 (hidden states), #36998 (observation plugin), #13360 (logits processors), #19161 (plugin architecture) all aim at the gaps Anvil exploits. If vLLM ships these in 2026–2027 with stable APIs, our research-first wedge narrows.

**Mitigation:** our wedge is not just the features — it's the **abstractions and the manifest**. Even if vLLM ships hidden-state extraction, they will not ship `ChatTemplate.hash`, `Sampler.diff`, or `manifest verify`. The reproducibility envelope is a different product.

### 13.2 SGLang positioning

SGLang has a stronger day-zero culture and 25–30% throughput edge on H100 prefix-heavy workloads. A user might choose SGLang directly.

**Mitigation:** use SGLang as a backend. `anvil-engine-sglang` makes this trivial. Anvil's value is upstream of any specific engine.

### 13.3 Adoption — researchers are conservative

Researchers running MMLU on a Tuesday will not switch libraries unless the migration cost is near zero.

**Mitigation:** the lm-evaluation-harness compatibility shim. Day-zero, an existing lm-eval workflow runs through Anvil with one flag change and produces a manifest. Show, don't tell.

### 13.4 KB rot

The CaaS known-issue database degrades as engines evolve. An entry that was right for vLLM 0.10 may be wrong for 0.20.

**Mitigation:** every KB entry has an `engines:` version constraint; CaaS only consults entries whose constraint matches the active engine version. Quarterly KB review with bulk import from issue trackers. User-submitted entries (with citations) PR-reviewed.

### 13.5 CaaS masks real bugs

The existential failure mode. Auto-pinning torch to silence a CUDA error masks a hardware compatibility problem. Adding `--trust-remote-code` is a code-execution surface.

**Mitigation:** the §7.7 / §7.8 hard rules. `--caas=research` (diff-and-confirm) is the default. CI mode requires explicit allow-list opt-in. Every action is in the manifest. Specific high-risk actions (`install_package`, `--trust-remote-code`) require explicit user approval in any mode.

### 13.6 Maintenance burden

A library that wraps vLLM, SGLang, and HF transformers, and ships 30 model fast paths, and maintains a KB — is a lot of code. With one or two maintainers, this is not sustainable.

**Mitigation:** the slow-path-by-default architecture is the maintenance load-bearer. Every architecture *not* on the fast path is automatically supported via transformers. We commit to fast-path coverage only for the top ~30 — the long tail is the slow path forever, and that's fine. The KB is community-curated PRs after v1. The engine plugins are explicitly user-extensible.

### 13.7 Engine API drift

vLLM breaks something every six weeks. SGLang has install pain. We pin to known-good versions and our wrapper layer becomes fragile.

**Mitigation:** the wrapper layer is *thicker* than a typical wrapper precisely because of this. We re-implement abstractions that vLLM dropped (per-request logits processors, hidden states) above the engine, so we are robust to engine churn. We pin to the last-known-good vLLM minor version per Anvil release and run our smoke-test corpus against engine upgrades before bumping.

### 13.8 Scope creep into a generic agent framework

The instinct to expand CaaS into a full coding agent is strong. An expanded-scope agent that touches user code is a different product with a much larger surface area.

**Mitigation:** v0/v1/v2 explicitly scope CaaS to configs, env, and launch flags. Any user-code edits land behind `--caas=expand-scope`, which is opt-in and gated for v2+. Resist.

---

## 14. Open Questions

These are decisions we have not made and want feedback on before v0:

1. **License.** Apache 2.0 is the default for ecosystem fit. AGPL would protect against cloud-rehost. Probably Apache.
2. **Built-in tasks v0.** Six is the proposal. Argument for fewer (3): test the migration story harder. Argument for more (12): bigger immediate value to early adopters.
3. **CaaS LLM in v0 or v1.** Currently v1. Argument for v0: differentiation. Argument for v1 (current call): rule + KB delivers most of the value with none of the safety risk.
4. **Engine choice for the default fast path.** vLLM is the safe pick. SGLang has a throughput edge but more install pain. The current call is "vLLM default, SGLang plugin" — but if SGLang's install story improves, swap.
5. **Naming.** §15.
6. **Governance.** Single maintainer for v0 → community for v1. The "vLLM Foundation" model (broad sponsorship) is appealing but premature.
7. **The `Manifest` signing key.** Users sign their own runs with a per-org key. We do *not* run a central signing service. We do publish a verifier.

---

## 15. Naming

The library does not have a name yet. Candidates:

- **Anvil** (current placeholder). Connotes precision, building, reshaping; short; .com taken; PyPI free. Slight collision with anvil.works (Python web framework) but different domain.
- **Forge.** Same connotations; .com aggressively taken; PyPI taken.
- **Crucible.** Distinct, evocative of testing; longer.
- **Helix.** Suggests precision/structure; PyPI free; nothing in our space.
- **Caliper.** Measurement instrument; precise; underused.
- **Probe.** Direct; short; possibly too generic.
- **Bench.** Honest; possibly too generic.
- **Plumb.** Implies depth and precision; underused.
- **Ledger.** Reproducibility-first connotation; potentially confusing with finance.

Recommendation: **Anvil** as default working name. Final naming should be a function of: (a) PyPI / GitHub availability, (b) one-syllable preferred, (c) connotes precision or correctness, not speed. Open to replacement.

---

## 16. Build Specification

The preceding sections explain *why* and *what for*. This section is the implementation contract: repo layout, type signatures, dependency pins, conventions, and a milestone-ordered build plan with acceptance tests. A Claude Code agent reading this section together with §1–§15 should be able to produce v0 without making further design decisions.

The principle: every claim made in §1–§15 has either an acceptance test in §16.10 or a concrete artifact specified in §16.1–§16.9. If something is unclear, the design manuscript wins; this section is meant to be a faithful translation, not a divergence.

### 16.1 Repo layout

```
anvil/
├── pyproject.toml
├── README.md
├── LICENSE                          # Apache 2.0
├── .pre-commit-config.yaml
├── .github/workflows/               # ci.yml, wheels.yml, release.yml
├── docs/
│   ├── design.md                    # this manuscript
│   ├── tutorials/
│   ├── api/
│   └── caas_kb/                     # rendered KB docs
├── src/anvil/
│   ├── __init__.py                  # public API surface (see §16.3)
│   ├── _version.py
│   ├── config.py                    # global Config object, env-var binding
│   ├── logging.py                   # structured logging setup
│   ├── exceptions.py                # error hierarchy
│   │
│   ├── primitives/                  # the typed objects users touch
│   │   ├── __init__.py
│   │   ├── chat_template.py
│   │   ├── tokenization.py
│   │   ├── sampler.py
│   │   ├── logits_processor.py
│   │   ├── hidden_state_spec.py
│   │   ├── request.py               # Generate / LogLikelihood / Embed / Classify / Custom
│   │   └── response.py
│   │
│   ├── engine/                      # private; only public.py is re-exported
│   │   ├── __init__.py
│   │   ├── public.py                # Engine protocol
│   │   ├── factory.py               # picks backend per (model, hardware)
│   │   ├── _vllm/
│   │   │   ├── adapter.py           # vLLM → Engine wrapper
│   │   │   ├── version_compat.py    # CLI flag normalization per vLLM minor
│   │   │   └── logits_proxy.py      # restores per-request LP API
│   │   ├── _hf/
│   │   │   └── runner.py            # transformers slow path
│   │   └── _wrappers/
│   │       ├── logits_processor.py
│   │       └── hidden_state.py
│   │
│   ├── models/
│   │   ├── __init__.py              # public: load, load_custom
│   │   ├── registry.py              # fast-path registration
│   │   ├── _fast/                   # llama.py, qwen.py, qwen_vl.py, mistral.py, gemma.py, phi.py
│   │   └── _slow/
│   │       ├── transformers_text.py
│   │       └── transformers_mm.py
│   │
│   ├── tasks/
│   │   ├── __init__.py              # public: eval, Task, register_task
│   │   ├── base.py                  # Task ABC
│   │   ├── registry.py
│   │   ├── runner.py                # batched eval loop
│   │   ├── lm_eval_shim/
│   │   │   ├── compiler.py
│   │   │   └── compat.py
│   │   └── builtin/
│   │       ├── mmlu.py
│   │       ├── gsm8k.py
│   │       ├── humaneval.py
│   │       ├── ifeval.py
│   │       ├── math500.py
│   │       └── mmmu.py
│   │
│   ├── manifest/
│   │   ├── __init__.py
│   │   ├── schema.py                # Pydantic v2 models
│   │   ├── canonical.py             # canonical JSON serialization
│   │   ├── sign.py
│   │   ├── verify.py
│   │   ├── diff.py
│   │   └── replay.py
│   │
│   ├── caas/
│   │   ├── __init__.py
│   │   ├── preflight.py             # smoke-test orchestrator
│   │   ├── rule_engine.py
│   │   ├── kb/
│   │   │   ├── loader.py
│   │   │   ├── schema.py
│   │   │   └── entries/
│   │   │       ├── install.yaml
│   │   │       ├── model_loading.yaml
│   │   │       ├── memory.yaml
│   │   │       ├── tokenization.yaml
│   │   │       ├── sampler.yaml
│   │   │       ├── multimodal.yaml
│   │   │       ├── tool_calling.yaml
│   │   │       ├── parallelism.yaml
│   │   │       └── harness.yaml
│   │   ├── llm_tier.py              # v1; v0 stub raises NotImplementedError
│   │   ├── actions.py               # typed action allow-list
│   │   ├── audit.py                 # JSONL log
│   │   └── sentinel.py              # quality sentinel
│   │
│   ├── metrics/
│   │   ├── exact_match.py
│   │   ├── pass_at_k.py
│   │   ├── correlation.py           # spearman, pearson
│   │   ├── classification.py        # accuracy, f1, auroc, mcc
│   │   └── regression.py            # rmse, mae
│   │
│   ├── datasets/
│   │   └── loader.py
│   │
│   ├── plugins/
│   │   └── v1.py                    # versioned protocol
│   │
│   ├── server/
│   │   ├── app.py                   # FastAPI
│   │   ├── routes.py                # OpenAI-compatible
│   │   └── tool_calling.py          # constrained-decoding implementation
│   │
│   ├── cli/
│   │   ├── main.py                  # typer entrypoint
│   │   ├── eval.py
│   │   ├── serve.py
│   │   ├── manifest.py
│   │   ├── caas.py
│   │   └── doctor.py
│   │
│   └── research/
│       ├── __init__.py
│       └── dola.py                  # example LogitsProcessor
│
└── tests/
    ├── unit/
    ├── integration/
    │   ├── test_milestone_0.py
    │   ├── test_milestone_1.py
    │   ├── test_milestone_2.py
    │   └── ...
    ├── corpus/
    │   ├── caas_test_cases.yaml
    │   └── manifest_fixtures/
    └── conftest.py
```

### 16.2 Module import graph (CI-enforced)

The package enforces a one-way import graph at CI time using `import-linter`. Configuration in `pyproject.toml`:

```toml
[tool.importlinter]
root_packages = ["anvil"]

[[tool.importlinter.contracts]]
name = "Layered architecture"
type = "layers"
layers = [
    "anvil.cli | anvil.server",
    "anvil.tasks",
    "anvil.models",
    "anvil.engine",
    "anvil.primitives",
]

[[tool.importlinter.contracts]]
name = "primitives is a leaf"
type = "forbidden"
source_modules = ["anvil.primitives"]
forbidden_modules = [
    "anvil.engine", "anvil.models", "anvil.tasks",
    "anvil.cli", "anvil.server", "anvil.manifest", "anvil.caas",
]
```

`anvil.manifest`, `anvil.caas`, and `anvil.metrics` are cross-cutting and may import from `primitives` only. Plugins must not import any `anvil.*` path other than `anvil.plugins.v1`. **Violations block CI.** This is what makes principle §2.5 a build-time invariant rather than a code-review aspiration.

### 16.3 Public API surface (`anvil/__init__.py`)

The top-level `anvil` package re-exports exactly:

```python
from anvil.models import load, load_custom
from anvil.tasks import eval, Task, register_task
from anvil.server import serve

from anvil.primitives.chat_template import ChatTemplate
from anvil.primitives.tokenization import Tokenization
from anvil.primitives.sampler import Sampler
from anvil.primitives.logits_processor import LogitsProcessor
from anvil.primitives.hidden_state_spec import HiddenStateSpec
from anvil.primitives.request import Generate, LogLikelihood, Embed, Classify, Custom
from anvil.primitives.response import Response, Generation, EmbedResult, ClassifyResult

from anvil.manifest.schema import Manifest
from anvil.metrics.exact_match import exact_match
from anvil.metrics.pass_at_k import pass_at_k

from anvil import research

__version__ = "0.0.1"
__all__ = [
    "load", "load_custom", "eval", "Task", "register_task", "serve",
    "ChatTemplate", "Tokenization", "Sampler", "LogitsProcessor", "HiddenStateSpec",
    "Generate", "LogLikelihood", "Embed", "Classify", "Custom",
    "Response", "Generation", "EmbedResult", "ClassifyResult",
    "Manifest", "exact_match", "pass_at_k", "research",
]
```

Anything not in `__all__` is private and may break in any release.

### 16.4 Type signatures of the core abstractions

These are the source of truth. Field names and types match §4 of the design exactly.

```python
# src/anvil/primitives/chat_template.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

FewshotStyle = Literal["interleaved", "concat-system", "raw", "none"]

@dataclass(frozen=True, slots=True)
class ChatTemplate:
    """Versioned, content-hashed chat template.

    Two ChatTemplates with the same .hash MUST produce identical
    prompts for identical inputs.
    """
    jinja_source: str
    fewshot_style: FewshotStyle = "interleaved"
    name: str = "anonymous"
    source: str = "user-supplied"
    # source ∈ {"tokenizer_config.json", "chat_template.json",
    #           "user-supplied", "builtin", "model-card-override"}

    @classmethod
    def from_model(cls, model_id: str, *, revision: str | None = None) -> Self: ...

    @classmethod
    def from_jinja_file(cls, path: str | Path, *,
                        fewshot_style: FewshotStyle = "interleaved") -> Self: ...

    @property
    def hash(self) -> str:
        """sha256(canonicalize(jinja_source) + '\\n' + fewshot_style)"""

    @property
    def version(self) -> str:
        """Human-readable, e.g., 'qwen2.5-instruct@v1'."""

    def canonicalize(self) -> str:
        """Whitespace normalized, conditional branches sorted, no semantic change."""

    def render(self, messages: list[dict], *,
               add_generation_prompt: bool = True,
               tools: list[dict] | None = None) -> str:
        """Render the template; asserts no double-BOS."""

    def to_manifest_field(self) -> dict:
        """Returns: {hash, version, source, fewshot_style}."""
```

```python
# src/anvil/primitives/tokenization.py
from typing import Literal, Self
from transformers import PreTrainedTokenizerBase

BOSHandling = Literal["auto-prepend", "never", "from-template"]
PaddingSide = Literal["left", "right"]

@dataclass(frozen=True, slots=True)
class Tokenization:
    tokenizer: PreTrainedTokenizerBase
    bos_handling: BOSHandling = "from-template"
    eos_token_ids: tuple[int, ...] = ()  # ALL candidates (Llama-3 has multiple)
    padding_side: PaddingSide = "left"
    add_special_tokens: bool = False
    assert_no_double_bos: bool = True

    @classmethod
    def from_model(cls, model_id: str, *, revision: str | None = None) -> Self: ...

    @property
    def hash(self) -> str: ...

    def encode(self, text: str, *,
               with_chat_template: ChatTemplate | None = None) -> list[int]: ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str: ...

    def to_manifest_field(self) -> dict: ...
```

```python
# src/anvil/primitives/sampler.py

@dataclass(frozen=True, slots=True)
class Sampler:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_tokens: int = 2048
    seed: int | None = None
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    n: int = 1
    source: str = "explicit"
    # source ∈ {"explicit", "generation_config", "greedy", "reasoning_default"}

    @classmethod
    def greedy(cls, *, max_tokens: int = 2048, **overrides) -> Self: ...

    @classmethod
    def from_generation_config(cls, model_id: str, *,
                               revision: str | None = None) -> Self:
        """Explicit opt-in to model's generation_config.json defaults.
        ALWAYS records source='generation_config' in the manifest."""

    @classmethod
    def for_reasoning_model(cls, model_id: str, *,
                            max_tokens: int = 32768, **overrides) -> Self: ...

    @property
    def hash(self) -> str: ...

    def diff(self, other: Sampler) -> dict:
        """Dict of fields where self != other."""

    def is_argmax_invariant(self) -> bool:
        """True if temperature=0, top_k in {-1, 1}, no penalties."""

    def to_manifest_field(self) -> dict: ...
```

```python
# src/anvil/primitives/logits_processor.py
from typing import Protocol
import torch

class LogitsProcessor(Protocol):
    """Per-request logits processor (V0-style API, restored).

    The engine batches requests; processors that don't apply to a
    particular request should return its logits unchanged. The wrapper
    layer (anvil/engine/_wrappers/logits_processor.py) handles batching.
    """
    requires_hidden_states: bool = False
    argmax_invariant: bool = True

    def process(
        self,
        request_id: str,
        token_ids: torch.Tensor,           # int64 [seq_len]
        logits: torch.Tensor,              # float [vocab_size]
        hidden_states: torch.Tensor | None,  # float [num_layers, seq_len, hidden_dim] or None
    ) -> torch.Tensor:                     # float [vocab_size]
        ...
```

```python
# src/anvil/primitives/hidden_state_spec.py
from typing import Literal
import torch

PositionSpec = (
    Literal["all", "last", "first", "image_tokens", "text_tokens"]
    | tuple[int, ...]
)

@dataclass(frozen=True, slots=True)
class HiddenStateSpec:
    layers: tuple[int, ...]            # negative indices supported
    positions: PositionSpec = "last"
    dtype: torch.dtype | None = None   # cast on copy; None = native
    pin_memory: bool = True

    def estimate_bytes(self, *, seq_len: int, hidden_dim: int) -> int: ...

    def to_manifest_field(self) -> dict: ...
```

```python
# src/anvil/primitives/request.py
from typing import Any, Callable, Literal, Union

@dataclass(frozen=True, slots=True)
class Generate:
    messages: list[dict] | None = None
    prompt: str | None = None
    sampler: Sampler | None = None
    logits_processors: tuple[LogitsProcessor, ...] = ()
    capture: HiddenStateSpec | None = None

@dataclass(frozen=True, slots=True)
class LogLikelihood:
    context: str
    continuation: str

@dataclass(frozen=True, slots=True)
class Embed:
    input: Any                                # text, image, audio, sequence — anything the model accepts
    layer: int = -1
    pool: Literal["mean", "cls", "last", "max", "none"] = "mean"

@dataclass(frozen=True, slots=True)
class Classify:
    input: Any
    label_set: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class Custom:
    fn: Callable[[list[Any]], list[Any]]
    inputs: list[Any] | None = None

Request = Union[Generate, LogLikelihood, Embed, Classify, Custom]
```

```python
# src/anvil/tasks/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterator

class Task(ABC):
    name: str
    dataset: str | Path | Callable[[], Iterator[dict]]
    fewshot_style: FewshotStyle = "interleaved"
    n_fewshot_default: int = 0
    metric: Callable | None = None

    # Optional sentinel for Tier-1 quality checking
    sentinel_prompt: str | None = None
    sentinel_expected: str | None = None
    sentinel_baseline_scores: dict[str, float] = {}  # model_id → expected score

    @abstractmethod
    def doc_to_request(self, doc: dict) -> Request: ...

    @abstractmethod
    def request_to_prediction(self, response: Any, doc: dict) -> Any: ...

    def aggregate(self, predictions: list[Any], docs: list[dict]) -> dict[str, float]:
        """Default: applies self.metric per-doc and means.
        Override for rank correlations, top-k, weighted averages, etc."""
        ...
```

```python
# src/anvil/engine/public.py
from typing import Any, Callable, Protocol

class Engine(Protocol):
    def loglikelihood(self, requests: list[LogLikelihood]) -> list[tuple[float, bool]]: ...
    def loglikelihood_rolling(self, requests: list[str]) -> list[float]: ...
    def generate_until(self, requests: list[tuple[str, list[str]]]) -> list[Generation]: ...
    def generate_logprobs(self, requests: list[Generate], top_k: int = 5) -> list[Generation]: ...
    def embed(self, requests: list[Embed]) -> list[EmbedResult]: ...
    def classify(self, requests: list[Classify]) -> list[ClassifyResult]: ...
    def custom(self, fn: Callable, inputs: list[Any]) -> list[Any]: ...

    @property
    def model_info(self) -> dict: ...

    @property
    def backend_hash(self) -> str:
        """sha256 of (engine name + version + key compile flags)."""

    def shutdown(self) -> None: ...
```

```python
# src/anvil/manifest/schema.py
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal

class ModelInfo(BaseModel):
    id: str
    revision: str           # always sha256
    dtype: str
    quantization: str | None
    config_hash: str        # sha256 of config.json
    architecture: str

class TaskInfo(BaseModel):
    name: str
    tier: Literal["curated", "imported", "custom"]
    version: str
    dataset_revision: str   # sha256
    n_fewshot: int
    metric: str
    request_type: str       # Generate | LogLikelihood | Embed | Classify | Custom

class CaaSAction(BaseModel):
    ts: str
    step: int
    trigger: str
    match_source: Literal["rule_engine", "kb", "llm"]
    action: str
    args: dict
    rationale: str
    confidence: float
    kb_entry_ids: list[str] = []
    previous_value: Any | None = None
    validator_result: Literal["pass", "fail", "timeout"]
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0

class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    anvil_version: str
    engine: dict                # {name, version, backend_hash}
    model: ModelInfo
    tokenization: dict
    chat_template: dict | None  # None for non-chat tasks
    sampler: dict | None        # None for embed/classify/custom
    tasks: list[TaskInfo]
    scores: dict[str, dict[str, float]]
    smoke_test: dict
    caas_log: list[CaaSAction] = []
    hardware: dict
    started_at: str
    ended_at: str
    manifest_signature: str = ""  # set by sign(); excluded from canonical form

    def canonical_json(self) -> str: ...
    def sign(self) -> "Manifest": ...
    def verify(self) -> bool: ...

    @classmethod
    def diff(cls, a: "Manifest", b: "Manifest") -> dict:
        """Dict of every field that differs and could explain a score delta."""
```

### 16.5 Manifest canonical JSON

The manifest is signed by computing sha256 of the *canonical* JSON. Canonicalization rules:

1. UTF-8, no BOM.
2. Keys sorted lexicographically at every nesting level.
3. Output formatted with `indent=2, separators=(',', ': '), ensure_ascii=False`.
4. The `manifest_signature` field is **excluded** from canonical form.
5. List ordering is preserved.
6. `None` → `null`. Absent fields are simply not present (no null-padding).
7. Floats: `repr()` then strip trailing `.0` only for integer-valued floats. NaN and Inf are forbidden — raise.
8. The `hash` fields of `ChatTemplate`, `Sampler`, etc. are computed on construction and embedded — they are not recomputed during signing.

`anvil.manifest.canonical.canonical_json(m: Manifest) -> str` is the reference. `anvil.manifest.sign.sign(m) -> Manifest` returns a copy with signature set. **Two manifests with the same canonical JSON have the same signature, byte-for-byte, on any machine.** This is an invariant the signature test in §16.10 verifies.

### 16.6 CaaS KB entry schema

```python
# src/anvil/caas/kb/schema.py
from pydantic import BaseModel, Field
from typing import Any, Literal

Severity = Literal["blocking", "warning", "review-required"]

class EngineConstraint(BaseModel):
    engine: str   # "vllm" | "transformers" | "anvil" | "any"
    spec: str     # PEP 440-style: ">=0.5,<0.20"

class FixSpec(BaseModel):
    type: Literal[
        "set_engine_flag", "unset_engine_flag", "set_env_var",
        "edit_yaml", "set_sampling_param", "install_package",
        "restart_engine", "give_up",
    ]
    flag: str | None = None
    name: str | None = None
    value: Any | None = None
    value_strategy: str | None = None
    # value_strategy ∈ {"literal", "lookup_from_model_card",
    #                   "largest_divisor_of_attn_heads",
    #                   "max_pixels_from_dataset", ...}

class KBEntry(BaseModel):
    id: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    category: Literal[
        "install", "model_loading", "memory", "tokenization",
        "sampler", "multimodal", "tool_calling", "harness", "parallelism",
    ]
    engines: list[EngineConstraint]
    signatures: list[str]            # regex patterns; entry matches if ANY matches
    fix: FixSpec
    severity: Severity = "blocking"
    references: list[str] = []        # GitHub issue URLs
    human_message: str
    expires_after: str | None = None  # e.g., "vllm>=0.25"
    requires_user_consent: bool = False
```

### 16.7 Seed CaaS KB (15 entries — covers the highest-frequency cases)

Hand-write these in v0. They cover an estimated ~70% of the silly-error volume from the gap analysis. Files live at `src/anvil/caas/kb/entries/<category>.yaml`.

**`install.yaml`:**
```yaml
- id: cuda_libcudart_version_mismatch
  category: install
  engines: [{engine: vllm, spec: ">=0.10"}]
  signatures:
    - "libcudart\\.so\\.\\d+: cannot open shared object file"
    - "ImportError: libcudart"
  fix:
    type: give_up   # surface; do not auto-fix driver/CUDA mismatches
  severity: blocking
  references: ["https://github.com/vllm-project/vllm/issues/28669"]
  human_message: |
    Your installed vLLM was built against a different CUDA major version
    than what's available on the host. Install the matching wheel:
        uv pip install -U vllm --extra-index-url https://download.pytorch.org/whl/cuXXX
    where cuXXX matches `nvidia-smi` output.

- id: flash_attention_sm_unsupported
  category: install
  engines: [{engine: any, spec: ">=0"}]
  signatures:
    - "FlashAttention requires building with sm version sm80-sm90"
    - "FATAL: FlashAttention.* but"
  fix:
    type: set_env_var
    name: VLLM_ATTENTION_BACKEND
    value: XFORMERS
  severity: warning
  references: ["https://github.com/vllm-project/vllm/issues/9060"]
  human_message: |
    Your GPU is older than Ampere; FlashAttention won't compile.
    Falling back to XFormers backend. Expect ~20–40% throughput loss.

- id: numpy_2x_abi_break
  category: install
  engines: [{engine: any, spec: ">=0"}]
  signatures:
    - "A module that was compiled using NumPy 1\\.x cannot be run in NumPy 2"
    - "numpy\\.dtype size changed"
  fix:
    type: install_package
    name: numpy
    value: "<2"
  severity: blocking
  requires_user_consent: true
  references: []
  human_message: |
    NumPy 2.x ABI break detected. Pin numpy<2 or upgrade transformers.
```

**`model_loading.yaml`:**
```yaml
- id: trust_remote_code_required
  category: model_loading
  engines: [{engine: any, spec: ">=0"}]
  signatures:
    - "requires you to execute the configuration file"
    - "set trust_remote_code=True"
  fix:
    type: set_engine_flag
    flag: "--trust-remote-code"
    value: "true"
  severity: review-required
  requires_user_consent: true   # ALWAYS. Code-execution surface.
  references:
    - "https://github.com/vllm-project/vllm/issues/354"
    - "https://github.com/vllm-project/vllm/issues/5244"
  human_message: |
    This model loads custom Python code from its repo. We will NOT enable
    --trust-remote-code without your explicit approval. Review the model
    card and confirm you trust the source.

- id: bf16_nan_on_volta
  category: model_loading
  engines: [{engine: any, spec: ">=0"}]
  signatures:
    - "NaN in.*bf16|bfloat16.*not supported"
  fix:
    type: set_engine_flag
    flag: "--dtype"
    value: "float16"
  severity: blocking
  references: []
  human_message: |
    Volta GPUs (V100) don't support bfloat16. Switching to fp16.

- id: quantization_method_mismatch
  category: model_loading
  engines: [{engine: any, spec: ">=0"}]
  signatures:
    - "quantization method.*does not match.*config\\.json"
    - "Unknown quantization type"
  fix:
    type: set_engine_flag
    flag: "--quantization"
    value_strategy: lookup_from_model_card
  severity: blocking
  references: []
  human_message: |
    The --quantization flag does not match the model's quantization_config
    in config.json. Detected: {detected}. Set explicitly.
```

**`memory.yaml`:**
```yaml
- id: tp_attn_heads_divisibility
  category: parallelism
  engines: [{engine: vllm, spec: ">=0.4"}]
  signatures:
    - "Total number of attention heads \\((\\d+)\\) must be divisible by tensor parallel size \\((\\d+)\\)"
  fix:
    type: set_engine_flag
    flag: "--tensor-parallel-size"
    value_strategy: largest_divisor_of_attn_heads
  severity: blocking
  references:
    - "https://github.com/vllm-project/vllm/issues/414"
    - "https://github.com/vllm-project/vllm/issues/596"
    - "https://github.com/vllm-project/vllm/issues/1041"
    - "https://github.com/vllm-project/vllm/issues/11797"
  human_message: |
    num_attention_heads={heads} is not divisible by --tensor-parallel-size={tp}.
    Largest divisor of {heads} that's ≤ available_gpus is {fix_value}.

- id: kv_cache_oom_high_max_model_len
  category: memory
  engines: [{engine: any, spec: ">=0"}]
  signatures:
    - "No available memory for the cache blocks"
    - "Try increasing.*gpu_memory_utilization|reducing.*max_model_len"
  fix:
    type: set_engine_flag
    flag: "--max-model-len"
    value_strategy: max_input_token_length_plus_max_output
  severity: blocking
  references: []
  human_message: |
    KV cache won't fit. Lowering --max-model-len to fit your dataset's
    actual longest input ({max_in}) plus generation budget ({max_out}).

- id: qwen_vl_max_pixels_default_too_high
  category: multimodal
  engines: [{engine: any, spec: ">=0"}]
  signatures:
    - "model_id_matches: Qwen/Qwen[2-9](\\.[0-9]+)?-VL-.*"
    - "memory profiling expects.*256.*GB|expected.*\\d{3,}.*GB"
  fix:
    type: set_engine_flag
    flag: "--mm-processor-kwargs"
    value_strategy: max_pixels_from_dataset
  severity: blocking
  references:
    - "https://github.com/vllm-project/vllm/issues/27706"
    - "https://github.com/vllm-project/vllm/issues/14184"
    - "https://github.com/vllm-project/vllm/issues/15364"
  human_message: |
    Qwen-VL family defaults to max_pixels=12,845,056. Your dataset's
    largest image is {max_dim}. Setting max_pixels to {fix_value}.
```

**`tokenization.yaml`:**
```yaml
- id: instruct_model_no_chat_template_warning
  category: tokenization
  engines: [{engine: anvil, spec: ">=0"}]
  signatures:
    - "model_id_matches: .*-(Instruct|instruct|Chat|chat|IT|it)$"
    - "no_chat_template_applied"
  fix:
    type: set_engine_flag
    flag: "--apply-chat-template"
    value: "true"
  severity: blocking
  references:
    - "https://github.com/EleutherAI/lm-evaluation-harness/issues/1841"
  human_message: |
    Loaded an Instruct/Chat model without a chat template. This silently
    degrades to base-model quality. Anvil refuses to run instruct models
    without a chat template. Use ChatTemplate.from_model() or pass
    --no-chat-template to override (NOT recommended).

- id: llama3_eot_runaway
  category: tokenization
  engines: [{engine: any, spec: ">=0"}]
  signatures:
    - "model_id_matches: meta-llama/(Meta-)?Llama-3.*-Instruct"
    - "generation_length >= max_tokens"
  fix:
    type: set_sampling_param
    name: stop_token_ids
    value: [128009]
  severity: blocking
  references:
    - "https://github.com/vllm-project/vllm/issues/4180"
    - "https://github.com/vllm-project/vllm/issues/4297"
  human_message: |
    Llama-3-Instruct emits <|eot_id|> (128009) as the assistant turn
    terminator, but vLLM doesn't add it to stop_token_ids by default.
    Adding it explicitly.

- id: chat_template_not_found_v044
  category: tokenization
  engines: [{engine: any, spec: ">=0"}]
  signatures:
    - "default chat template is no longer allowed"
    - "Cannot use chat template functions because tokenizer.chat_template is not set"
  fix:
    type: set_engine_flag
    flag: "--chat-template"
    value_strategy: lookup_from_model_card
  severity: blocking
  references: []
  human_message: |
    transformers v4.44+ rejects the default chat template. Looking up
    a compatible template from the model card or examples/chat_templates.
```

**`sampler.yaml`:**
```yaml
- id: generation_config_overrides_sampler_v0_8
  category: sampler
  engines: [{engine: vllm, spec: ">=0.8.0"}]
  signatures:
    - "generation_config.json defaults applied"
    - "sampler_source != 'explicit' AND output_differs_from_user_config"
  fix:
    type: set_engine_flag
    flag: "--generation-config"
    value: "vllm"
  severity: warning
  references: []
  human_message: |
    vLLM v0.8.0+ reads sampler defaults from generation_config.json by
    default, which silently overrides your Sampler. Setting
    --generation-config=vllm to force neutral defaults.

- id: reasoning_model_max_tokens_too_low
  category: sampler
  engines: [{engine: any, spec: ">=0"}]
  signatures:
    - "model_id_matches: .*(R1|QwQ|o1|deepseek-r1|reasoning).*"
    - "max_tokens < 4096"
  fix:
    type: set_sampling_param
    name: max_tokens
    value: 32768
  severity: warning
  references: []
  human_message: |
    Reasoning models (R1, QwQ, o1) emit long <think> blocks before the
    final answer. max_tokens={current} will truncate. Recommended: 32768.
```

**`harness.yaml`:**
```yaml
- id: gsm8k_flexible_extract_picks_first_number
  category: harness
  engines: [{engine: anvil, spec: ">=0"}]
  signatures:
    - "task_name: gsm8k"
    - "filter: flexible-extract"
  fix:
    type: edit_yaml
    name: filter
    value: "strict-match"
  severity: warning
  references:
    - "https://github.com/EleutherAI/lm-evaluation-harness/issues/2278"
    - "https://github.com/EleutherAI/lm-evaluation-harness/issues/3214"
  human_message: |
    GSM8K's flexible-extract filter picks the FIRST number in the output,
    not the last. For chain-of-thought models this scores intermediate
    reasoning numbers as final answers. Switching to strict-match.
```

**Adding entries:** v0 ships exactly these 15. v0.5 expands to ~50, v1 to ~150. Every new entry needs (a) at least one citation, (b) an `engines:` constraint, (c) a unit test in `tests/unit/test_caas_kb_<category>.py` that the rule engine matches the signature.

### 16.8 Pinned dependencies (`pyproject.toml`)

```toml
[project]
name = "anvil"
version = "0.0.1"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4,<3",
    "transformers>=4.45,<5",
    "tokenizers>=0.20",
    "accelerate>=1.0",
    "pydantic>=2.7,<3",
    "typer>=0.12",
    "rich>=13.7",          # CLI rendering
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pyyaml>=6.0",
    "jinja2>=3.1",
    "huggingface-hub>=0.26",
    "datasets>=3.0",
    "numpy>=1.26,<3",
    "Pillow>=10.0",
    "safetensors>=0.4",
    "filelock>=3.13",
    "tqdm>=4.66",
]

[project.optional-dependencies]
vllm = ["vllm==0.20.1"]                # pinned per Anvil release
flash-attn = ["flash-attn>=2.6"]
multimodal = ["av>=12.0", "decord>=0.6", "librosa>=0.10"]
xgrammar = ["xgrammar>=0.1.10"]
outlines = ["outlines>=0.1"]
rocm = ["torch>=2.4,<3"]               # constraint differs at install
xpu = ["torch>=2.4,<3"]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",
    "pytest-cov>=5.0",
    "ruff>=0.6",
    "mypy>=1.10",
    "import-linter>=2.0",
    "pre-commit>=3.7",
    "hypothesis>=6.100",
]

[project.scripts]
anvil = "anvil.cli.main:app"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM", "TCH"]

[tool.mypy]
strict = true
python_version = "3.11"
```

`vllm==0.20.1` is the v0 pin. Bumping requires running the integration test suite against the new vLLM and updating `version_compat.py` for any CLI changes.

### 16.9 Conventions

**Logging.** stdlib `logging`, configured in `anvil.logging.setup()`. Each record is a JSON line on stderr in CI, human-formatted on TTY (rich). Levels: DEBUG (engine internals), INFO (user-facing progress), WARNING (recoverable), ERROR (run-aborting). Module loggers: `anvil.engine`, `anvil.tasks.<task_name>`, `anvil.caas`, etc. Never use `print()` outside CLI output formatters.

**Errors.**
```python
class AnvilError(Exception): ...
class ConfigError(AnvilError): ...
class EngineError(AnvilError): ...
class ModelLoadError(AnvilError): ...
class TokenizationError(AnvilError): ...
class TaskError(AnvilError): ...
class ManifestError(AnvilError): ...
class CaaSError(AnvilError): ...
class CaaSCannotFix(CaaSError): ...
class PluginError(AnvilError): ...
```
Each error has a stable `error_code` attribute (`"ANVIL-E0042"`) referenced in docs and the CaaS KB.

**Async model.** Engine internals are sync; the engine wraps async vLLM calls in a single sync facade. The HTTP server is async (FastAPI). The CLI is sync. Revisit if vLLM's async-only paths force the issue.

**CLI parsing.** `typer`. One subcommand per top-level verb. Every flag has a Python-API equivalent. No mutually-exclusive flag chains; if logic gets complex, accept a YAML config instead.

**Config files.** YAML, validated against Pydantic `Config`. Default location: `./anvil.yaml` or `~/.config/anvil/config.yaml`. Env-var override: `ANVIL_*` for any field. Precedence: CLI flag > env var > local config > user config > defaults.

**Plugin discovery.** Python entry points under `anvil.plugins.v1`. Plugin protocol versioned; v1 has a one-major-release deprecation cycle when v2 ships. Plugins register: model architectures, request types, metrics, KB entries, engines.

**Versioning policy (semver).** Public API is `anvil.__all__` plus the documented schemas (`Manifest`, `KBEntry`, plugin protocols). Breaking change: any field removal, any field type change, any required-field addition in `Manifest`, any change to canonical-JSON serialization, any rename in `__all__`. KB entries are NOT public API; they may change at any release with the `expires_after` mechanism.

### 16.10 Milestone-ordered build plan with acceptance tests

Each milestone produces a runnable artifact and a passing acceptance test. Later milestones depend on earlier ones; **do not start M+1 until M is green.** Time estimates assume one engineer or one Claude Code agent session.

#### Milestone 0 — "Hello, eval" (week 1–2)

Smallest end-to-end slice. HF slow path only. One task. Manifest verifiable.

Scope: `primitives/`, `engine/_hf/runner.py`, `manifest/` (no signing yet), `tasks/base.py`, `tasks/builtin/gsm8k.py`, `models/_slow/transformers_text.py`, `cli/main.py`, `cli/eval.py`. **No vLLM, no CaaS, no chat template canonicalization, no fast paths.**

Acceptance test (`tests/integration/test_milestone_0.py`):
```python
def test_milestone_0_end_to_end(tmp_path):
    """Smallest viable run: HF slow path, GSM8K, 50 samples, manifest emitted."""
    import anvil
    result = anvil.eval(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tasks=["gsm8k"],
        n_fewshot=5,
        limit=50,                      # subset for speed
        output_dir=tmp_path,
    )
    assert "gsm8k" in result.scores
    assert 0.40 < result.scores["gsm8k"]["accuracy"] < 0.95   # sanity
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    m = anvil.Manifest.model_validate_json(manifest_path.read_text())
    assert m.model.id == "meta-llama/Llama-3.1-8B-Instruct"
    assert m.tasks[0].name == "gsm8k"
    assert m.tasks[0].n_fewshot == 5
```

#### Milestone 1 — "Real engine" (week 3–4)

Add vLLM wrapper backend. Add MMLU and HumanEval+. ChatTemplate canonicalization. Engine factory.

Scope: `engine/_vllm/`, `engine/factory.py`, `tasks/builtin/mmlu.py`, `tasks/builtin/humaneval.py`, `primitives/chat_template.py` full implementation including canonicalization.

Acceptance:
```python
def test_milestone_1_chat_template_hash_stable():
    ct1 = anvil.ChatTemplate.from_model("Qwen/Qwen2.5-7B-Instruct")
    ct2 = anvil.ChatTemplate.from_model("Qwen/Qwen2.5-7B-Instruct")
    assert ct1.hash == ct2.hash

def test_milestone_1_vllm_matches_hf_within_tolerance():
    """Same model, same task, same sampler — vLLM and HF agree to 0.5pp."""
    cfg = dict(model="meta-llama/Llama-3.1-8B-Instruct",
               tasks=["mmlu"], n_fewshot=5, limit=200,
               sampler=anvil.Sampler.greedy())
    a = anvil.eval(**cfg, engine="hf")
    b = anvil.eval(**cfg, engine="vllm")
    assert abs(a.scores["mmlu"]["accuracy"] - b.scores["mmlu"]["accuracy"]) < 0.005

def test_milestone_1_known_baseline():
    """Llama-3.1-8B-Instruct on MMLU 5-shot ≈ 0.69 ± 0.01 (published)."""
    result = anvil.eval(model="meta-llama/Llama-3.1-8B-Instruct",
                        tasks=["mmlu"], n_fewshot=5)
    assert 0.68 < result.scores["mmlu"]["accuracy"] < 0.70
```

#### Milestone 2 — "Reproducibility" (week 5–6)

Manifest canonical JSON + signing + verify/diff/replay. Cross-engine determinism.

Scope: `manifest/canonical.py`, `manifest/sign.py`, `manifest/verify.py`, `manifest/diff.py`, `manifest/replay.py`, CLI `manifest verify/diff/replay`.

Acceptance:
```python
def test_milestone_2_canonical_byte_stable():
    """Same manifest, two machines, same canonical bytes."""
    m = make_fixture_manifest()
    j1 = m.canonical_json()
    j2 = m.canonical_json()
    assert j1 == j2
    assert hashlib.sha256(j1.encode()).hexdigest() == m.sign().manifest_signature

def test_milestone_2_replay_reproduces_score():
    """Run, replay manifest, scores byte-identical."""
    r1 = anvil.eval(model="...", tasks=["gsm8k"], limit=20)
    r1.manifest.save("run1.json")
    r2 = anvil.manifest.replay("run1.json")
    assert r1.scores == r2.scores

def test_milestone_2_diff_explains_score_delta():
    """Two runs with one delta in sampler explain the delta in diff."""
    r_greedy = anvil.eval(model="...", tasks=["gsm8k"],
                          sampler=anvil.Sampler.greedy(), limit=20)
    r_t07    = anvil.eval(model="...", tasks=["gsm8k"],
                          sampler=anvil.Sampler(temperature=0.7, seed=42), limit=20)
    diff = anvil.Manifest.diff(r_greedy.manifest, r_t07.manifest)
    assert "sampler" in diff
    assert "temperature" in diff["sampler"]
```

#### Milestone 3 — "CaaS v0" (week 7–9)

Smoke-test orchestrator + rule engine + KB loader + 15 seed entries (§16.7) + diff-and-confirm UI + audit log → manifest.

Scope: full `caas/` directory except `llm_tier.py` (stub).

Acceptance:
```python
def test_milestone_3_kb_loads_all_entries():
    """All shipped KB entries are valid against the schema."""
    from anvil.caas.kb.loader import load_all
    entries = load_all()
    assert len(entries) >= 15
    for e in entries:
        assert e.id and e.signatures and e.fix and e.human_message

def test_milestone_3_tp_divisibility_auto_fixes(monkeypatch):
    """Setting tp_size=3 on a 32-head model auto-corrects to 4."""
    monkeypatch.setenv("ANVIL_CAAS_MODE", "ci")
    result = anvil.eval(
        model="meta-llama/Llama-3.1-8B-Instruct",  # 32 heads
        tasks=["gsm8k"], limit=5,
        engine_args={"tensor_parallel_size": 3},
    )
    actions = result.manifest.caas_log
    assert any(a.action == "set_engine_flag"
               and a.args.get("flag") == "--tensor-parallel-size"
               and a.args.get("value") in (4, 2, 1)
               for a in actions)

def test_milestone_3_test_corpus_resolution_rate():
    """≥60% auto-resolve on the seed test corpus, 0% false-positive."""
    from tests.corpus import iter_caas_test_cases
    resolved = 0; surfaced = 0; false_positive = 0; total = 0
    for case in iter_caas_test_cases():
        total += 1
        outcome = run_caas_against(case)
        if outcome == "resolved": resolved += 1
        elif outcome == "surfaced_correctly": surfaced += 1
        elif outcome == "false_positive": false_positive += 1
    assert resolved / total >= 0.60
    assert false_positive / total <= 0.02
```

#### Milestone 4 — "Multimodal" (week 10–12)

VLM via slow path; Qwen2.5-VL fast path; MMMU task; VLM-aware preflight (smallest/median/largest image samples).

Scope: `models/_slow/transformers_mm.py`, `models/_fast/qwen_vl.py`, `tasks/builtin/mmmu.py`, VLM-specific KB entries.

Acceptance:
```python
def test_milestone_4_vlm_basic_generation():
    m = anvil.load("Qwen/Qwen2.5-VL-7B-Instruct")
    out = m.generate(messages=[{"role": "user", "content": [
        {"type": "image", "image": load_test_image("cat.png")},
        {"type": "text", "text": "What is in this image?"}
    ]}])
    assert "cat" in out.text.lower()
    assert out.image_token_counts == [1280] or len(out.image_token_counts) == 1

def test_milestone_4_mmmu_known_baseline():
    """Qwen2.5-VL-7B on MMMU ≈ 0.50 ± 0.02 (published)."""
    result = anvil.eval(model="Qwen/Qwen2.5-VL-7B-Instruct",
                        tasks=["mmmu"], limit=200)
    assert 0.47 < result.scores["mmmu"]["accuracy"] < 0.55

def test_milestone_4_image_size_smoke():
    """Preflight runs smallest + median + largest image; OOM caught here, not at sample N."""
    # synthetic dataset with 4K image at index 7
    result = anvil.eval(model="Qwen/Qwen2.5-VL-7B-Instruct",
                        tasks=[make_dataset_with_4k_image_at(7)],
                        limit=10)
    assert any(a.kb_entry_ids and "qwen_vl" in a.kb_entry_ids[0]
               for a in result.manifest.caas_log)
```

#### Milestone 5 — "Migration" (week 13–14)

lm-eval-harness shim. `anvil eval --compare-with-lm-eval`. Custom-task registration. Non-text custom modality.

Scope: `tasks/lm_eval_shim/`, `tasks/registry.py` full, the RNA example test.

Acceptance:
```python
def test_milestone_5_lm_eval_yaml_imports():
    """Import an existing lm-eval task and run."""
    result = anvil.eval(model="meta-llama/Llama-3.1-8B-Instruct",
                        lm_eval_tasks=["arc_challenge"], limit=50)
    assert "arc_challenge" in result.scores
    assert result.manifest.tasks[0].tier == "imported"

def test_milestone_5_compare_with_lm_eval():
    """Anvil score within 1pp of lm-eval-harness on the same task."""
    a = anvil.eval(model="...", lm_eval_tasks=["arc_challenge"], limit=200)
    b = run_lm_eval_directly(model="...", task="arc_challenge", limit=200)
    assert abs(a.scores["arc_challenge"]["acc"] - b["acc"]) < 0.01

def test_milestone_5_custom_modality_rna():
    """The §6.7 RNA example actually runs."""
    # see tests/integration/test_rna_custom_task.py
    ...
```

#### Milestone 6 — "v0 release" (week 15–16)

uv-installable wheels (cu121, cu128, cu130). README walkthrough. Five fast-path architectures (Llama 3, Qwen 2.5, Mistral, Gemma 3, Phi 4). OpenAI-compatible serve with constrained-decoding tool calls. `anvil doctor`.

Acceptance:
- `uv pip install anvil` works on Linux x86_64 with torch 2.4–2.10.
- README example runs end-to-end on H100, A100, RTX 5090, MI300X, GH200.
- All five fast-path architectures pass `test_milestone_1_known_baseline` for their canonical reference checkpoint.
- `anvil serve --model X` answers OpenAI-compatible chat completions and tool calls.
- `anvil doctor` correctly diagnoses 8 of 10 simulated environment problems from the test corpus.

### 16.11 Test corpus seed (`tests/corpus/caas_test_cases.yaml`)

Hand-write 10 test cases for v0; expand to 50 by v0.5. Each case has the original error, the expected match, and the expected fix.

```yaml
- name: tp_3_on_32_head_llama
  scenario_setup: |
    engine = vllm
    model = meta-llama/Llama-3.1-8B-Instruct
    args = {tensor_parallel_size: 3}
  expected_error_signature: "Total number of attention heads (32) must be divisible by tensor parallel size (3)"
  expected_match: tp_attn_heads_divisibility
  expected_fix:
    type: set_engine_flag
    flag: "--tensor-parallel-size"
    value: 4   # largest divisor of 32 ≤ available_gpus
  expected_outcome: resolved

- name: llama3_instruct_no_chat_template
  scenario_setup: |
    model = meta-llama/Llama-3.1-8B-Instruct
    apply_chat_template = false
  expected_error_signature: "Loaded an Instruct/Chat model without a chat template"
  expected_match: instruct_model_no_chat_template_warning
  expected_fix:
    type: set_engine_flag
    flag: "--apply-chat-template"
    value: true
  expected_outcome: resolved

- name: qwen_vl_3b_oom_default_pixels
  scenario_setup: |
    model = Qwen/Qwen2.5-VL-3B-Instruct
    dataset_largest_image: [1280, 768]
  expected_error_signature: "memory profiling expects.*GB"
  expected_match: qwen_vl_max_pixels_default_too_high
  expected_fix:
    type: set_engine_flag
    flag: "--mm-processor-kwargs"
    value: {max_pixels: 1003520}
  expected_outcome: resolved

- name: glm47_requires_transformers_main
  scenario_setup: |
    model = glm-4.7-flash
    transformers_version: "4.45.0"
  expected_error_signature: "transformers does not recognize.*glm4_moe_lite"
  expected_match: null   # surfaces, does not auto-fix
  expected_outcome: surfaced_correctly
  surfaced_message_contains: ["transformers from git", "review"]

- name: cuda_12_wheel_on_cuda_13_host
  scenario_setup: |
    vllm_built_against: cuda12
    host_cuda: 13.0
  expected_error_signature: "libcudart\\.so\\.12: cannot open shared object file"
  expected_match: cuda_libcudart_version_mismatch
  expected_outcome: surfaced_correctly  # never auto-installs torch
  surfaced_message_contains: ["matching wheel", "cuXXX"]

# ...4 more cases for v0...
```

### 16.12 What the agent should do, in order

1. Initialize the repo with the layout in §16.1 and `pyproject.toml` from §16.8.
2. Set up `import-linter` (§16.2), `ruff`, `mypy strict`, `pre-commit`, GitHub Actions CI.
3. Implement `primitives/` with the signatures in §16.4. Unit-test each.
4. Implement `manifest/schema.py` and `manifest/canonical.py`. Test that `canonical_json` is byte-stable.
5. Implement `engine/_hf/runner.py` (HF slow path). Test loading Llama-3.1-8B-Instruct.
6. Implement `tasks/base.py`, `tasks/runner.py`, `tasks/builtin/gsm8k.py`. Pass M0 acceptance.
7. Implement `engine/_vllm/adapter.py`, `tasks/builtin/mmlu.py`, `tasks/builtin/humaneval.py`. Pass M1.
8. Implement `manifest/sign.py`, `verify.py`, `diff.py`, `replay.py`. Pass M2.
9. Implement `caas/`. Hand-write the 15 KB entries from §16.7 and 10 test corpus cases from §16.11. Pass M3.
10. Implement multimodal slow path + Qwen2.5-VL fast path. Pass M4.
11. Implement lm-eval-harness shim. Pass M5.
12. Build wheels, write README, register five fast paths. Pass M6.

Open questions to surface to the human (do not silently decide):
- Plugin entry-point naming (`anvil.plugins.v1.models` vs flat namespace) — recommend the agent picks and documents.
- `anvil doctor` output format (rich-table vs JSON) — the agent should ship rich-table by default with `--json` flag.
- HuggingFace token handling (env var only vs config-file fallback) — env var only for v0, config later.
- Caching layer for tokenized datasets — defer until profiling shows it's needed.

Anything not specified here is the agent's decision, with the constraint that it must be consistent with the principles in §2 of the design manuscript.

---

## 17. Appendix A — Concrete API Examples

### A.1 The simplest possible eval

```python
import anvil
result = anvil.eval(model="Qwen/Qwen2.5-7B-Instruct", tasks=["mmlu"])
print(result.scores["mmlu"])    # 0.7451
result.manifest.save("run.json")
```

### A.2 The full research-mode eval

```python
import anvil
from anvil import Sampler, ChatTemplate, HiddenStateSpec
from anvil.research import DoLa

ct = ChatTemplate.from_model("meta-llama/Llama-3.3-70B-Instruct")
print(ct.hash)   # pin for reproducibility

result = anvil.eval(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tasks=["gsm8k", "math500"],
    chat_template=ct,
    sampler=Sampler(temperature=0.0, max_tokens=4096, seed=42),
    logits_processors=[DoLa(mature_layer=-1, premature_layers=[0, 12, 24])],
    capture=HiddenStateSpec(layers=[-1], positions="last"),
    n_fewshot=4,
    caas="research",
    output_dir="./runs/dola-llama33/",
)

result.scores              # {"gsm8k": ..., "math500": ...}
result.manifest.save("./runs/dola-llama33/manifest.json")
result.outputs[0].hidden_states[-1]   # captured activations
```

### A.3 Custom decoding strategy

```python
from anvil import LogitsProcessor

class TopOnlyAfterFirstNumber(LogitsProcessor):
    """Once the model emits a digit, restrict to digits and arithmetic."""
    requires_hidden_states = False
    argmax_invariant = False    # we modify the argmax

    def process(self, request_id, token_ids, logits, hidden_states):
        decoded = self.tokenizer.decode(token_ids)
        if any(c.isdigit() for c in decoded):
            mask = torch.full_like(logits, float("-inf"))
            allowed = self.tokenizer.encode("0123456789+-*/= ", add_special_tokens=False)
            mask[..., allowed] = 0
            return logits + mask
        return logits

result = anvil.eval(
    model="Qwen/Qwen2.5-Math-7B-Instruct",
    tasks=["gsm8k"],
    logits_processors=[TopOnlyAfterFirstNumber()],
)
```

### A.4 Custom dataset

```python
from anvil import Task, register_task, exact_match

register_task(Task(
    name="my_internal_qa",
    dataset="myorg/private-qa-set",
    template="Q: {question}\nA:",
    target_field="answer",
    metric=exact_match,
    fewshot_style="interleaved",
))

result = anvil.eval(model="...", tasks=["my_internal_qa"])
```

### A.5 Custom non-text modality (RNA sequences, embedding model)

```python
import anvil
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

# Non-causal model, slow path.
model = anvil.load_custom(
    model_id="multimolecule/rnafm",
    model_class=AutoModel,
    tokenizer_class=AutoTokenizer,
)

@anvil.register_task
class RNAStability(anvil.Task):
    name = "rna_stability_v1"
    dataset = "myorg/rna-stability-set"     # or local jsonl

    def doc_to_request(self, doc):
        return anvil.Embed(input=doc["sequence"], pool="mean", layer=-1)

    def request_to_prediction(self, response, doc):
        return response.embedding

    def aggregate(self, predictions, docs):
        # Held-out cross-validated linear probe — your call, your metric.
        X = [p.numpy() for p in predictions]
        y = [d["half_life"] for d in docs]
        model = Ridge(alpha=1.0).fit(X[:800], y[:800])
        preds = model.predict(X[800:])
        return {"spearman": spearmanr(preds, y[800:]).statistic,
                "r2":       model.score(X[800:], y[800:])}

result = anvil.eval(model=model, tasks=["rna_stability_v1"])
print(result.scores)        # {"rna_stability_v1": {"spearman": 0.79, "r2": 0.61}}
result.manifest.save("rna_run.json")
# Manifest records: model SHA, dataset SHA, library version, pooling strategy,
# layer index, no chat_template (correctly absent), no sampler (correctly absent),
# CaaS log if any preflight fixes were applied.
```

The same pattern works for audio, protein, graph, image-encoder, and any other modality where you have a model and a dataset. The only constraint is that `engine.embed` (or `engine.custom`) accepts your input type and the model produces something `aggregate()` can score.

### A.6 Migrating an existing lm-evaluation-harness workflow

```bash
# Before:
lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,gpu_memory_utilization=0.9 \
    --tasks mmlu_pro,arc_challenge \
    --apply_chat_template \
    --num_fewshot 5 \
    --output_path ./out

# After:
anvil eval --model Qwen/Qwen2.5-7B-Instruct \
    --lm-eval-tasks mmlu_pro,arc_challenge \
    --n-fewshot 5 \
    --output ./run.json

# Verify:
anvil eval --model Qwen/Qwen2.5-7B-Instruct \
    --lm-eval-tasks mmlu_pro,arc_challenge \
    --n-fewshot 5 \
    --compare-with-lm-eval \
    --output ./run.json
# Produces a delta report: which deltas (chat_template, sampler, fewshot_style) account for any score difference.
```

### A.6 What CaaS looks like in practice

```bash
$ anvil eval --model Qwen/Qwen3-VL-8B --tasks mmmu --caas research

[anvil] Loading Qwen/Qwen3-VL-8B (fast path)...
[anvil] Smoke test: 3 samples + quality sentinel...
[caas]  ✗ smoke test failed:
        OOM: requested 256 GB for KV cache, available 80 GB.
[caas]  Diagnosis (rule_engine + kb match: mm_processor_kwargs_default_too_high):
        Qwen3-VL defaults to max_pixels=12,845,056 per image.
        Your images are ≤ 1280×768. Recommended: max_pixels=1280*28*28.

  Proposed fix:
  - mm_processor_kwargs: {}
  + mm_processor_kwargs: {max_pixels: 1003520, min_pixels: 50176}
  + limit_mm_per_prompt: {image: 1, video: 0}

  References: vllm #27706, #14184, #15364

  Apply? [y/N/edit/explain]: y

[anvil] Re-running smoke test...
[caas]  ✓ smoke test passed (sentinel: 0.94 vs baseline 0.91)
[anvil] Starting eval: mmmu (10500 samples)...
[anvil] ETA: 23 minutes.
```

---

## 18. Appendix B — Comparison to Existing Libraries

| | Anvil | vLLM | SGLang | lm-eval-harness | lighteval | OpenCompass |
|---|---|---|---|---|---|---|
| Day-zero new model support | ✓ (slow path default) | ~ (PR cycle) | ✓ | n/a (depends on backend) | ~ | ~ |
| Per-request logits processors | ✓ | ✗ (V1 dropped) | ✗ | n/a | n/a | n/a |
| Hidden-state extraction (public) | ✓ | ✗ (RFC open) | ✗ | n/a | n/a | n/a |
| Custom decoding (DoLa, contrastive, etc.) | ✓ stable API | ✗ | ✗ | n/a | n/a | n/a |
| Versioned ChatTemplate | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Reproducibility manifest | ✓ | ✗ | ✗ | partial | partial | partial |
| Preflight self-healing (CaaS) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Batched eval primitives at engine speed | ✓ | n/a | n/a | ✗ (per-doc loop) | ✗ (slow) | partial |
| lm-eval-harness migration shim | ✓ | n/a | n/a | n/a | partial | ✗ |
| Constrained-decoding tool calling | ✓ | ✗ (per-model parsers) | partial | n/a | n/a | n/a |
| uv-installable, multi-CUDA wheels | ✓ | ✗ | ✗ | n/a | ✓ | n/a |
| Production datacenter throughput | ~ | ✓ | ✓ | n/a | n/a | n/a |
| OpenAI-compatible server | ✓ | ✓ | ✓ | n/a | n/a | n/a |

The table is honest. Anvil is not the throughput winner. Anvil is the correctness-and-reproducibility-and-research-ergonomics winner, which is a different niche.

---

## 19. Appendix C — Acknowledgments and Source Material

This design synthesizes prior work and observations from:

- **vLLM** (Kwon et al., SOSP 2023; the project's RFCs and release notes 2024–2026)
- **SGLang** (Zheng et al., NeurIPS 2024)
- **lm-evaluation-harness** (EleutherAI, Biderman et al., "Lessons from the Trenches on Reproducible Evaluation of Language Models", NeurIPS Datasets 2024)
- **Agentless** (Xia et al., arXiv:2407.01489) for the CaaS architectural pattern
- **RepairAgent** (Bouzenia et al., ICSE 2025) for the FSM-bounded action space
- **"When Can LLMs Actually Correct Their Own Mistakes?"** (Huang et al., arXiv:2406.01297) for the no-self-correction-without-oracle constraint
- **Aider** (Gauthier 2023–2026) for the SEARCH/REPLACE edit format and small-model design lessons
- **Qwen2.5-Coder** (Hui et al., arXiv:2409.12186) for the recommended CaaS coder model
- **Granite-4.1** (IBM, 2026) for the production-tuned alternative
- The vLLM, SGLang, lm-evaluation-harness, lighteval, and OpenCompass GitHub issue trackers, where the actual user pain is documented one issue at a time

---

*End of manuscript.*
