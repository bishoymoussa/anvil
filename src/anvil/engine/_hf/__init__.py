"""HuggingFace transformers slow path (design §3.1, §3.3).

This is the day-zero coverage path: anything that loads in
``transformers.AutoModelForCausalLM`` loads in Anvil. Throughput is 2–4×
lower than the vLLM fast path on most architectures; for evaluation at the
day a model drops, that is the right trade-off.
"""

from __future__ import annotations
