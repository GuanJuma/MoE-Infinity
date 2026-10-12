# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Tencent Hy3 (``HYV3ForCausalLM``) on MoE-Infinity's expert store.

The non-expert backbone (embeddings, attention, dense layer 0, shared
experts, routers, ``lm_head``) is dequantized once to BF16 and stays on the
GPU; the 79 x 192 routed FP8 experts live in host memory and are moved to a
GPU expert cache on demand.  Two stores are available:

* ``mi`` -- MoE-Infinity's native store (``moe_infinity._store``): offload
  files on disk, pinned host pool, ``begin``/``end``/``prefetch_tensors``
  fetch path and its sparse-cache eviction;
* ``pinned`` -- a pure-PyTorch pinned host copy with an LRU slot cache.

Routed experts run as FP8 GEMMs (``torch._scaled_mm`` with the checkpoint's
static input scales) or BF16 after on-device dequantization.  Neither path
goes through ``MoEMLP``/``fused_moe_ffn_into`` (issue #245).  Every MoE
layer's top-k expert ids can be recorded per forward pass
(:mod:`moe_infinity.hy3.routing_trace`).
"""

from .checkpoint import Hy3Checkpoint
from .engine import Hy3Engine, Hy3EngineConfig
from .routing_trace import RoutingTraceRecorder, load_trace

__all__ = [
    "Hy3Checkpoint",
    "Hy3Engine",
    "Hy3EngineConfig",
    "RoutingTraceRecorder",
    "load_trace",
]
