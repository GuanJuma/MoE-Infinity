# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Hy3 inference with offloaded routed experts.

transformers' ``HYV3ForCausalLM`` (transformers >= 5.12) supplies attention
(GQA 64/8, q/k RMSNorm, RoPE), the dense layer 0, norms and the KV cache.
Each sparse layer's ``HYV3MoE`` is replaced by :class:`Hy3OffloadMoE`: the
same router (sigmoid + ``e_score_correction_bias`` top-8, renormalized and
scaled by ``router_scaling_factor``) and shared expert, with the routed
experts fetched from the expert store and run by :class:`ExpertFFN`.
Batch size 1, chunked prefill, greedy or sampled decoding.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .checkpoint import Hy3Checkpoint, describe
from .experts import ExpertFFN, build_store
from .routing_trace import RoutingTraceRecorder

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class Hy3EngineConfig:
    model_dir: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    store: str = "mi"  # mi | pinned
    offload_dir: Optional[str] = None
    expert_cache_gib: float = 32.0
    device_memory_ratio: float = 0.6
    expert_compute: str = "fp8"  # fp8 | bf16
    attn_implementation: str = "sdpa"
    prefill_chunk: int = 4096
    max_context: int = 20480
    rebuild_store: bool = False
    record_weights: bool = False


def _hf_modules():
    try:
        from transformers.models.hy_v3 import modeling_hy_v3 as m
    except ImportError as e:  # pragma: no cover - depends on the image
        raise ImportError(
            "Hy3 needs transformers >= 5.12 (transformers.models.hy_v3); "
            f"found an older build: {e}"
        )
    return m


class Hy3OffloadMoE(nn.Module):
    """``HYV3MoE`` with routed experts served from the expert store."""

    def __init__(self, config, layer_idx: int, moe_idx: int):
        super().__init__()
        m = _hf_modules()
        self.layer_idx = layer_idx
        self.moe_idx = moe_idx
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.enable_moe_fp32_combine = config.enable_moe_fp32_combine
        self.gate = m.HYV3TopKRouter(config)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_experts, dtype=torch.float32),
        )
        self.shared_experts = m.HYV3MLP(
            config,
            intermediate_size=config.moe_intermediate_size
            * config.num_shared_experts,
        )
        self._engine = None

    def bind(self, engine: "Hy3Engine") -> None:
        object.__setattr__(self, "_engine", engine)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        x = hidden_states.view(-1, H)
        _, topk_w, topk_ids = self.gate(x, self.e_score_correction_bias)
        eng = self._engine
        eng.recorder.on_route(self.layer_idx, topk_ids, topk_w)
        routed = eng.run_experts(self.moe_idx, x, topk_ids, topk_w)
        shared = self.shared_experts(x)
        if self.enable_moe_fp32_combine:
            out = (routed.float() + shared.float()).to(x.dtype)
        else:
            out = routed + shared
        return out.view(B, S, H)


class Hy3Engine:
    def __init__(
        self, cfg: Hy3EngineConfig, log: Callable = print, store_lib=None
    ):
        self.cfg = cfg
        self.log = log
        self.device = torch.device(cfg.device)
        self.dtype = _DTYPES[cfg.dtype]
        t0 = time.perf_counter()
        self.ckpt = Hy3Checkpoint(cfg.model_dir)
        m = _hf_modules()
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(cfg.model_dir)
        if getattr(self.config, "model_type", "") != "hy_v3":
            raise ValueError(
                f"{cfg.model_dir} is model_type "
                f"{getattr(self.config, 'model_type', None)!r}, not hy_v3"
            )
        sparse = [
            i
            for i, t in enumerate(self.config.mlp_layer_types)
            if t == "sparse"
        ]
        if sparse != self.ckpt.moe_layers:
            raise ValueError(
                f"config sparse layers {sparse[:3]}..{sparse[-3:]} do not match "
                f"checkpoint expert layers {self.ckpt.moe_layers[:3]}.."
            )
        self.info = describe(self.ckpt)
        log(f"[hy3] checkpoint: {json.dumps(self.info, default=str)}")

        with torch.device("meta"):
            model = m.HYV3ForCausalLM._from_config(
                self.config,
                attn_implementation=cfg.attn_implementation,
                dtype=self.dtype,
            )
            self.moe_modules: List[Hy3OffloadMoE] = []
            for li, layer in enumerate(self.ckpt.moe_layers):
                mod = Hy3OffloadMoE(self.config, layer, li).to(self.dtype)
                model.model.layers[layer].mlp = mod
                self.moe_modules.append(mod)
        model.to_empty(device=self.device)
        for mod in self.moe_modules:
            mod.e_score_correction_bias.data = (
                mod.e_score_correction_bias.data.float()
            )
            mod.bind(self)
        model.model.rotary_emb = m.HYV3RotaryEmbedding(
            self.config, device=self.device
        )
        self._load_resident(model)
        model.eval()
        self.model = model
        self.resident_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) + sum(b.numel() * b.element_size() for b in model.buffers())
        log(
            f"[hy3] resident backbone: {self.resident_bytes / 2**30:.2f} GiB "
            f"({cfg.dtype}) on {self.device} ({time.perf_counter() - t0:.0f}s)"
        )

        scales = self.ckpt.expert_scales()
        self.ffn = ExpertFFN(
            cfg.expert_compute, scales, self.device, self.dtype
        )
        if cfg.expert_compute == "fp8" and not self.ffn.static:
            log("[hy3] WARNING: no static input scales; dynamic act quant")
        self.store = build_store(
            cfg.store,
            self.ckpt,
            self.device,
            int(cfg.expert_cache_gib * 2**30),
            offload_dir=cfg.offload_dir,
            device_memory_ratio=cfg.device_memory_ratio,
            rebuild=cfg.rebuild_store,
            log=log,
            lib=store_lib,
        )
        self.recorder = RoutingTraceRecorder(
            self.config.num_hidden_layers,
            self.config.num_experts_per_tok,
            record_weights=cfg.record_weights,
        )
        self._tokenizer = None
        self.stop_ids = self._stop_ids()
        log(f"[hy3] engine ready in {time.perf_counter() - t0:.0f}s")

    # -- loading ----------------------------------------------------------

    def _load_resident(self, model: nn.Module) -> None:
        targets: Dict[str, torch.Tensor] = dict(model.named_parameters())
        targets.update(dict(model.named_buffers()))
        loaded, unexpected = set(), []
        for name, t in self.ckpt.iter_resident(self.dtype, self.device):
            dst = targets.get(name)
            if dst is None:
                unexpected.append(name)
                continue
            if tuple(dst.shape) != tuple(t.shape):
                raise ValueError(
                    f"{name}: checkpoint {tuple(t.shape)} vs model "
                    f"{tuple(dst.shape)}"
                )
            with torch.no_grad():
                dst.copy_(t.to(dst.dtype))
            loaded.add(name)
        skip = ("inv_freq", "original_inv_freq")
        missing = [
            n for n in targets if n not in loaded and not n.endswith(skip)
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} model tensors not in the checkpoint, e.g. "
                f"{missing[:8]}"
            )
        if unexpected:
            self.log(
                f"[hy3] ignored {len(unexpected)} checkpoint tensors, e.g. "
                f"{unexpected[:5]}"
            )

    def _stop_ids(self) -> List[int]:
        ids = []
        sources = [getattr(self.config, "eos_token_id", None)]
        gc = Path(self.cfg.model_dir) / "generation_config.json"
        if gc.exists():
            sources.append(json.loads(gc.read_text()).get("eos_token_id"))
        for s in sources:
            if isinstance(s, int):
                ids.append(s)
            elif isinstance(s, list):
                ids.extend(int(x) for x in s)
        return sorted(set(ids))

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_dir)
            eos = getattr(self._tokenizer, "eos_token_id", None)
            if isinstance(eos, int) and eos not in self.stop_ids:
                self.stop_ids.append(eos)
        return self._tokenizer

    def encode_chat(
        self, messages: Sequence[dict], chat_template_kwargs: Optional[dict]
    ) -> List[int]:
        tok = self.tokenizer
        text = tok.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=False,
            **(chat_template_kwargs or {}),
        )
        return tok(text, add_special_tokens=False)["input_ids"]

    # -- experts ----------------------------------------------------------

    def run_experts(
        self,
        li: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_w: torch.Tensor,
    ) -> torch.Tensor:
        K = topk_ids.shape[1]
        E = self.config.num_experts
        flat = topk_ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=E).tolist()
        tok = order // K
        wts = topk_w.reshape(-1)[order]
        experts = [e for e in range(E) if counts[e]]
        self.store.prefetch(li, experts)
        out = torch.zeros_like(x)
        start = 0
        for e in experts:
            n = counts[e]
            t = tok[start : start + n]
            w = self.store.acquire(li, e)
            y = self.ffn(li, e, x[t], w)
            out.index_add_(
                0, t, (y * wts[start : start + n, None]).to(out.dtype)
            )
            self.store.release(li, e)
            start += n
        return out

    # -- generation -------------------------------------------------------

    def _sample(self, logits, temperature: float, top_p: float, gen):
        if temperature <= 0:
            return int(torch.argmax(logits, dim=-1))
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            keep = torch.cumsum(sp, dim=-1) - sp < top_p
            sp = sp * keep
            idx = torch.multinomial(sp / sp.sum(), 1, generator=gen)
            return int(si[idx])
        return int(torch.multinomial(probs, 1, generator=gen))

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        ignore_eos: bool = False,
        rid: Optional[str] = None,
        meta: Optional[dict] = None,
        trace: bool = True,
        on_token: Optional[Callable[[int], None]] = None,
    ) -> dict:
        from transformers import DynamicCache

        n = len(input_ids)
        if n == 0:
            raise ValueError("empty prompt")
        if n + max_new_tokens > self.cfg.max_context:
            raise ValueError(
                f"prompt {n} + max_new_tokens {max_new_tokens} exceeds "
                f"max_context {self.cfg.max_context}"
            )
        rid = rid or uuid.uuid4().hex[:12]
        gen = torch.Generator(device=self.device).manual_seed(seed)
        rec = self.recorder
        if trace:
            rec.begin_request(rid, meta)
        h2d0 = self.store.h2d_bytes()
        counts0 = {k: list(v) for k, v in self.store.counts.items()}
        cache = DynamicCache(config=self.config)
        ids = torch.tensor([list(input_ids)], device=self.device)
        self._sync()
        t0 = time.perf_counter()
        self.store.phase = "prefill"
        chunk = max(1, self.cfg.prefill_chunk)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            rec.begin_pass("extend", e - s, s)
            out = self.model(
                input_ids=ids[:, s:e],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            rec.end_pass()
        nxt = self._sample(out.logits[0, -1], temperature, top_p, gen)
        self._sync()
        t_first = time.perf_counter()
        output = [nxt]
        if on_token:
            on_token(nxt)
        self.store.phase = "decode"
        pos = n
        finish = "length"
        while len(output) < max_new_tokens:
            if not ignore_eos and nxt in self.stop_ids:
                finish = "stop"
                break
            rec.begin_pass("decode", 1, pos)
            out = self.model(
                input_ids=torch.tensor([[nxt]], device=self.device),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            rec.end_pass()
            pos += 1
            nxt = self._sample(out.logits[0, -1], temperature, top_p, gen)
            output.append(nxt)
            if on_token:
                on_token(nxt)
        if (
            finish == "length"
            and not ignore_eos
            and output
            and output[-1] in self.stop_ids
        ):
            finish = "stop"
        self._sync()
        t_end = time.perf_counter()
        stats = {
            "prompt_tokens": n,
            "completion_tokens": len(output),
            "finish_reason": finish,
            "ttft_s": t_first - t0,
            "decode_s": t_end - t_first,
            "decode_tok_per_s": (len(output) - 1) / (t_end - t_first)
            if len(output) > 1 and t_end > t_first
            else None,
            "h2d_bytes": self.store.h2d_bytes() - h2d0,
        }
        for ph, (h, mi) in self.store.counts.items():
            h0, m0 = counts0.get(ph, [0, 0])
            dh, dm = h - h0, mi - m0
            stats[f"{ph}_expert_hits"] = dh
            stats[f"{ph}_expert_misses"] = dm
            stats[f"{ph}_hit_rate"] = dh / (dh + dm) if dh + dm else None
        result = {"rid": rid, "output_ids": output, "stats": stats}
        if trace:
            result["trace"] = rec.end_request(
                {
                    "model": self.info.get("architectures"),
                    "model_dir": self.cfg.model_dir,
                    "engine": "moe_infinity.hy3",
                    "engine_config": asdict(self.cfg),
                    "first_k_dense_replace": self.ckpt.moe_layers[0],
                    "moe_layers": self.ckpt.moe_layers,
                    "num_experts": self.config.num_experts,
                    "expert_bytes": self.ckpt.expert_nbytes(),
                    "stats": stats,
                }
            )
        return result

    def memory_report(self) -> dict:
        rep = {
            "resident_gib": self.resident_bytes / 2**30,
            "expert_gib_each": self.ckpt.expert_nbytes() / 2**30,
            "experts_total": len(self.ckpt.moe_layers) * self.ckpt.num_experts,
            "expert_cache_gib": self.cfg.expert_cache_gib,
            "store": self.store.kind,
        }
        if self.device.type == "cuda":
            rep["torch_allocated_gib"] = (
                torch.cuda.memory_allocated(self.device) / 2**30
            )
            free, total = torch.cuda.mem_get_info(self.device)
            rep["device_free_gib"] = free / 2**30
            rep["device_total_gib"] = total / 2**30
        return rep
