# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Hy3-FP8 safetensors checkpoint access.

Hy3-FP8 stores every linear layer except ``lm_head`` and ``embed_tokens``
as per-tensor FP8 e4m3: ``<x>.weight`` (F8_E4M3), ``<x>.weight_scale``
(BF16 scalar, dequantization multiplier) and ``<x>.input_scale`` (F32
``[1]``, static activation scale; ``activation_scheme: static``).  Routed
experts are per-expert tensors
``model.layers.L.mlp.experts.E.{gate,up,down}_proj.*``.  Layer
``num_hidden_layers`` (80) is the MTP layer and is not used.

Checkpoint names differ from transformers' ``HYV3`` modules; the same
renames as transformers' ``conversion_mapping["hy_v3"]`` are applied.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from safetensors import safe_open

FP8_DTYPES = (torch.float8_e4m3fn,)
PROJS = ("gate_proj", "up_proj", "down_proj")

EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale|input_scale)$"
)
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")

# transformers conversion_mapping["hy_v3"] (minus the qwen2_moe expert
# fusion, which does not apply: routed experts never become module params).
_RENAMES = (
    (re.compile(r"mlp\.router\.gate\.weight$"), "mlp.gate.weight"),
    (re.compile(r"mlp\.expert_bias$"), "mlp.e_score_correction_bias"),
    (re.compile(r"mlp\.shared_mlp\."), "mlp.shared_experts."),
)
_AUX_SUFFIXES = (".weight_scale", ".input_scale", ".k_scale", ".v_scale")


def hf_name(name: str) -> str:
    for pat, repl in _RENAMES:
        name = pat.sub(repl, name)
    return name


def dequant_per_tensor(
    weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    if scale.numel() != 1:
        raise ValueError(
            f"expected a per-tensor scale, got shape {tuple(scale.shape)}"
        )
    s = scale.to(device=weight.device, dtype=torch.float32).reshape(())
    return (weight.to(torch.float32) * s).to(dtype)


@dataclass
class ExpertScales:
    """Per (layer, expert): weight scales and static input scales."""

    # [n_moe_layers, num_experts, 3] (gate, up, down), float32
    weight: torch.Tensor
    # [n_moe_layers, num_experts, 3] (gate, up, down), float32; 0 = dynamic
    input: torch.Tensor


class Hy3Checkpoint:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.config_dict = json.loads(
            (self.model_dir / "config.json").read_text()
        )
        index = self.model_dir / "model.safetensors.index.json"
        if index.exists():
            wm = json.loads(index.read_text())["weight_map"]
            self.weight_map = {k: self.model_dir / v for k, v in wm.items()}
        else:
            self.weight_map = {}
            for shard in sorted(self.model_dir.glob("*.safetensors")):
                with safe_open(str(shard), framework="pt") as f:
                    for k in f.keys():
                        self.weight_map[k] = shard
        if not self.weight_map:
            raise FileNotFoundError(f"no safetensors in {self.model_dir}")
        cfg = self.config_dict
        self.num_hidden_layers = int(cfg["num_hidden_layers"])
        self.num_experts = int(cfg["num_experts"])
        qc = cfg.get("quantization_config") or {}
        self.quant_method = str(qc.get("quant_method", "")).lower()
        self.activation_scheme = str(qc.get("activation_scheme", ""))
        self.moe_layers = self._discover_moe_layers()
        self._files: Dict[Path, object] = {}

    # -- discovery --------------------------------------------------------

    def _discover_moe_layers(self) -> List[int]:
        layers = set()
        for k in self.weight_map:
            m = EXPERT_RE.match(k)
            if m and int(m.group(1)) < self.num_hidden_layers:
                layers.add(int(m.group(1)))
        return sorted(layers)

    def fingerprint(self) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(json.dumps(self.config_dict, sort_keys=True).encode())
        for k in sorted(self.weight_map):
            if EXPERT_RE.match(k):
                h.update(k.encode())
                h.update(self.weight_map[k].name.encode())
        return h.hexdigest()[:16]

    def expert_key(self, layer: int, expert: int, proj: str, kind: str):
        return f"model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"

    # -- reading ----------------------------------------------------------

    def _open(self, path: Path):
        f = self._files.get(path)
        if f is None:
            f = safe_open(str(path), framework="pt", device="cpu")
            self._files[path] = f
        return f

    def get(self, name: str) -> torch.Tensor:
        return self._open(self.weight_map[name]).get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def resident_names(self) -> List[str]:
        """Checkpoint weights that become module parameters/buffers."""
        out = []
        for k in self.weight_map:
            if EXPERT_RE.match(k) or k.endswith(_AUX_SUFFIXES):
                continue
            m = LAYER_RE.match(k)
            if m and int(m.group(1)) >= self.num_hidden_layers:
                continue  # MTP layer
            out.append(k)
        return sorted(out)

    def iter_resident(
        self, dtype: torch.dtype, device: torch.device | str = "cpu"
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """(HF name, tensor) for every non-expert weight; FP8 dequantized."""
        by_file = defaultdict(list)
        for k in self.resident_names():
            by_file[self.weight_map[k]].append(k)
        for path in sorted(by_file):
            for k in by_file[path]:
                t = self.get(k)
                if t.dtype in FP8_DTYPES:
                    sk = k[: -len(".weight")] + ".weight_scale"
                    if not k.endswith(".weight") or not self.has(sk):
                        raise ValueError(f"FP8 tensor {k} has no weight_scale")
                    t = dequant_per_tensor(t.to(device), self.get(sk), dtype)
                elif t.is_floating_point():
                    # HYV3 keeps e_score_correction_bias in fp32.
                    keep_fp32 = k.endswith("expert_bias")
                    t = t.to(device, torch.float32 if keep_fp32 else dtype)
                yield hf_name(k), t

    def expert_scales(self) -> ExpertScales:
        L, E = len(self.moe_layers), self.num_experts
        w = torch.ones(L, E, 3, dtype=torch.float32)
        a = torch.zeros(L, E, 3, dtype=torch.float32)
        for li, layer in enumerate(self.moe_layers):
            for e in range(E):
                for p, proj in enumerate(PROJS):
                    ws = self.expert_key(layer, e, proj, "weight_scale")
                    if self.has(ws):
                        w[li, e, p] = float(self.get(ws).float().reshape(()))
                    ins = self.expert_key(layer, e, proj, "input_scale")
                    if self.has(ins):
                        a[li, e, p] = float(
                            self.get(ins).float().reshape(-1)[0]
                        )
        return ExpertScales(w, a)

    def expert_weight_names(self) -> List[Tuple[int, int, int, str]]:
        """(moe layer idx, expert, proj idx, name) in shard/file order."""
        li_of = {layer: i for i, layer in enumerate(self.moe_layers)}
        out = []
        for k, path in self.weight_map.items():
            m = EXPERT_RE.match(k)
            if not m or m.group(4) != "weight":
                continue
            layer = int(m.group(1))
            if layer not in li_of:
                continue
            out.append(
                (
                    path,
                    li_of[layer],
                    int(m.group(2)),
                    PROJS.index(m.group(3)),
                    k,
                )
            )
        out.sort(key=lambda r: (str(r[0]), r[4]))
        return [r[1:] for r in out]

    def expert_dtype(self) -> torch.dtype:
        layer = self.moe_layers[0]
        return self.get(self.expert_key(layer, 0, "gate_proj", "weight")).dtype

    def expert_shapes(self) -> Tuple[Tuple[int, int], ...]:
        layer = self.moe_layers[0]
        shapes = []
        for proj in PROJS:
            path = self.weight_map[self.expert_key(layer, 0, proj, "weight")]
            with safe_open(str(path), framework="pt") as f:
                shapes.append(
                    tuple(
                        f.get_slice(
                            self.expert_key(layer, 0, proj, "weight")
                        ).get_shape()
                    )
                )
        return tuple(shapes)

    def expert_nbytes(self) -> int:
        es = torch.tensor([], dtype=self.expert_dtype()).element_size()
        return sum(r * c * es for r, c in self.expert_shapes())

    def close(self) -> None:
        self._files.clear()


def describe(ckpt: Hy3Checkpoint, scales: Optional[ExpertScales] = None):
    cfg = ckpt.config_dict
    info = {
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "num_hidden_layers": ckpt.num_hidden_layers,
        "moe_layers": [ckpt.moe_layers[0], ckpt.moe_layers[-1]]
        if ckpt.moe_layers
        else [],
        "n_moe_layers": len(ckpt.moe_layers),
        "num_experts": ckpt.num_experts,
        "top_k": cfg.get("num_experts_per_tok"),
        "quant_method": ckpt.quant_method,
        "activation_scheme": ckpt.activation_scheme,
        "expert_dtype": str(ckpt.expert_dtype()),
        "expert_shapes": ckpt.expert_shapes(),
        "expert_mib": ckpt.expert_nbytes() / 2**20,
        "all_experts_gib": ckpt.expert_nbytes()
        * len(ckpt.moe_layers)
        * ckpt.num_experts
        / 2**30,
    }
    if scales is not None:
        info["static_input_scales"] = bool((scales.input > 0).all())
    return info
