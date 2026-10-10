# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Find and load one routed expert from a HuggingFace safetensors checkpoint.

Only the safetensors headers are parsed to discover experts, and only the
bytes of the selected ``(layer, expert)`` are read, so this works on
multi-hundred-GB checkpoints (e.g. Hy3-FP8) without loading shards.

Supported expert layouts:

* per-expert tensors: ``<...>.layers.L.<block>.experts.E.<proj>.weight``
  with ``proj`` in gate_proj/up_proj/down_proj, w1/w3/w2, or a fused
  gate_up_proj (Hy3, Qwen3-MoE, DeepSeek, Mixtral, OLMoE, ...);
* stacked tensors: ``<...>.layers.L.<block>.experts.gate_up_proj`` ``[E, ., .]``
  plus ``experts.down_proj`` (transformers v5 / converted checkpoints).

Supported weight formats: BF16/FP16/FP32, and FP8 (e4m3) with per-tensor,
per-output-channel or block scales (``weight_scale`` or ``weight_scale_inv``;
both are dequantization multipliers).  Static activation scales
(``input_scale``) are carried along when present.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ST_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}
FLOAT_WEIGHT_DTYPES = ("BF16", "F16", "F32")
FP8_DTYPES = ("F8_E4M3",)

GATE_NAMES = ("gate_proj", "w1")
UP_NAMES = ("up_proj", "w3")
DOWN_NAMES = ("down_proj", "w2")
GATE_UP_NAMES = ("gate_up_proj", "w13", "w13_weight")
WEIGHT_SCALE_PARAMS = (
    "weight_scale_inv",
    "weight_scale",
    "weight_scales",
    "scale",
)
INPUT_SCALE_PARAMS = ("input_scale", "activation_scale", "act_scale")
UNSUPPORTED_PARAMS = ("qweight", "qzeros", "g_idx", "weight_packed", "blocks")

_PER_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*?)\.?layers\.(?P<layer>\d+)\.(?P<block>(?:[\w]+\.)*?)"
    r"experts\.(?P<expert>\d+)\.(?P<proj>\w+)\.(?P<param>\w+)$"
)
_STACKED_RE = re.compile(
    r"^(?P<prefix>.*?)\.?layers\.(?P<layer>\d+)\.(?P<block>(?:[\w]+\.)*?)"
    r"experts\.(?P<proj>gate_up_proj|down_proj|gate_proj|up_proj|w13_weight"
    r"|w2_weight)(?:\.weight)?(?P<suffix>_scale_inv|_scale|_weight_scale"
    r"|_weight_scale_inv|_input_scale)?$"
)


class CheckpointError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# safetensors access (headers only, then exact byte ranges)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorRef:
    name: str
    path: Path
    dtype: str
    shape: Tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


class SafetensorsIndex:
    """Tensor-name -> shard map for a model directory."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise CheckpointError(f"model dir {self.model_dir} does not exist")
        self._headers: Dict[Path, Tuple[int, dict]] = {}
        index = self.model_dir / "model.safetensors.index.json"
        if index.exists():
            with open(index) as f:
                wm = json.load(f).get("weight_map")
            if not wm:
                raise CheckpointError(f"{index} has no weight_map")
            self.weight_map = {k: self.model_dir / v for k, v in wm.items()}
        else:
            shards = sorted(self.model_dir.glob("*.safetensors"))
            if not shards:
                raise CheckpointError(
                    f"no model.safetensors.index.json or *.safetensors in "
                    f"{self.model_dir}"
                )
            self.weight_map = {}
            for shard in shards:
                for k in self._header(shard)[1]:
                    if k != "__metadata__":
                        self.weight_map[k] = shard

    def names(self) -> List[str]:
        return list(self.weight_map)

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def _header(self, path: Path) -> Tuple[int, dict]:
        if path not in self._headers:
            if not path.exists():
                raise CheckpointError(f"shard {path} referenced but missing")
            with open(path, "rb") as f:
                raw = f.read(8)
                if len(raw) != 8:
                    raise CheckpointError(f"{path} is not a safetensors file")
                (n,) = struct.unpack("<Q", raw)
                hdr = json.loads(f.read(n))
            self._headers[path] = (8 + n, hdr)
        return self._headers[path]

    def ref(self, name: str) -> TensorRef:
        if name not in self.weight_map:
            raise CheckpointError(f"tensor {name!r} not in checkpoint")
        path = self.weight_map[name]
        base, hdr = self._header(path)
        if name not in hdr:
            raise CheckpointError(
                f"tensor {name!r} listed in index but missing from {path.name}"
            )
        info = hdr[name]
        b, e = info["data_offsets"]
        return TensorRef(
            name, path, info["dtype"], tuple(info["shape"]), base + b, base + e
        )

    def load(self, name: str) -> torch.Tensor:
        r = self.ref(name)
        if r.dtype not in ST_DTYPES:
            raise CheckpointError(f"{name}: unsupported dtype {r.dtype}")
        with open(r.path, "rb") as f:
            f.seek(r.begin)
            buf = bytearray(f.read(r.nbytes))
        if len(buf) != r.nbytes:
            raise CheckpointError(f"{name}: short read from {r.path}")
        dtype = ST_DTYPES[r.dtype]
        if r.nbytes == 0:
            return torch.empty(r.shape, dtype=dtype)
        t = torch.frombuffer(buf, dtype=torch.uint8).view(dtype)
        return t.reshape(r.shape).clone()

    def load_row(self, name: str, i: int) -> torch.Tensor:
        """``tensor[i]`` of a stacked tensor, reading only that slice."""
        r = self.ref(name)
        if not r.shape or not 0 <= i < r.shape[0]:
            raise CheckpointError(f"{name}: index {i} out of range {r.shape}")
        dtype = ST_DTYPES[r.dtype]
        row = r.nbytes // r.shape[0]
        with open(r.path, "rb") as f:
            f.seek(r.begin + i * row)
            buf = bytearray(f.read(row))
        if len(buf) != row:
            raise CheckpointError(f"{name}: short read from {r.path}")
        t = torch.frombuffer(buf, dtype=torch.uint8).view(dtype)
        return t.reshape(r.shape[1:]).clone()


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def load_config(model_dir: str | Path) -> dict:
    path = Path(model_dir) / "config.json"
    if not path.exists():
        raise CheckpointError(f"{path} not found")
    with open(path) as f:
        cfg = json.load(f)
    # Multimodal wrappers keep the LM config nested.
    for key in ("text_config", "language_config", "llm_config"):
        if isinstance(cfg.get(key), dict) and "hidden_size" in cfg[key]:
            merged = dict(cfg[key])
            merged.setdefault(
                "quantization_config", cfg.get("quantization_config")
            )
            merged.setdefault("architectures", cfg.get("architectures"))
            return merged
    return cfg


def config_summary(cfg: dict) -> dict:
    def first(*keys):
        for k in keys:
            if cfg.get(k) is not None:
                return cfg[k]
        return None

    q = cfg.get("quantization_config") or {}
    return {
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "hidden_size": cfg.get("hidden_size"),
        "moe_intermediate_size": first(
            "moe_intermediate_size",
            "expert_hidden_dim",
            "expert_intermediate_size",
        ),
        "num_experts": first(
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
            "moe_num_experts",
        ),
        "num_experts_per_tok": first(
            "num_experts_per_tok", "moe_topk", "top_k"
        ),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "first_k_dense_replace": cfg.get("first_k_dense_replace"),
        "hidden_act": cfg.get("hidden_act", "silu"),
        "dtype": first("dtype", "torch_dtype") or "bfloat16",
        "quant_method": q.get("quant_method"),
        "activation_scheme": q.get("activation_scheme"),
        "weight_block_size": q.get("weight_block_size"),
    }


# --------------------------------------------------------------------------
# expert discovery
# --------------------------------------------------------------------------


@dataclass
class LayerExperts:
    layer: int
    kind: str  # "per_expert" | "stacked"
    base: str  # name prefix up to and including "experts."
    expert_ids: List[int] = field(default_factory=list)
    projs: List[str] = field(default_factory=list)
    num_stacked: Optional[int] = None

    @property
    def num_experts(self) -> int:
        if self.kind == "stacked":
            return int(self.num_stacked or 0)
        return len(self.expert_ids)


def discover_experts(index: SafetensorsIndex) -> Dict[int, LayerExperts]:
    """Map layer id -> routed experts found in the tensor names."""
    layers: Dict[int, LayerExperts] = {}
    for name in index.names():
        m = _PER_EXPERT_RE.match(name)
        if m and m.group("param") != "bias":
            L = int(m.group("layer"))
            base = name[: m.start("expert")]
            le = layers.setdefault(L, LayerExperts(L, "per_expert", base))
            if le.kind != "per_expert" or le.base != base:
                continue
            e = int(m.group("expert"))
            if not le.expert_ids or le.expert_ids[-1] != e:
                le.expert_ids.append(e)
            if m.group("proj") not in le.projs:
                le.projs.append(m.group("proj"))
            continue
        m = _STACKED_RE.match(name)
        if m and not m.group("suffix"):
            L = int(m.group("layer"))
            base = name[: m.start("proj")]
            le = layers.setdefault(L, LayerExperts(L, "stacked", base))
            if le.kind != "stacked":
                continue
            if m.group("proj") not in le.projs:
                le.projs.append(m.group("proj"))
    for le in layers.values():
        le.expert_ids = sorted(set(le.expert_ids))
    for le in layers.values():
        if le.kind == "stacked":
            name = _stacked_weight_name(
                index, le, ("gate_up_proj", "w13_weight", "gate_proj")
            )
            if name is not None:
                le.num_stacked = index.ref(name).shape[0]
    return dict(sorted(layers.items()))


def _stacked_weight_name(index, le: LayerExperts, projs) -> Optional[str]:
    for p in projs:
        for cand in (f"{le.base}{p}", f"{le.base}{p}.weight"):
            if cand in index:
                return cand
    return None


# --------------------------------------------------------------------------
# loaded expert
# --------------------------------------------------------------------------


@dataclass
class ProjWeight:
    """One linear of an expert: ``y = x @ W^T`` with ``W`` ``[out, in]``."""

    role: str
    weight: torch.Tensor
    scheme: str = "none"  # none | per_tensor | per_channel | block
    scale: Optional[torch.Tensor] = None  # float32 dequant multiplier
    block: Optional[Tuple[int, int]] = None
    input_scale: Optional[torch.Tensor] = None  # float32 scalar
    names: List[str] = field(default_factory=list)

    @property
    def is_fp8(self) -> bool:
        return self.weight.dtype == torch.float8_e4m3fn

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    def scale_grid(self, block: Tuple[int, int]) -> torch.Tensor:
        """Scales broadcast to a ``[ceil(out/bn), ceil(in/bk)]`` block grid."""
        bn, bk = block
        nb_o = math.ceil(self.out_features / bn)
        nb_i = math.ceil(self.in_features / bk)
        if self.scheme == "per_tensor":
            return self.scale.reshape(1, 1).expand(nb_o, nb_i).contiguous()
        if self.scheme == "block" and tuple(self.block) == tuple(block):
            return self.scale
        raise CheckpointError(
            f"{self.role}: {self.scheme} scales cannot be expressed as "
            f"{block} block scales"
        )

    def dequant(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        w = self.weight.to(torch.float32)
        if self.scheme == "per_tensor":
            w = w * self.scale.reshape(())
        elif self.scheme == "per_channel":
            w = w * self.scale.reshape(-1, 1)
        elif self.scheme == "block":
            bn, bk = self.block
            s = self.scale.repeat_interleave(bn, 0)[: w.shape[0]]
            s = s.repeat_interleave(bk, 1)[:, : w.shape[1]]
            w = w * s
        return w.to(dtype)

    def slice_rows(self, start: int, stop: int, role: str) -> "ProjWeight":
        scale = self.scale
        if self.scheme == "per_channel":
            scale = self.scale[start:stop]
        elif self.scheme == "block":
            bn = self.block[0]
            if start % bn or stop % bn:
                raise CheckpointError(
                    f"fused gate_up rows {start}:{stop} not aligned to the "
                    f"{bn}-row scale blocks"
                )
            scale = self.scale[start // bn : stop // bn]
        return ProjWeight(
            role,
            self.weight[start:stop].contiguous(),
            self.scheme,
            scale,
            self.block,
            self.input_scale,
            list(self.names),
        )


@dataclass
class ExpertWeights:
    layer: int
    expert: int
    gate: ProjWeight
    up: ProjWeight
    down: ProjWeight
    activation: str = "silu"
    act_dtype: torch.dtype = torch.bfloat16
    layout: str = "per_expert"

    @property
    def hidden_size(self) -> int:
        return self.gate.in_features

    @property
    def intermediate_size(self) -> int:
        return self.gate.out_features

    @property
    def projs(self) -> Tuple[ProjWeight, ProjWeight, ProjWeight]:
        return (self.gate, self.up, self.down)

    @property
    def quant(self) -> str:
        g = self.gate
        if not g.is_fp8:
            return str(g.weight.dtype).replace("torch.", "")
        return f"fp8_{g.scheme}"

    @property
    def checkpoint_nbytes(self) -> int:
        n = 0
        for p in self.projs:
            n += p.weight.numel() * p.weight.element_size()
            if p.scale is not None:
                n += p.scale.numel() * 4
        return n

    def static_input_scales(self) -> Tuple[Optional[float], Optional[float]]:
        """(a1, a2): max of gate/up input scales and the down input scale."""
        a1 = [
            p.input_scale
            for p in (self.gate, self.up)
            if p.input_scale is not None
        ]
        a1v = max(float(s.max()) for s in a1) if a1 else None
        a2v = (
            float(self.down.input_scale.max())
            if self.down.input_scale is not None
            else None
        )
        return a1v, a2v

    def describe(self) -> dict:
        a1, a2 = self.static_input_scales()
        d = {
            "layer": self.layer,
            "expert": self.expert,
            "layout": self.layout,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "quant": self.quant,
            "activation": self.activation,
            "act_dtype": str(self.act_dtype).replace("torch.", ""),
            "checkpoint_bytes": self.checkpoint_nbytes,
            "static_input_scale_gate_up": a1,
            "static_input_scale_down": a2,
            "projs": {},
        }
        for p in self.projs:
            d["projs"][p.role] = {
                "shape": list(p.weight.shape),
                "dtype": str(p.weight.dtype).replace("torch.", ""),
                "scheme": p.scheme,
                "scale_shape": None if p.scale is None else list(p.scale.shape),
                "scale_value": (
                    float(p.scale.reshape(()))
                    if p.scheme == "per_tensor"
                    else None
                ),
                "block": list(p.block) if p.block else None,
                "input_scale": (
                    None
                    if p.input_scale is None
                    else float(p.input_scale.max())
                ),
                "tensors": p.names,
            }
        return d


_TORCH_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "float": torch.float32,
}


def _infer_scheme(
    role: str,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    cfg_block: Optional[List[int]],
) -> Tuple[str, Optional[torch.Tensor], Optional[Tuple[int, int]]]:
    if weight.dtype != torch.float8_e4m3fn:
        if scale is not None and weight.dtype in (
            torch.int8,
            torch.uint8,
            torch.int32,
        ):
            raise CheckpointError(
                f"{role}: integer-packed weights ({weight.dtype}, e.g. "
                f"GPTQ/AWQ/MXFP4) are not supported by this benchmark"
            )
        if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise CheckpointError(
                f"{role}: unsupported weight dtype {weight.dtype}"
            )
        return "none", None, None
    if scale is None:
        raise CheckpointError(
            f"{role}: FP8 weight without a weight_scale / weight_scale_inv tensor"
        )
    s = scale.to(torch.float32)
    O, I = weight.shape
    if s.numel() == 1:
        return "per_tensor", s.reshape(()), None
    if tuple(s.shape) in ((O,), (O, 1)):
        return "per_channel", s.reshape(O), None
    if s.dim() == 2:
        cands = []
        if cfg_block:
            cands.append(tuple(int(b) for b in cfg_block))
        for b in (128, 64, 256, 32, 512):
            cands.append((b, b))
        for bn, bk in cands:
            if (math.ceil(O / bn), math.ceil(I / bk)) == tuple(s.shape):
                return "block", s.contiguous(), (bn, bk)
    raise CheckpointError(
        f"{role}: cannot classify weight scale of shape {tuple(s.shape)} for "
        f"weight {tuple(weight.shape)} (expected scalar, [out], or a block grid "
        f"matching quantization_config.weight_block_size={cfg_block})"
    )


def _find_param(index, stem: str, params) -> Optional[str]:
    for p in params:
        if f"{stem}.{p}" in index:
            return f"{stem}.{p}"
    return None


def _load_per_expert_proj(index, stem, role, cfg_block) -> ProjWeight:
    wname = f"{stem}.weight"
    if wname not in index:
        bad = _find_param(index, stem, UNSUPPORTED_PARAMS)
        if bad:
            raise CheckpointError(
                f"{bad}: packed/quantized-int expert weights are not supported"
            )
        raise CheckpointError(f"missing {wname}")
    w = index.load(wname)
    if w.dim() != 2:
        raise CheckpointError(
            f"{wname}: expected 2-D weight, got {tuple(w.shape)}"
        )
    names = [wname]
    sname = _find_param(index, stem, WEIGHT_SCALE_PARAMS)
    scale = None
    if sname:
        scale = index.load(sname)
        names.append(sname)
    scheme, scale, block = _infer_scheme(wname, w, scale, cfg_block)
    iname = _find_param(index, stem, INPUT_SCALE_PARAMS)
    inp = None
    if iname:
        inp = index.load(iname).to(torch.float32).reshape(-1)
        names.append(iname)
    return ProjWeight(role, w, scheme, scale, block, inp, names)


def _orient(
    t: torch.Tensor, out_dim: int, in_dim: int, name: str
) -> torch.Tensor:
    """Return ``t`` as ``[out, in]`` (stacked checkpoints store either way)."""
    if tuple(t.shape) == (out_dim, in_dim):
        return t
    if tuple(t.shape) == (in_dim, out_dim):
        return t.t().contiguous()
    raise CheckpointError(
        f"{name}: shape {tuple(t.shape)} does not match ({out_dim}, {in_dim}) "
        f"in either orientation"
    )


def _load_stacked(index, le: LayerExperts, expert: int, hidden: int, cfg_block):
    gu = _stacked_weight_name(index, le, ("gate_up_proj", "w13_weight"))
    dn = _stacked_weight_name(index, le, ("down_proj", "w2_weight"))
    if dn is None:
        raise CheckpointError(f"layer {le.layer}: stacked down_proj not found")
    E = le.num_experts
    if not 0 <= expert < E:
        raise CheckpointError(f"expert {expert} out of range (layer has {E})")

    def scale_for(wname):
        stem = wname[:-7] if wname.endswith(".weight") else wname
        for suf in (
            "_scale_inv",
            "_scale",
            "_weight_scale_inv",
            "_weight_scale",
        ):
            for cand in (f"{stem}{suf}", f"{wname}{suf}"):
                if cand in index:
                    return cand
        for p in WEIGHT_SCALE_PARAMS:
            if f"{stem}.{p}" in index:
                return f"{stem}.{p}"
        return None

    def proj(wname, out_dim, in_dim, role):
        raw = index.load_row(wname, expert)
        w = _orient(raw, out_dim, in_dim, wname)
        transposed = tuple(raw.shape) != tuple(w.shape)
        names = [wname]
        s = None
        sname = scale_for(wname)
        if sname:
            sref = index.ref(sname)
            if sref.shape and sref.shape[0] == E and len(sref.shape) >= 1:
                s = index.load_row(sname, expert)
            else:
                s = index.load(sname)
            if transposed and s.dim() == 2 and s.numel() > 1:
                s = s.t().contiguous()
            names.append(sname)
        scheme, s, block = _infer_scheme(wname, w, s, cfg_block)
        return ProjWeight(role, w, scheme, s, block, None, names)

    d_raw = index.ref(dn).shape[1:]
    if hidden not in d_raw:
        raise CheckpointError(
            f"{dn}: shape {d_raw} has no hidden_size={hidden} dim"
        )
    inter = d_raw[0] if d_raw[1] == hidden else d_raw[1]
    down = proj(dn, hidden, inter, "down")
    if gu is not None:
        fused = proj(gu, 2 * inter, hidden, "gate_up")
        gate = fused.slice_rows(0, inter, "gate")
        up = fused.slice_rows(inter, 2 * inter, "up")
    else:
        g = _stacked_weight_name(index, le, ("gate_proj",))
        u = _stacked_weight_name(index, le, ("up_proj",))
        if g is None or u is None:
            raise CheckpointError(
                f"layer {le.layer}: stacked gate/up not found"
            )
        gate = proj(g, inter, hidden, "gate")
        up = proj(u, inter, hidden, "up")
    return gate, up, down


def load_expert(
    index: SafetensorsIndex,
    cfg: dict,
    layer: int,
    expert: int,
    layers: Optional[Dict[int, LayerExperts]] = None,
) -> ExpertWeights:
    layers = layers if layers is not None else discover_experts(index)
    if not layers:
        raise CheckpointError(
            "no routed experts found: expected tensor names like "
            "'model.layers.L.mlp.experts.E.gate_proj.weight' or stacked "
            "'model.layers.L.mlp.experts.gate_up_proj'. Run --list-experts to "
            "see what the checkpoint contains."
        )
    if layer not in layers:
        raise CheckpointError(
            f"layer {layer} has no routed experts; MoE layers: "
            f"{_compress(sorted(layers))}"
        )
    le = layers[layer]
    summary = config_summary(cfg)
    cfg_block = summary["weight_block_size"]
    hidden = summary["hidden_size"]
    if le.kind == "per_expert":
        if expert not in le.expert_ids:
            raise CheckpointError(
                f"layer {layer} has experts {_compress(le.expert_ids)}; "
                f"expert {expert} not found"
            )
        stem = f"{le.base}{expert}"
        projs = {p: f"{stem}.{p}" for p in le.projs}
        gname = next((projs[p] for p in GATE_NAMES if p in projs), None)
        uname = next((projs[p] for p in UP_NAMES if p in projs), None)
        dname = next((projs[p] for p in DOWN_NAMES if p in projs), None)
        guname = next((projs[p] for p in GATE_UP_NAMES if p in projs), None)
        if dname is None or (
            guname is None and (gname is None or uname is None)
        ):
            raise CheckpointError(
                f"layer {layer} expert {expert}: projections {le.projs} do not "
                f"form gate/up/down (expected gate_proj/up_proj/down_proj, "
                f"w1/w3/w2, or gate_up_proj + down_proj)"
            )
        down = _load_per_expert_proj(index, dname, "down", cfg_block)
        if guname is not None:
            fused = _load_per_expert_proj(index, guname, "gate_up", cfg_block)
            n = fused.out_features // 2
            gate = fused.slice_rows(0, n, "gate")
            up = fused.slice_rows(n, 2 * n, "up")
        else:
            gate = _load_per_expert_proj(index, gname, "gate", cfg_block)
            up = _load_per_expert_proj(index, uname, "up", cfg_block)
    else:
        if hidden is None:
            raise CheckpointError("config.json has no hidden_size")
        gate, up, down = _load_stacked(index, le, expert, hidden, cfg_block)

    N, K = gate.weight.shape
    if up.weight.shape != (N, K) or down.weight.shape != (K, N):
        raise CheckpointError(
            f"inconsistent expert shapes: gate {tuple(gate.weight.shape)}, up "
            f"{tuple(up.weight.shape)}, down {tuple(down.weight.shape)}"
        )
    if hidden is not None and K != hidden:
        raise CheckpointError(
            f"expert input dim {K} != config hidden_size {hidden}"
        )
    if (
        len({p.weight.dtype for p in (gate, up, down)}) != 1
        or len({p.scheme for p in (gate, up, down)}) != 1
    ):
        raise CheckpointError(
            "gate/up/down use different dtypes or scale schemes: "
            + ", ".join(
                f"{p.role}={p.weight.dtype}/{p.scheme}"
                for p in (gate, up, down)
            )
        )
    act = str(summary["hidden_act"] or "silu").lower()
    if act in ("swiglu", "silu_and_mul"):
        act = "silu"
    if act not in ("silu", "gelu"):
        raise CheckpointError(f"unsupported expert activation {act!r}")
    act_dtype = _TORCH_DTYPE.get(str(summary["dtype"]).lower(), torch.bfloat16)
    if act_dtype == torch.float32:
        act_dtype = torch.bfloat16
    return ExpertWeights(
        layer, expert, gate, up, down, act, act_dtype, layout=le.kind
    )


def _compress(ids: List[int]) -> str:
    """[0,1,2,5,7,8] -> '0-2,5,7-8'."""
    if not ids:
        return "-"
    out, start, prev = [], ids[0], ids[0]
    for i in ids[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ",".join(out)


def inspect_checkpoint(model_dir: str | Path, layer=None, expert=None) -> dict:
    """Everything ``--list-experts`` prints, as a dict."""
    index = SafetensorsIndex(model_dir)
    cfg = load_config(model_dir)
    layers = discover_experts(index)
    info = {
        "model_dir": str(model_dir),
        "num_tensors": len(index.names()),
        "config": config_summary(cfg),
        "moe_layers": _compress(sorted(layers)),
        "layers": {
            L: {
                "kind": le.kind,
                "base": le.base,
                "num_experts": le.num_experts,
                "experts": _compress(le.expert_ids) if le.expert_ids else None,
                "projs": le.projs,
            }
            for L, le in layers.items()
        },
    }
    if layers:
        L = layer if layer is not None else next(iter(layers))
        E = expert if expert is not None else 0
        info["selected"] = load_expert(index, cfg, L, E, layers).describe()
    return info
