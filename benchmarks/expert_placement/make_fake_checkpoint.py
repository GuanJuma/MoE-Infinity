# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Write a tiny synthetic MoE checkpoint for smoke-testing the benchmark.

``--format hy3_fp8`` mirrors Tencent Hy3-FP8's layout exactly (checked
against its model.safetensors.index.json and shard headers): per-expert
``model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight`` in F8_E4M3 with
a scalar BF16 ``weight_scale`` and an F32 ``[1]`` ``input_scale``, plus
``mlp.router.gate``, ``mlp.shared_mlp.*`` and ``mlp.expert_bias``;
``quantization_config = {quant_method: fp8, activation_scheme: static}``;
layer 0 dense (``first_k_dense_replace=1``).  Other formats:
``block_fp8`` (DeepSeek-V3 style ``weight_scale_inv`` 128x128 blocks),
``bf16``, ``stacked_bf16`` (transformers-v5 fused ``experts.gate_up_proj``).

    python make_fake_checkpoint.py /tmp/fake-hy3 --format hy3_fp8
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import torch

FP8_MAX = 448.0
_ST = {
    torch.float8_e4m3fn: "F8_E4M3",
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
}


def save_safetensors(path: Path, tensors: dict):
    header, blobs, off = {}, [], 0
    for name, t in tensors.items():
        t = t.contiguous()
        raw = (
            t.reshape(-1).view(torch.uint8).numpy().tobytes()
            if t.numel()
            else b""
        )
        header[name] = {
            "dtype": _ST[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [off, off + len(raw)],
        }
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def _fp8_per_tensor(w):
    s = w.abs().amax().clamp(min=1e-8) / FP8_MAX
    return (w / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s


def _fp8_block(w, b=128):
    O, I = w.shape
    no, ni = math.ceil(O / b), math.ceil(I / b)
    pad = torch.zeros(no * b, ni * b)
    pad[:O, :I] = w
    blk = pad.view(no, b, ni, b)
    s = blk.abs().amax(dim=(1, 3)).clamp(min=1e-8) / FP8_MAX
    q = (
        (blk / s[:, None, :, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .view(no * b, ni * b)[:O, :I]
    )
    return q.to(torch.float8_e4m3fn).contiguous(), s


def make(
    out: Path,
    fmt: str = "hy3_fp8",
    hidden: int = 256,
    inter: int = 128,
    experts: int = 4,
    layers: int = 3,
    seed: int = 0,
) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, fan_in):
        return torch.randn(*shape, generator=g) / fan_in**0.5

    shards = [{}, {}]
    first_dense = 1
    for L in range(layers):
        p = f"model.layers.{L}"
        shards[0][f"{p}.input_layernorm.weight"] = torch.ones(
            hidden, dtype=torch.bfloat16
        )
        if L < first_dense:
            for n, shape in (
                ("gate_proj", (inter * 2, hidden)),
                ("up_proj", (inter * 2, hidden)),
                ("down_proj", (hidden, inter * 2)),
            ):
                shards[0][f"{p}.mlp.{n}.weight"] = rnd(
                    *shape, fan_in=shape[1]
                ).to(torch.bfloat16)
            continue
        shards[0][f"{p}.mlp.router.gate.weight"] = rnd(
            experts, hidden, fan_in=hidden
        ).to(torch.bfloat16)
        shards[0][f"{p}.mlp.expert_bias"] = torch.zeros(
            experts, dtype=torch.float32
        )
        if fmt == "stacked_bf16":
            gu = rnd(experts, 2 * inter, hidden, fan_in=hidden)
            dn = rnd(experts, hidden, inter, fan_in=inter)
            shards[1][f"{p}.mlp.experts.gate_up_proj"] = gu.to(torch.bfloat16)
            shards[1][f"{p}.mlp.experts.down_proj"] = dn.to(torch.bfloat16)
            continue
        for e in range(experts):
            stem = f"{p}.mlp.experts.{e}"
            shard = shards[e % 2]
            for n, shape in (
                ("gate_proj", (inter, hidden)),
                ("up_proj", (inter, hidden)),
                ("down_proj", (hidden, inter)),
            ):
                w = rnd(*shape, fan_in=shape[1])
                if fmt == "hy3_fp8":
                    q, s = _fp8_per_tensor(w)
                    shard[f"{stem}.{n}.weight"] = q
                    shard[f"{stem}.{n}.weight_scale"] = s.to(
                        torch.bfloat16
                    ).reshape(())
                    # Hy3 keeps every input_scale in the last shard.
                    shards[1][f"{stem}.{n}.input_scale"] = torch.tensor(
                        [0.0021], dtype=torch.float32
                    )
                elif fmt == "block_fp8":
                    q, s = _fp8_block(w)
                    shard[f"{stem}.{n}.weight"] = q
                    shard[f"{stem}.{n}.weight_scale_inv"] = s.to(torch.float32)
                elif fmt == "bf16":
                    shard[f"{stem}.{n}.weight"] = w.to(torch.bfloat16)
                else:
                    raise ValueError(f"unknown format {fmt!r}")
    weight_map = {}
    for i, sh in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-00002.safetensors"
        save_safetensors(out / fname, sh)
        weight_map.update({k: fname for k in sh})
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)

    cfg = {
        "architectures": ["HYV3ForCausalLM"],
        "model_type": "hy_v3",
        "dtype": "bfloat16",
        "hidden_size": hidden,
        "moe_intermediate_size": inter,
        "expert_hidden_dim": inter,
        "intermediate_size": inter * 2,
        "num_experts": experts,
        "num_experts_per_tok": 2,
        "num_shared_experts": 0,
        "num_hidden_layers": layers,
        "first_k_dense_replace": first_dense,
        "hidden_act": "silu",
        "_fake_checkpoint_format": fmt,
    }
    if fmt == "hy3_fp8":
        cfg["quantization_config"] = {
            "activation_scheme": "static",
            "ignored_layers": ["lm_head", "model.embed_tokens"],
            "quant_method": "fp8",
            "kv_cache_scheme": "static",
        }
    elif fmt == "block_fp8":
        cfg["quantization_config"] = {
            "activation_scheme": "dynamic",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument(
        "--format",
        default="hy3_fp8",
        choices=("hy3_fp8", "block_fp8", "bf16", "stacked_bf16"),
    )
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--inter", type=int, default=128)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    a = p.parse_args()
    print(make(Path(a.out), a.format, a.hidden, a.inter, a.experts, a.layers))


if __name__ == "__main__":
    main()
