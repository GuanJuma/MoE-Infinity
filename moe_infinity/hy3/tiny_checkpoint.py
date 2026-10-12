# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""A tiny random Hy3-FP8 checkpoint with the real checkpoint's layout.

Same tensor names and formats as Tencent Hy3-FP8: per-expert
``mlp.experts.E.{gate,up,down}_proj``, ``mlp.router.gate``,
``mlp.expert_bias``, ``mlp.shared_mlp``; every linear except ``lm_head``
and ``embed_tokens`` is per-tensor FP8 with a BF16 ``weight_scale`` and a
calibrated F32 ``[1]`` ``input_scale`` (kept in the last shard); attention
``k_scale``/``v_scale``; an MTP layer after the last decoder layer; a
byte-level tokenizer with a chat template.  Used by the tests and for
CPU dry runs::

    python -m moe_infinity.hy3.tiny_checkpoint /tmp/tiny-hy3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file

FP8_MAX = 448.0
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _fp8(w: torch.Tensor):
    s = w.abs().amax().clamp(min=1e-8) / FP8_MAX
    s = s.to(torch.bfloat16).float()
    q = (w / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, s.to(torch.bfloat16).reshape(())


def tiny_config(**over):
    from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config

    kw = dict(
        vocab_size=320,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=32,
        max_position_embeddings=4096,
        initializer_range=0.2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    kw.update(over)
    return HYV3Config(**kw)


def _calibrate(model, config, n_tokens: int = 256, seed: int = 1):
    """amax of every linear input (incl. per-layer expert inputs)."""
    from transformers.models.hy_v3 import modeling_hy_v3 as m

    amax: Dict[str, float] = {}
    hooks = []

    def lin_hook(name):
        def fn(mod, args):
            v = float(args[0].detach().abs().amax())
            amax[name] = max(amax.get(name, 0.0), v)

        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_pre_hook(lin_hook(name)))
        if isinstance(mod, m.HYV3Experts):

            def ex_hook(mod, args, name=name):
                x = args[0].detach().float()
                gu = mod.gate_up_proj.float()
                dn = mod.down_proj.float()
                for e in range(gu.shape[0]):
                    g, u = (x @ gu[e].t()).chunk(2, dim=-1)
                    h = torch.nn.functional.silu(g) * u
                    for proj, v in (
                        ("gate_proj", x),
                        ("up_proj", x),
                        ("down_proj", h),
                    ):
                        k = f"{name}.{e}.{proj}"
                        amax[k] = max(amax.get(k, 0.0), float(v.abs().amax()))
                del dn

            hooks.append(mod.register_forward_pre_hook(ex_hook))
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, config.vocab_size, (1, n_tokens), generator=g)
    with torch.no_grad():
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    return amax


def _ckpt_name(name: str) -> str:
    name = name.replace(".mlp.gate.weight", ".mlp.router.gate.weight")
    name = name.replace(".mlp.e_score_correction_bias", ".mlp.expert_bias")
    return name.replace(".mlp.shared_experts.", ".mlp.shared_mlp.")


def _byte_tokenizer(out: Path, config):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    specials = [
        "<pad>",
        "<s>",
        "</s>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
    ]
    vocab = {t: i for i, t in enumerate(specials)}
    for ch in sorted(alphabet):
        vocab[ch] = len(vocab)
    assert len(vocab) <= config.vocab_size, len(vocab)
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(specials)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(str(out))


def make_tiny_hy3(out, seed: int = 0, **config_over) -> Path:
    from transformers.models.hy_v3 import modeling_hy_v3 as m

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = tiny_config(**config_over)
    model = m.HYV3ForCausalLM(config).float().eval()
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, m.HYV3MoE):
                mod.e_score_correction_bias.normal_(0, 0.05)
            if isinstance(mod, m.HYV3RMSNorm):
                mod.weight.uniform_(0.8, 1.2)
    amax = _calibrate(model, config)
    L = config.num_hidden_layers
    main: Dict[str, torch.Tensor] = {}
    last: Dict[str, torch.Tensor] = {}

    def put_linear(name: str, w: torch.Tensor, amax_key: str):
        q, s = _fp8(w.float())
        main[name] = q
        main[name[: -len(".weight")] + ".weight_scale"] = s
        a = max(amax.get(amax_key, 1.0), 1e-4) * 1.25 / FP8_MAX
        last[name[: -len(".weight")] + ".input_scale"] = torch.tensor(
            [a], dtype=torch.float32
        )

    I = config.moe_intermediate_size
    for name, t in model.state_dict().items():
        if name.endswith("inv_freq"):
            continue
        if ".mlp.experts.gate_up_proj" in name:
            stem = name[: -len(".gate_up_proj")]
            for e in range(t.shape[0]):
                for proj, w in (("gate_proj", t[e, :I]), ("up_proj", t[e, I:])):
                    put_linear(
                        f"{stem}.{e}.{proj}.weight", w, f"{stem}.{e}.{proj}"
                    )
            continue
        if ".mlp.experts.down_proj" in name:
            stem = name[: -len(".down_proj")]
            for e in range(t.shape[0]):
                put_linear(
                    f"{stem}.{e}.down_proj.weight",
                    t[e],
                    f"{stem}.{e}.down_proj",
                )
            continue
        ck = _ckpt_name(name)
        is_linear = (
            t.dim() == 2
            and name.endswith(".weight")
            and "embed_tokens" not in name
            and not name.startswith("lm_head")
            and ".mlp.gate.weight" not in name
        )
        if is_linear:
            put_linear(ck, t, name[: -len(".weight")])
        elif name.endswith("e_score_correction_bias"):
            main[ck] = t.float().clone()
        else:
            main[ck] = t.to(torch.bfloat16).clone()
    for layer in range(L):
        for kv in ("k_scale", "v_scale"):
            last[f"model.layers.{layer}.self_attn.{kv}"] = torch.tensor(
                1.0, dtype=torch.float32
            )
    # MTP layer (index num_hidden_layers): a copy of the last layer.
    mtp = {}
    for k, v in list(main.items()) + list(last.items()):
        pre = f"model.layers.{L - 1}."
        if k.startswith(pre):
            mtp[f"model.layers.{L}." + k[len(pre) :]] = v.clone()
    mtp[f"model.layers.{L}.enorm.weight"] = torch.ones(config.hidden_size)
    last.update(mtp)

    shards = {
        "model-00001-of-00002.safetensors": main,
        "model-00002-of-00002.safetensors": last,
    }
    weight_map = {}
    for fname, tensors in shards.items():
        save_file(
            {k: v.contiguous() for k, v in tensors.items()}, str(out / fname)
        )
        weight_map.update({k: fname for k in tensors})
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1)
    )
    cfg = config.to_dict()
    cfg["architectures"] = ["HYV3ForCausalLM"]
    cfg["dtype"] = "bfloat16"
    cfg["quantization_config"] = {
        "activation_scheme": "static",
        "ignored_layers": ["lm_head", "model.embed_tokens"],
        "quant_method": "fp8",
        "kv_cache_scheme": "static",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2], "bos_token_id": 1})
    )
    _byte_tokenizer(out, config)
    return out


def reference_model(model_dir, dtype=torch.float32):
    """transformers ``HYV3ForCausalLM`` with the checkpoint's dequantized
    weights (experts stacked back into ``gate_up_proj``/``down_proj``)."""
    from transformers import AutoConfig
    from transformers.models.hy_v3 import modeling_hy_v3 as m

    from .checkpoint import Hy3Checkpoint, dequant_per_tensor

    ck = Hy3Checkpoint(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    model = m.HYV3ForCausalLM(config).to(dtype).eval()
    sd = {}
    for name, t in ck.iter_resident(dtype):
        sd[name] = t
    for layer in ck.moe_layers:
        gu, dn = [], []
        for e in range(ck.num_experts):
            w = {}
            for proj in ("gate_proj", "up_proj", "down_proj"):
                q = ck.get(ck.expert_key(layer, e, proj, "weight"))
                s = ck.get(ck.expert_key(layer, e, proj, "weight_scale"))
                w[proj] = dequant_per_tensor(q, s, dtype)
            gu.append(torch.cat([w["gate_proj"], w["up_proj"]], 0))
            dn.append(w["down_proj"])
        p = f"model.layers.{layer}.mlp.experts."
        sd[p + "gate_up_proj"] = torch.stack(gu)
        sd[p + "down_proj"] = torch.stack(dn)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("inv_freq")]
    if missing or unexpected:
        raise ValueError(f"missing {missing[:5]} unexpected {unexpected[:5]}")
    for mod in model.modules():
        if isinstance(mod, m.HYV3MoE):
            mod.e_score_correction_bias.data = (
                mod.e_score_correction_bias.data.float()
            )
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--experts", type=int, default=8)
    a = p.parse_args()
    path = make_tiny_hy3(
        a.out, seed=a.seed, num_hidden_layers=a.layers, num_experts=a.experts
    )
    print(path)


if __name__ == "__main__":
    main()
