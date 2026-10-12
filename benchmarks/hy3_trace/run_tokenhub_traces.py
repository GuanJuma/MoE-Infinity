# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Run tokenhub_bench datasets through Hy3 on MoE-Infinity, save traces.

Reads the same jsonl rows tokenhub_bench sends (one OpenAI ``messages``
list per line), applies the chat template with tokenhub's thinking-off
setting for Hy3 (``chat_template_kwargs.reasoning_effort=no_think``) and
generates in-process, one request at a time.  Per request it writes a
routing trace (``traces/<dataset>/<row>.pt``, see
``moe_infinity/hy3/routing_trace.py``) and a line in ``requests.jsonl``.

    python benchmarks/hy3_trace/run_tokenhub_traces.py \
        --model-dir /model --offload-dir /scratch/$ME/hy3_store \
        --tokenhub-dir /scratch/$ME/tokenhub_bench \
        --datasets cache_1k_hot cache_1k_cold cache_1k_mix50 \
        --limit 20 --max-new-tokens 32 --out /scratch/$ME/hy3_traces/run1

Dataset names: ``cache_<1k|4k|8k|16k>_<hot|cold|mix50>``
(``datasets/cache_variants``), ``perf_zh_<1k|9k|...>`` (``datasets/perf_zh``,
not shipped in the zip) or any ``.jsonl`` path.  ``--count-only`` only
tokenizes (no GPU) to size a run.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from moe_infinity.hy3.cli import add_engine_args, config_from_args  # noqa: E402


def resolve_dataset(name: str, tokenhub_dir: Path) -> Path:
    p = Path(name)
    if p.suffix == ".jsonl" and p.exists():
        return p
    if name.startswith("cache_"):
        p = tokenhub_dir / "datasets" / "cache_variants" / f"{name}.jsonl"
    elif name.startswith("perf_zh_"):
        p = tokenhub_dir / "datasets" / "perf_zh" / f"{name}.jsonl"
    else:
        raise SystemExit(f"unknown dataset {name!r}")
    if not p.exists():
        raise SystemExit(f"dataset file not found: {p}")
    return p


def load_rows(path: Path, offset: int, limit: int):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj if isinstance(obj, list) else obj.get("messages")
            if not isinstance(msgs, list):
                continue
            rows.append((i, msgs))
    rows = rows[offset:]
    return rows[:limit] if limit > 0 else rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_engine_args(p)
    p.add_argument("--tokenhub-dir", required=True)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["cache_1k_hot", "cache_1k_cold", "cache_1k_mix50"],
    )
    p.add_argument(
        "--limit", type=int, default=20, help="rows per dataset (0 = all)"
    )
    p.add_argument("--offset", type=int, default=0)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="32 = tokenhub cache_hit; 4096 = its OTPS runs",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument(
        "--chat-template-kwargs", default='{"reasoning_effort": "no_think"}'
    )
    p.add_argument("--out", required=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--count-only", action="store_true")
    a = p.parse_args(argv)

    tokenhub = Path(a.tokenhub_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else {}
    plan = [(d, resolve_dataset(d, tokenhub)) for d in a.datasets]

    if a.count_only:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.model_dir)
        for name, path in plan:
            rows = load_rows(path, a.offset, a.limit)
            lens = []
            for _, msgs in rows:
                text = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False, **kw
                )
                lens.append(
                    len(tok(text, add_special_tokens=False)["input_ids"])
                )
            if lens:
                print(
                    f"{name}: {len(lens)} rows, prompt tokens "
                    f"min {min(lens)} mean {sum(lens) / len(lens):.0f} "
                    f"max {max(lens)}"
                )
        return

    from moe_infinity.hy3.engine import Hy3Engine
    from moe_infinity.hy3.experts import exit_process
    from moe_infinity.hy3.routing_trace import save_trace

    eng = Hy3Engine(config_from_args(a))
    mem = eng.memory_report()
    print(json.dumps(mem, indent=1), flush=True)
    import torch
    import transformers

    (out / "run_meta.json").write_text(
        json.dumps(
            {
                "engine_config": vars(config_from_args(a)),
                "memory": mem,
                "checkpoint": eng.info,
                "datasets": {n: str(pth) for n, pth in plan},
                "limit": a.limit,
                "offset": a.offset,
                "max_new_tokens": a.max_new_tokens,
                "chat_template_kwargs": kw,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "host": platform.node(),
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=1,
            default=str,
        )
    )

    log = open(out / "requests.jsonl", "a")
    t_all = time.perf_counter()
    done = 0
    try:
        for name, path in plan:
            rows = load_rows(path, a.offset, a.limit)
            print(f"[run] {name}: {len(rows)} rows from {path}", flush=True)
            for row, msgs in rows:
                tpath = out / "traces" / name / f"{row:05d}.pt"
                if tpath.exists() and not a.no_resume:
                    continue
                ids = eng.encode_chat(msgs, kw)
                if len(ids) + a.max_new_tokens > eng.cfg.max_context:
                    print(
                        f"[run] skip {name}#{row}: {len(ids)} tokens > "
                        f"max_context",
                        flush=True,
                    )
                    continue
                rid = f"{name}_{row:05d}"
                r = eng.generate(
                    ids,
                    max_new_tokens=a.max_new_tokens,
                    temperature=a.temperature,
                    ignore_eos=a.ignore_eos,
                    rid=rid,
                    meta={
                        "dataset": name,
                        "row": row,
                        "dataset_file": str(path),
                    },
                )
                save_trace(r["trace"], tpath)
                text = eng.tokenizer.decode(
                    r["output_ids"], skip_special_tokens=True
                )
                st = r["stats"]
                log.write(
                    json.dumps(
                        {
                            "dataset": name,
                            "row": row,
                            "rid": rid,
                            **st,
                            "output": text[:200],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log.flush()
                done += 1
                print(
                    f"[run] {rid}: prompt {st['prompt_tokens']} "
                    f"out {st['completion_tokens']} ttft {st['ttft_s']:.2f}s "
                    f"decode {st['decode_tok_per_s'] or 0:.2f} tok/s "
                    f"hit p/d {st['prefill_hit_rate'] or 0:.2f}/"
                    f"{st['decode_hit_rate'] or 0:.2f} "
                    f"h2d {st['h2d_bytes'] / 2**30:.1f} GiB",
                    flush=True,
                )
    finally:
        log.close()
        print(
            f"[run] {done} requests in {time.perf_counter() - t_all:.0f}s; "
            f"traces in {out / 'traces'}",
            flush=True,
        )
    exit_process(0)


if __name__ == "__main__":
    main()
