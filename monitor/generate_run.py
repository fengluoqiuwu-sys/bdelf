#!/usr/bin/env python3
"""本机一次生成 + 廉价指标（由 monitor API 子进程调用；结束即释放 GPU）。

stdout 为 NDJSON：status / sample（每条生成完立刻）/ eval（全部生成完后统一）/ done / error。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_out(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fail(out: Path, message: str, extra: dict[str, Any] | None = None) -> int:
    payload = {"ok": False, "error": message}
    if extra:
        payload.update(extra)
    _write_out(out, payload)
    print(f"[generate-run] {message}", file=sys.stderr, flush=True)
    return 1


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


def _mean(vals: list[float]) -> float:
    finite = [v for v in vals if isinstance(v, (int, float)) and v == v]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _cpu_metrics(token_rows: list[list[int]], texts: list[str]) -> dict[str, Any]:
    from eval.gen_ppl import unigram_entropy
    from eval.nonword import score_nonwords
    from eval.repetition import score_repetition
    import numpy as np

    per: list[dict[str, Any]] = []
    uniq_sum = 0.0
    nonempty = 0
    for ids, text in zip(token_rows, texts):
        arr = np.asarray(ids, dtype=np.int64)
        uniq = int(np.unique(arr).size) if arr.size else 0
        n = int(arr.size)
        uniq_sum += float(uniq)
        is_ne = bool(isinstance(text, str) and text.strip())
        nonempty += int(is_ne)
        per.append({
            "n_tokens": n,
            "n_chars": len(text or ""),
            "uniq": uniq,
            "dist1": (float(uniq) / float(n)) if n else float("nan"),
            "src_entropy": unigram_entropy(arr) if n else float("nan"),
            "nonempty": is_ne,
        })
    reps, accepts, rep_sum = score_repetition(texts)
    try:
        _, _, nw_sum = score_nonwords(texts)
    except Exception:
        nw_sum = {}
    n = max(len(token_rows), 1)
    for i, row in enumerate(per):
        row["seq_rep_4"] = reps[i] if i < len(reps) else float("nan")
        row["accept_human"] = bool(accepts[i]) if i < len(accepts) else False
    return {
        "gen_uniq_mean": uniq_sum / float(n),
        "nonempty_frac": float(nonempty) / float(n),
        "mean_n_tokens": _mean([float(r["n_tokens"]) for r in per]),
        "mean_src_entropy": _mean([float(r["src_entropy"]) for r in per]),
        "mean_dist1": _mean([float(r["dist1"]) for r in per]),
        "median_rep": rep_sum.get("median_rep"),
        "accept_at_human": rep_sum.get("accept_at_human"),
        "nonword_word_pct": nw_sum.get("nonword_word_pct"),
        "per_sample": per,
    }


def _model_ppl(model, tokens, device, dtype) -> float:
    import torch
    from train.eval import _eval_loss_branch, forward_loss
    from train.metrics import loss_to_ppl

    branch = _eval_loss_branch(model)
    use_amp = device.type == "cuda"
    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=dtype, enabled=use_amp):
            loss = forward_loss(model, tokens, branch=branch)
    if loss is None:
        return float("nan")
    val = float(loss.item())
    if not math.isfinite(val):
        return float("nan")
    return float(loss_to_ppl(val))


def _score_gpt2(texts: list[str], device, amp_dtype) -> dict[str, Any]:
    from models import get_hf_model, is_hf_model_cached
    from eval.gen_ppl import score_texts

    repo = "gpt2-large"
    if not is_hf_model_cached(repo):
        return {
            "skipped": True,
            "reason": "未缓存 gpt2-large，跳过 gen-ppl / entropy",
            "gen_ppl": float("nan"),
            "entropy": float("nan"),
            "per_gen_ppl": [],
            "per_entropy": [],
        }
    gpt2 = get_hf_model(repo, torch_dtype=amp_dtype, device=str(device))
    gpt2.eval()
    for p in gpt2.parameters():
        p.requires_grad_(False)
    try:
        per_ppl, per_ent, corpus = score_texts(
            texts, gpt2_model=gpt2, max_length=1024, device=device, amp_dtype=amp_dtype,
        )
        return {
            "skipped": False,
            "reason": "",
            "gen_ppl": float(corpus),
            "entropy": _mean([float(x) for x in per_ent]),
            "per_gen_ppl": [float(x) for x in per_ppl],
            "per_entropy": [float(x) for x in per_ent],
        }
    finally:
        del gpt2
        _release_cuda()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    mod = getattr(type(obj), "__module__", "") or ""
    if mod.startswith("numpy") and hasattr(obj, "item"):
        try:
            return _jsonable(obj.item())
        except Exception:
            return None
    return obj


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(_jsonable(event), ensure_ascii=False, allow_nan=False) + "\n")
    sys.stdout.flush()


def run_spec(spec: dict[str, Any]) -> dict[str, Any]:
    import hf_config  # noqa: F401
    import torch
    from tokenizer import get_tokenizer

    from generate import (
        encode_prefix_tokens,
        generate_tokens,
        load_model_from_checkpoint,
        model_supports_prefix,
        resolve_dtype,
    )
    from monitor.generate_meta import resolve_ckpt_file, run_model_name
    from monitor.runs import resolve_run_dir
    from train.generate_config import get_generate

    run = str(spec.get("run") or "")
    ckpt_id = str(spec.get("checkpoint") or "latest")
    profile = str(spec.get("profile") or "generate")
    num_tokens = int(spec.get("num_tokens") or 1024)
    num_samples = max(1, min(16, int(spec.get("num_samples") or 1)))
    seed0 = int(spec.get("seed") or 42)
    prompt = spec.get("prompt")
    sampling_over = spec.get("sampling") if isinstance(spec.get("sampling"), dict) else {}

    if num_tokens < 8 or num_tokens > 2048:
        raise ValueError("num_tokens 须在 8–2048")

    checkpoint_root = ROOT / "cache" / "checkpoints"
    run_dir = resolve_run_dir(checkpoint_root, run)
    if run_dir is None:
        raise FileNotFoundError(f"run 不存在: {run}")
    ckpt_path = resolve_ckpt_file(run_dir, ckpt_id)
    if ckpt_path is None:
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_id}")

    if not torch.cuda.is_available():
        raise RuntimeError("本机没有可用 CUDA")
    device = torch.device("cuda")

    _emit({"type": "status", "message": "加载模型…"})
    model, model_meta, step, train_cfg = load_model_from_checkpoint(ckpt_path, device)
    dtype = resolve_dtype(device, train_cfg)
    tokenizer_name = (model_meta.get("config") or {}).get("tokenizer")
    if not tokenizer_name:
        raise ValueError("模型配置缺少 tokenizer")
    tokenizer = get_tokenizer(tokenizer_name)

    if getattr(model, "ace_attachable", False):
        from models.lm.elf.ace import attach_ace_identity, model_hash_from_checkpoint

        ace_hash = model_hash_from_checkpoint(ckpt_path)
        if ace_hash:
            attach_ace_identity(
                model, model_hash=ace_hash, step=step, tokenizer=tokenizer_name,
            )

    prefix_tokens = None
    prefix_len = 0
    prompt_text = prompt if isinstance(prompt, str) and prompt.strip() else None
    if prompt_text is not None:
        if not model_supports_prefix(model):
            raise ValueError(f"{model_meta.get('name')} 为无条件生成，不支持前缀续写")
        prefix_tokens = encode_prefix_tokens(
            prompt_text,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            num_samples=1,
            device=device,
        )
        prefix_len = int(prefix_tokens.size(1))
        if prefix_len >= num_tokens:
            raise ValueError(f"前缀 {prefix_len} token，须短于 num_tokens={num_tokens}")

    model_name = str(model_meta.get("name") or run_model_name(run_dir))
    gen_cfg = get_generate(model_name, profile)
    sampling_cfg = gen_cfg.to_sampling_cfg()
    for k, v in sampling_over.items():
        if k in ("ace_direction", "ace_step_lo", "ace_step_hi"):
            continue
        if v == "" or v is None:
            sampling_cfg.pop(k, None)
        else:
            sampling_cfg[k] = v
    sampling_cfg.pop("ace_direction", None)
    sampling_cfg.pop("ace_step_lo", None)
    sampling_cfg.pop("ace_step_hi", None)

    token_rows: list[list[int]] = []
    cpu_tensors: list[Any] = []
    texts: list[str] = []
    completions: list[str] = []
    nfes: list[int] = []
    seeds: list[int] = []

    for i in range(num_samples):
        seed_i = seed0 + i
        _emit({"type": "status", "message": f"生成 {i + 1}/{num_samples}（seed {seed_i}）…"})
        tokens, nfe = generate_tokens(
            model,
            num_tokens=num_tokens,
            num_samples=1,
            seed=seed_i,
            device=device,
            dtype=dtype,
            sampling_cfg=sampling_cfg,
            prefix_tokens=prefix_tokens,
        )
        row = tokens[0].detach().cpu()
        ids = row.tolist()
        full = tokenizer.decode(ids, skip_special_tokens=False)
        if prefix_len > 0:
            completion = tokenizer.decode(ids[prefix_len:], skip_special_tokens=False)
        else:
            completion = full
        cpu_tensors.append(row)
        token_rows.append(ids)
        texts.append(full)
        completions.append(completion)
        nfes.append(int(nfe))
        seeds.append(seed_i)
        _emit({
            "type": "sample",
            "index": i,
            "n": num_samples,
            "seed": seed_i,
            "nfe": int(nfe),
            "text": full,
            "completion": completion,
        })
        del tokens

    _emit({"type": "status", "message": "生成完毕，开始评测…"})
    model_ppls: list[float] = []
    for row in cpu_tensors:
        try:
            batch = row.unsqueeze(0).to(device)
            model_ppls.append(_model_ppl(model, batch, device, dtype))
            del batch
        except Exception as exc:
            print(f"[generate-run] model ppl skipped: {exc}", file=sys.stderr, flush=True)
            model_ppls.append(float("nan"))

    del prefix_tokens, model, cpu_tensors
    _release_cuda()

    cpu = _cpu_metrics(token_rows, texts)
    gpt = _score_gpt2(texts, device, dtype)
    _release_cuda()

    per = cpu.pop("per_sample")
    pps = gpt.get("per_gen_ppl") or []
    ents = gpt.get("per_entropy") or []
    for i, row in enumerate(per):
        row["text"] = texts[i]
        row["completion"] = completions[i]
        row["ppl"] = model_ppls[i] if i < len(model_ppls) else float("nan")
        row["gen_ppl"] = pps[i] if i < len(pps) else float("nan")
        row["entropy"] = ents[i] if i < len(ents) else float("nan")
        _emit({
            "type": "eval",
            "index": i,
            "seed": seeds[i],
            "metrics": {
                "ppl": row["ppl"],
                "gen_ppl": row["gen_ppl"],
                "entropy": row["entropy"],
                "gen_uniq": row.get("uniq"),
                "src_entropy": row.get("src_entropy"),
                "dist1": row.get("dist1"),
                "n_tokens": row.get("n_tokens"),
                "nonempty": row.get("nonempty"),
                "seq_rep_4": row.get("seq_rep_4"),
                "accept_human": row.get("accept_human"),
            },
        })

    payload = {
        "ok": True,
        "run": run,
        "checkpoint": ckpt_path.name,
        "step": step,
        "model": model_name,
        "nfe": int(sum(nfes)),
        "prefix_len": prefix_len,
        "num_tokens": num_tokens,
        "num_samples": num_samples,
        "seed": seed0,
        "seed_next": seed0 + num_samples,
        "profile": profile,
        "metrics": {
            "ppl": _mean(model_ppls),
            "gen_ppl": gpt.get("gen_ppl"),
            "entropy": gpt.get("entropy"),
            "gen_uniq_mean": cpu.get("gen_uniq_mean"),
            "nonempty_frac": cpu.get("nonempty_frac"),
            "mean_src_entropy": cpu.get("mean_src_entropy"),
            "mean_dist1": cpu.get("mean_dist1"),
            "median_rep": cpu.get("median_rep"),
            "accept_at_human": cpu.get("accept_at_human"),
            "nonword_word_pct": cpu.get("nonword_word_pct"),
            "gpt2_skipped": bool(gpt.get("skipped")),
            "gpt2_reason": gpt.get("reason") or "",
        },
        "samples": per,
    }
    _emit({"type": "done", "seed_next": seed0 + num_samples, "metrics": payload["metrics"]})
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="monitor 本机生成作业")
    parser.add_argument("--spec", required=True, help="输入 JSON 路径")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    args = parser.parse_args(argv)
    out = Path(args.out)
    spec_path = Path(args.spec)
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _emit({"type": "error", "error": f"无法读 spec: {exc}"})
        return _fail(out, f"无法读 spec: {exc}")
    if not isinstance(spec, dict):
        _emit({"type": "error", "error": "spec 必须是对象"})
        return _fail(out, "spec 必须是对象")
    try:
        payload = run_spec(spec)
    except Exception as exc:
        _release_cuda()
        _emit({"type": "error", "error": str(exc)})
        return _fail(out, str(exc))
    _write_out(out, _jsonable(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
