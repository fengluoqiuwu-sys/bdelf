"""``python -m eval``：TriFluency CLI（score / naive / smoke）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _ensure_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        import os

        os.chdir(root)
    root_s = str(root)
    if sys.path[:1] != [root_s]:
        if root_s in sys.path:
            sys.path.remove(root_s)
        sys.path.insert(0, root_s)
    return root


def _device(s: str | None) -> torch.device:
    if s:
        return torch.device(s)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cmd_score(args: argparse.Namespace) -> int:
    from eval.io import read_annotated
    from eval.protocol import run_trifluency

    doc = read_annotated(args.input)
    if not doc.texts:
        print(f"[eval] no texts in {args.input}", file=sys.stderr)
        return 1
    run_trifluency(
        doc.texts,
        out_dir=args.out_dir,
        device=_device(args.device),
        max_length=args.max_length,
        gen_eval_dtype=args.gen_eval_dtype,
        skip_cola=args.skip_cola,
        skip_gen_ppl=args.skip_gen_ppl,
        n_min_accept=args.n_min_accept,
    )
    return 0


def cmd_naive(args: argparse.Namespace) -> int:
    from eval.io import write_texts
    from eval.naive import make_naive
    from eval.protocol import run_trifluency

    texts = make_naive(args.kind, args.n, num_tokens=args.num_tokens)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.txt"
    write_texts(
        samples_path,
        texts,
        meta={"kind": args.kind, "n": str(args.n), "num_tokens": str(args.num_tokens)},
    )
    run_trifluency(
        texts,
        out_dir=out_dir,
        device=_device(args.device),
        max_length=args.num_tokens,
        gen_eval_dtype=args.gen_eval_dtype,
        skip_cola=args.skip_cola,
        skip_gen_ppl=args.skip_gen_ppl,
        n_min_accept=args.n_min_accept,
    )
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    root = _ensure_root()
    default_in = (
        root / "temp/auto-research/elf-cfg/samples_32x1024_annotated.txt"
    )
    args.input = args.input or default_in
    args.out_dir = args.out_dir or (
        root / "temp/idea/trifluency/baselines/elf-cfg-32"
    )
    return cmd_score(args)


def main(argv: list[str] | None = None) -> int:
    _ensure_root()
    import hf_config  # noqa: F401

    parser = argparse.ArgumentParser(
        prog="python -m eval",
        description="TriFluency offline eval (A/B/C′ + Gen.PPL/entropy)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", default=None)
        p.add_argument("--gen-eval-dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
        p.add_argument("--skip-cola", action="store_true")
        p.add_argument("--skip-gen-ppl", action="store_true")

    p_score = sub.add_parser("score", help="Score annotated or plain texts")
    p_score.add_argument("--input", type=Path, required=True)
    p_score.add_argument("--out-dir", type=Path, required=True)
    p_score.add_argument("--max-length", type=int, default=1024)
    p_score.add_argument("--n-min-accept", type=int, default=50)
    add_common(p_score)
    p_score.set_defaults(func=cmd_score)

    p_naive = sub.add_parser("naive", help="Periodic / Phrase-bank negative controls")
    p_naive.add_argument("--kind", required=True, choices=("periodic", "phrase_bank"))
    p_naive.add_argument("--n", type=int, default=128)
    p_naive.add_argument("--num-tokens", type=int, default=1024)
    p_naive.add_argument("--out-dir", type=Path, required=True)
    p_naive.add_argument("--n-min-accept", type=int, default=50)
    add_common(p_naive)
    p_naive.set_defaults(func=cmd_naive)

    p_smoke = sub.add_parser("smoke", help="Score elf-cfg 32-sample annotated file")
    p_smoke.add_argument("--input", type=Path, default=None)
    p_smoke.add_argument("--out-dir", type=Path, default=None)
    p_smoke.add_argument("--max-length", type=int, default=1024)
    # 32 条冒烟：N_min=50 会使 clean_ppl 恒 invalid；默认放宽到 5
    p_smoke.add_argument("--n-min-accept", type=int, default=5)
    add_common(p_smoke)
    p_smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
