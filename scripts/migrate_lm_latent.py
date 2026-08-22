#!/usr/bin/env python3
"""One-shot lm/latent layout migration (run from repo root)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LM_FAMILIES = (
    "ar",
    "ar1_5",
    "ar2",
    "bd3lm",
    "bdelf",
    "cola",
    "denoiser_chart",
    "elf",
    "jac_ellipsoid",
    "late_ce",
    "lexce",
    "loopsc",
    "odar",
    "posbeta",
    "residw",
    "trace",
)
LATENT_FAMILIES = ("cola_vae",)
SHARED_MODEL_DIRS = ("elf_core",)

# Longest-first replacement for import rewriting
FAMILY_IMPORT_RE = re.compile(
    r"\b(?:from|import)\s+models\.(?:"
    + "|".join(re.escape(f) for f in sorted(LM_FAMILIES + LATENT_FAMILIES, key=len, reverse=True))
    + r")(?:\.|\b)"
)


def _kind_of(family: str) -> str:
    if family in LATENT_FAMILIES:
        return "latent"
    if family in LM_FAMILIES:
        return "lm"
    raise KeyError(family)


def move_models() -> None:
    models = REPO / "models"
    (models / "lm").mkdir(exist_ok=True)
    (models / "latent").mkdir(exist_ok=True)
    for fam in LM_FAMILIES:
        src = models / fam
        dst = models / "lm" / fam
        if src.is_dir() and not dst.exists():
            shutil.move(str(src), str(dst))
    for fam in LATENT_FAMILIES:
        src = models / fam
        dst = models / "latent" / fam
        if src.is_dir() and not dst.exists():
            shutil.move(str(src), str(dst))
    for shared in SHARED_MODEL_DIRS:
        src = models / shared
        dst = models / "lm" / shared
        if src.is_dir() and not dst.exists():
            shutil.move(str(src), str(dst))


def move_configs() -> None:
    for base in ("config/models", "config/train/model", "config/generate"):
        root = REPO / base
        (root / "lm").mkdir(exist_ok=True)
        (root / "latent").mkdir(exist_ok=True)
        for fam in LM_FAMILIES:
            src = root / fam
            dst = root / "lm" / fam
            if src.is_dir() and not dst.exists():
                shutil.move(str(src), str(dst))
        for fam in LATENT_FAMILIES:
            src = root / fam
            dst = root / "latent" / fam
            if src.is_dir() and not dst.exists():
                shutil.move(str(src), str(dst))
        # prototype stays at root for train/model
        if base == "config/train/model":
            pass


def rewrite_import_line(line: str) -> str:
    def repl(m: re.Match[str]) -> str:
        prefix = m.group(0)
        # extract family name after models.
        rest = line[m.end() :]
        fam_match = re.match(
            r"(" + "|".join(re.escape(f) for f in LM_FAMILIES + LATENT_FAMILIES) + r")",
            rest,
        )
        if not fam_match:
            return prefix
        fam = fam_match.group(1)
        kind = _kind_of(fam)
        return prefix.replace(f"models.{fam}", f"models.{kind}.{fam}", 1)

    if "models.lm.elf_core" in line:
        line = line.replace("models.lm.elf_core", "models.lm.elf_core")
    for fam in sorted(LM_FAMILIES + LATENT_FAMILIES, key=len, reverse=True):
        kind = _kind_of(fam)
        line = re.sub(
            rf"\bmodels\.{re.escape(fam)}\b",
            f"models.{kind}.{fam}",
            line,
        )
    return line


def rewrite_python_files() -> None:
    for path in REPO.rglob("*.py"):
        if ".venv" in path.parts or "cache" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        new_lines = [rewrite_import_line(ln) for ln in text.splitlines(keepends=True)]
        new_text = "".join(new_lines)
        # fix double kind if any
        new_text = new_text.replace("models.lm.", "models.lm.")
        new_text = new_text.replace("models.latent.", "models.latent.")
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")


def add_kind_constants() -> None:
    for fam in LM_FAMILIES:
        init = REPO / "models" / "lm" / fam / "__init__.py"
        if not init.is_file():
            continue
        text = init.read_text(encoding="utf-8")
        if "KIND" in text:
            continue
        init.write_text('KIND = "lm"\n\n' + text, encoding="utf-8")
    for fam in LATENT_FAMILIES:
        init = REPO / "models" / "latent" / fam / "__init__.py"
        if not init.is_file():
            continue
        text = init.read_text(encoding="utf-8")
        if "KIND" in text:
            continue
        init.write_text('KIND = "latent"\n\n' + text, encoding="utf-8")


def main() -> None:
    move_models()
    move_configs()
    rewrite_python_files()
    add_kind_constants()
    print("migration: models + configs moved, imports rewritten")


if __name__ == "__main__":
    main()
