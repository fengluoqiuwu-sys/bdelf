#!/usr/bin/env python3
"""Mask schematics: one claim per figure. Labels in English; keep text off neighbors."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent

C_SEE = "#1f4e79"
C_HIDE = "#f2f3f4"
C_SPLIT = "#1c2833"
C_NONE = "#d5d8dc"
C_MSE = "#5dade2"
C_CE = "#e67e22"
C_KNOWN = "#a9dfbf"
C_TOK = "#d6eaf8"
C_MARK = "#f5b7b1"
DARK = {"#1c2833", "#7d6608", "#1f4e79"}


def setup_font() -> mpl.font_manager.FontProperties:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 220,
            "figure.facecolor": "white",
        }
    )
    return mpl.font_manager.FontProperties(family="DejaVu Sans")


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.png", bbox_inches="tight", facecolor="white", pad_inches=0.12)
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)


def as_rgb(mask: np.ndarray, on: str, off: str = C_HIDE) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 1:
        m = m[np.newaxis, :]
    out = np.empty(m.shape + (3,), dtype=np.float64)
    out[m] = mcolors.to_rgb(on)
    out[~m] = mcolors.to_rgb(off)
    return out


def cell_grid(
    ax: plt.Axes,
    rgb: np.ndarray,
    *,
    splits: list[int],
    units: list[int] | None = None,
) -> None:
    n_row, n_col = rgb.shape[:2]
    ax.imshow(rgb, origin="upper", interpolation="nearest", aspect="equal")
    ax.set_xticks(np.arange(-0.5, n_col, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_row, 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False, length=0)
    ax.grid(which="minor", color="white", linewidth=0.6, zorder=3)
    split_set = set(splits)
    for u in units or []:
        if u <= 0 or u in split_set:
            continue
        ax.axvline(u - 0.5, color="#7f8c8d", lw=0.9, zorder=4)
        ax.axhline(u - 0.5, color="#7f8c8d", lw=0.9, zorder=4)
    for s in splits:
        ax.axvline(s - 0.5, color=C_SPLIT, lw=1.6, zorder=5)
        ax.axhline(s - 0.5, color=C_SPLIT, lw=1.6, zorder=5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)


def group_ticks(ax: plt.Axes, groups: list[tuple[str, int, int]], font, *, xy: str) -> None:
    centers = [(a + b - 1) / 2 for _, a, b in groups]
    names = [g[0] for g in groups]
    if xy == "x":
        ax.set_xticks(centers)
        ax.set_xticklabels(names, fontproperties=font, fontsize=8)
    else:
        ax.set_yticks(centers)
        ax.set_yticklabels(names, fontproperties=font, fontsize=8)


def vis_unit_causal(n: int, unit: int) -> np.ndarray:
    u = np.arange(n) // int(unit)
    return u[None, :] <= u[:, None]


def pack_attn(n_left: int, n_right: int, left_unit: int, right_unit: int) -> np.ndarray:
    n = n_left + n_right
    vis = np.zeros((n, n), dtype=bool)
    vis[:n_left, :n_left] = vis_unit_causal(n_left, left_unit)
    vis[n_left:, n_left:] = vis_unit_causal(n_right, right_unit)
    vis[n_left:, :n_left] = True
    return vis


def add_legend(fig: plt.Figure, handles: list, *, ncol: int, fontsize: int = 8) -> None:
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        borderaxespad=0.2,
        handlelength=1.1,
        columnspacing=1.1,
        handletextpad=0.4,
    )


def _one_attn(
    font: mpl.font_manager.FontProperties,
    vis: np.ndarray,
    groups: list[tuple[str, int, int]],
    splits: list[int],
    stem: str,
    *,
    units: list[int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.7), layout="constrained")
    cell_grid(ax, as_rgb(vis, C_SEE), splits=splits, units=units)
    group_ticks(ax, groups, font, xy="x")
    group_ticks(ax, groups, font, xy="y")
    ax.set_xlabel("Key", fontproperties=font, fontsize=9, labelpad=3)
    ax.set_ylabel("Query", fontproperties=font, fontsize=9, labelpad=3)
    add_legend(
        fig,
        [
            Patch(facecolor=C_SEE, label="visible"),
            Patch(facecolor=C_HIDE, edgecolor="#b0b7bd", label="masked"),
        ],
        ncol=2,
    )
    save(fig, stem)


def plot_attn(font: mpl.font_manager.FontProperties) -> None:
    w = 4
    n_done, n_cur = 2 * w, w
    _one_attn(
        font,
        pack_attn(n_done, n_cur, left_unit=w, right_unit=w),
        [("block 0", 0, w), ("block 1", w, 2 * w), ("current", 2 * w, 3 * w)],
        [n_done],
        "belf_attn",
        units=[w, 2 * w],
    )
    s, win, n_left = 2, 8, 4
    _one_attn(
        font,
        pack_attn(n_left, win, left_unit=1, right_unit=s),
        [("prefix", 0, n_left), ("window", n_left, n_left + win)],
        [n_left],
        "relf_attn",
        units=list(range(s, n_left, s)) + list(range(n_left + s, n_left + win, s)),
    )


def _cell_fs(txt: str) -> float:
    plain = txt.replace("$", "").replace(r"\varepsilon", "e").replace("\\", "")
    if len(plain) >= 6:
        return 6.0
    if len(plain) >= 4:
        return 7.0
    return 8.0


def _fill_row(
    ax: plt.Axes,
    colors: list[str],
    texts: list[str],
    font,
    *,
    ytick: str,
    show_y: bool = True,
) -> None:
    n = len(colors)
    rgb = np.array([mcolors.to_rgb(c) for c in colors], dtype=np.float64)[np.newaxis, :, :]
    ax.imshow(rgb, origin="upper", interpolation="nearest", aspect="auto")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0.52, -0.52)
    for i, (c, txt) in enumerate(zip(colors, texts)):
        if not txt:
            continue
        ink = "white" if c.lower() in DARK else "#1c2833"
        ax.text(
            i,
            0,
            txt,
            ha="center",
            va="center",
            fontsize=_cell_fs(txt),
            color=ink,
            fontproperties=font,
            clip_on=True,
            zorder=6,
        )
        ax.axvline(i + 0.5, color="white", lw=1.0, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([0])
    if show_y:
        ax.set_yticklabels([ytick], fontproperties=font, fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#7f8c8d")


def plot_belf_pack(font: mpl.font_manager.FontProperties) -> None:
    n_pre, n_blk = 4, 4
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 1.85), layout="constrained")
    rows = [
        (axes[0], "flow hop", [C_NONE] * n_pre + [C_MSE] * n_blk, [""] * n_pre + ["MSE"] * n_blk),
        (axes[1], "decode hop", [C_NONE] * n_pre + [C_CE] * n_blk, [""] * n_pre + ["CE"] * n_blk),
    ]
    for ax, ytick, colors, texts in rows:
        _fill_row(ax, colors, texts, font, ytick=ytick)
        ax.axvline(n_pre - 0.5, color=C_SPLIT, lw=1.6, zorder=5)
    axes[1].set_xticks([(n_pre - 1) / 2, n_pre + (n_blk - 1) / 2])
    axes[1].set_xticklabels(["completed blocks", "current block"], fontproperties=font, fontsize=8)
    add_legend(
        fig,
        [
            Patch(facecolor=C_NONE, edgecolor="#b0b7bd", label="no loss"),
            Patch(facecolor=C_MSE, label="velocity MSE only"),
            Patch(facecolor=C_CE, label="CE only"),
        ],
        ncol=3,
    )
    save(fig, "belf_pack")


def plot_belf_clean(font: mpl.font_manager.FontProperties) -> None:
    n_pre, n_known, n_unk = 4, 2, 2
    n_left = n_pre
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 1.95), layout="constrained")
    mid = [C_NONE] * n_pre + [C_KNOWN] * n_known + [C_MSE] * n_unk
    last = [C_NONE] * n_pre + [C_KNOWN] * n_known + [C_CE] * n_unk
    mid_txt = [""] * n_pre + ["clean"] * n_known + ["MSE"] * n_unk
    last_txt = [""] * n_pre + ["clean"] * n_known + ["CE"] * n_unk
    rows = [
        (axes[0], "flow hop", mid, mid_txt),
        (axes[1], "decode hop", last, last_txt),
    ]
    for ax, ytick, colors, texts in rows:
        _fill_row(ax, colors, texts, font, ytick=ytick)
        ax.axvline(n_left - 0.5, color=C_SPLIT, lw=1.6, zorder=5)
        ax.axvline(n_left + n_known - 0.5, color="#7f8c8d", lw=1.1, zorder=4)
    axes[1].set_xticks(
        [
            (n_pre - 1) / 2,
            n_pre + (n_known - 1) / 2,
            n_pre + n_known + (n_unk - 1) / 2,
        ]
    )
    axes[1].set_xticklabels(
        ["completed", "known rem.", "unknown"],
        fontproperties=font,
        fontsize=8,
    )
    add_legend(
        fig,
        [
            Patch(facecolor=C_NONE, edgecolor="#b0b7bd", label="prefix, no loss"),
            Patch(facecolor=C_KNOWN, label="known, pinned clean"),
            Patch(facecolor=C_MSE, label="velocity MSE only"),
            Patch(facecolor=C_CE, label="CE only"),
        ],
        ncol=2,
    )
    save(fig, "belf_clean")


def _t_color(label: str) -> str:
    table = {
        "0": "#1c2833",
        r"$L_0$": "#7d6608",
        r"$L_1$": "#b7950b",
        r"$L_2$": "#f4d03f",
        "1-ε": "#f9e79f",
        "1": C_KNOWN,
        "t": "#b7950b",
        "": C_NONE,
    }
    return table[label]


def plot_relf_masks(font: mpl.font_manager.FontProperties) -> None:
    # Four examples of composing independent left/right clips; loss is per-cell on leftover F.
    branches = [
        (
            "No clip: full $F$",
            ["·"] * 8,
            ["1-ε", "1-ε", r"$L_2$", r"$L_2$", r"$L_1$", r"$L_1$", r"$L_0$", r"$L_0$"],
            ["CE", "CE", "MSE", "MSE", "MSE", "MSE", "MSE", "MSE"],
        ),
        (
            "Left clip: BOS at $k{=}4$",
            [""] * 4 + ["BOS"] + ["·"] * 3,
            [""] * 4 + [r"$L_1$", r"$L_1$", r"$L_0$", r"$L_0$"],
            [""] * 4 + ["MSE", "MSE", "MSE", "MSE"],
        ),
        (
            "Right clip: EOS at $k{=}3$",
            ["·"] * 3 + ["EOS"] + [""] * 4,
            ["1-ε", "1-ε", r"$L_2$", r"$L_2$"] + [""] * 4,
            ["CE", "CE", "MSE", "MSE"] + [""] * 4,
        ),
        (
            "Both clips: BOS $k{=}1$, EOS $k{=}5$",
            [""] + ["BOS"] + ["·"] * 3 + ["EOS"] + [""] * 2,
            [""] + ["1-ε"] + [r"$L_2$", r"$L_2$", r"$L_1$", r"$L_1$"] + [""] * 2,
            [""] + ["CE"] + ["MSE"] * 4 + [""] * 2,
        ),
    ]
    seq_color = {"": C_NONE, "BOS": C_MARK, "EOS": C_MARK, "·": C_TOK}
    loss_color = {"CE": C_CE, "MSE": C_MSE, "": C_NONE}
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(9.6, 3.15),
        layout="constrained",
        height_ratios=[0.72, 1.05, 0.72],
    )
    yticks = ["seq", "time", "loss"]
    for j, (title, seq, t_labs, loss) in enumerate(branches):
        ax_s, ax_t, ax_l = axes[0, j], axes[1, j], axes[2, j]
        show = j == 0
        _fill_row(ax_s, [seq_color[x] for x in seq], seq, font, ytick=yticks[0], show_y=show)
        _fill_row(ax_t, [_t_color(x) for x in t_labs], t_labs, font, ytick=yticks[1], show_y=show)
        _fill_row(ax_l, [loss_color[x] for x in loss], loss, font, ytick=yticks[2], show_y=show)
        for ax in (ax_s, ax_t, ax_l):
            for x in (2, 4, 6):
                ax.axvline(x - 0.5, color=C_SPLIT, lw=1.0, zorder=4)
        ax_s.set_title(title, fontproperties=font, fontsize=7, pad=3)
        ax_l.set_xticks(list(range(8)))
        ax_l.set_xticklabels([str(k) for k in range(8)], fontsize=7)
    fig.supxlabel("virtual full-window index $k$  (bars = rung boundaries)", fontsize=8, y=-0.02)
    add_legend(
        fig,
        [
            Patch(facecolor=C_MARK, label="BOS / EOS cell"),
            Patch(facecolor=C_CE, label="CE only"),
            Patch(facecolor=C_MSE, label="MSE only"),
            Patch(facecolor=C_NONE, edgecolor="#b0b7bd", label="truncated (not in window)"),
        ],
        ncol=4,
        fontsize=7,
    )
    save(fig, "relf_masks")


def _relf_f_labels(w: int = 8, s: int = 2) -> list[str]:
    n_rung = w // s
    names = ["1-ε"] + [rf"$L_{n_rung - 2 - r}$" for r in range(n_rung - 1)]
    out: list[str] = []
    for name in names:
        out.extend([name] * s)
    return out


def _loss_from_t(t_labs: list[str]) -> list[str]:
    out: list[str] = []
    for x in t_labs:
        if not x:
            out.append("")
        elif x == "1-ε":
            out.append("CE")
        else:
            out.append("MSE")
    return out


def _mark_idx(ax: plt.Axes, i: int) -> None:
    ax.add_patch(
        plt.Rectangle(
            (i - 0.48, -0.48),
            0.96,
            0.96,
            fill=False,
            edgecolor="#922b21",
            linewidth=1.5,
            zorder=8,
        )
    )


def plot_relf_emit(font: mpl.font_manager.FontProperties) -> None:
    # Two left-to-right rollouts, W=8, S=2. Column spacing matches relf_masks.
    w = 8
    f_labs = _relf_f_labels(w, 2)
    seq_color = {"": C_NONE, "BOS": C_MARK, "EOS": C_MARK, "·": C_TOK}
    loss_color = {"CE": C_CE, "MSE": C_MSE, "": C_NONE}

    def left_clip(k_bos: int) -> tuple[list[str], list[str], list[str], int]:
        seq = [""] * k_bos + ["BOS"] + ["·"] * (w - 1 - k_bos)
        t_labs = [""] * k_bos + f_labs[k_bos:]
        return seq, t_labs, _loss_from_t(t_labs), k_bos

    def right_clip(k_eos: int) -> tuple[list[str], list[str], list[str], int]:
        seq = ["·"] * k_eos + ["EOS"] + [""] * (w - 1 - k_eos)
        t_labs = f_labs[: k_eos + 1] + [""] * (w - 1 - k_eos)
        return seq, t_labs, _loss_from_t(t_labs), k_eos

    strips = [
        (
            "First token (preroll)",
            [left_clip(k) for k in (6, 4, 2, 0)],
            [r"BOS at $k{=}6$", r"$k{=}4$", r"$k{=}2$", r"$k{=}0$: emit BOS"],
        ),
        (
            "Last token (postroll)",
            [right_clip(k) for k in (6, 4, 2, 0)],
            [r"EOS at $k{=}6$", r"$k{=}4$", r"$k{=}2$", r"$k{=}0$: emit EOS"],
        ),
    ]
    fig = plt.figure(figsize=(9.6, 6.35), layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=0.06)
    yticks = ["seq", "time", "loss"]
    for subfig, (strip_title, frames, titles) in zip(subfigs, strips):
        subfig.suptitle(strip_title, fontsize=8, fontproperties=font, y=1.02)
        axes = subfig.subplots(3, 4, height_ratios=[0.72, 1.05, 0.72])
        for j, ((seq, t_labs, loss, mark), title) in enumerate(zip(frames, titles)):
            ax_s, ax_t, ax_l = axes[0, j], axes[1, j], axes[2, j]
            show = j == 0
            _fill_row(ax_s, [seq_color[x] for x in seq], seq, font, ytick=yticks[0], show_y=show)
            _fill_row(ax_t, [_t_color(x) for x in t_labs], t_labs, font, ytick=yticks[1], show_y=show)
            _fill_row(ax_l, [loss_color[x] for x in loss], loss, font, ytick=yticks[2], show_y=show)
            _mark_idx(ax_s, mark)
            for ax in (ax_s, ax_t, ax_l):
                for x in (2, 4, 6):
                    ax.axvline(x - 0.5, color=C_SPLIT, lw=1.0, zorder=4)
            ax_s.set_title(title, fontproperties=font, fontsize=7, pad=3)
            ax_l.set_xticks(list(range(w)))
            ax_l.set_xticklabels([str(k) for k in range(w)], fontsize=7)
    fig.supxlabel(
        r"virtual full-window index $k$  (left $\to$ right = successive $G$ frames; $u$ moves by $S$)",
        fontsize=8,
        y=-0.02,
    )
    add_legend(
        fig,
        [
            Patch(facecolor=C_MARK, label="BOS / EOS cell"),
            Patch(facecolor=C_CE, label="CE only (read / pop)"),
            Patch(facecolor=C_MSE, label="MSE only"),
            Patch(facecolor=C_NONE, edgecolor="#b0b7bd", label="truncated (not in window)"),
        ],
        ncol=4,
        fontsize=7,
    )
    save(fig, "relf_emit")


def plot_relf_clean(font: mpl.font_manager.FontProperties) -> None:
    # Left: mid-window remainder. Right: postroll remainder (EOS clip; F unshifted).
    loss_color = {"CE": C_CE, "MSE": C_MSE, "": C_NONE}
    cases = [
        (
            "No clip (mid)",
            ["clean", "1-ε", r"$L_2$", r"$L_2$", r"$L_1$", r"$L_1$", r"$L_0$", r"$L_0$"],
            ["", "CE", "MSE", "MSE", "MSE", "MSE", "MSE", "MSE"],
        ),
        (
            "Right clip: EOS at $k{=}4$",
            ["clean", "1-ε", r"$L_2$", r"$L_2$", r"$L_1$", "", "", ""],
            ["", "CE", "MSE", "MSE", "MSE", "", "", ""],
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 1.95), layout="constrained")
    for j, (title, t_labs, loss) in enumerate(cases):
        t_cols = [C_KNOWN if x == "clean" else _t_color(x) for x in t_labs]
        show = j == 0
        _fill_row(axes[0, j], t_cols, t_labs, font, ytick="time", show_y=show)
        _fill_row(
            axes[1, j],
            [loss_color[x] for x in loss],
            loss,
            font,
            ytick="loss",
            show_y=show,
        )
        for ax in (axes[0, j], axes[1, j]):
            ax.axvline(0.5, color="#7f8c8d", lw=1.1, zorder=4)
            for x in (2, 4, 6):
                ax.axvline(x - 0.5, color=C_SPLIT, lw=1.1, zorder=4)
        axes[0, j].set_title(title, fontproperties=font, fontsize=7, pad=3)
        axes[1, j].set_xticks(list(range(8)))
        axes[1, j].set_xticklabels([str(k) for k in range(8)], fontsize=7)
    fig.supxlabel(
        "window index $k$  (gray = known/unknown; black = rung)",
        fontsize=8,
        y=-0.02,
    )
    add_legend(
        fig,
        [
            Patch(facecolor=C_KNOWN, label="known, pinned clean"),
            Patch(facecolor=C_CE, label="CE only"),
            Patch(facecolor=C_MSE, label="MSE only"),
            Patch(facecolor=C_NONE, edgecolor="#b0b7bd", label="truncated (not in window)"),
        ],
        ncol=4,
        fontsize=7,
    )
    save(fig, "relf_clean")


def _plot_cont_strip(
    font: mpl.font_manager.FontProperties,
    frames: list[tuple[list[str], list[str], list[str], list[int], list[int]]],
    titles: list[str],
    stem: str,
    xlabel: str,
    *,
    figsize: tuple[float, float],
) -> None:
    seq_color = {"": C_NONE, "K": C_KNOWN, "·": C_TOK, "P": C_TOK}
    loss_color = {"CE": C_CE, "MSE": C_MSE, "": C_NONE}
    n = len(frames)
    fig, axes = plt.subplots(
        3,
        n,
        figsize=figsize,
        layout="constrained",
        height_ratios=[0.72, 1.05, 0.72],
    )
    yticks = ["seq", "time", "loss"]
    for j, ((seq, t_labs, loss, marks, splits), title) in enumerate(zip(frames, titles)):
        ax_s, ax_t, ax_l = axes[0, j], axes[1, j], axes[2, j]
        show = j == 0
        _fill_row(ax_s, [seq_color[x] for x in seq], seq, font, ytick=yticks[0], show_y=show)
        _fill_row(ax_t, [_t_color(x) for x in t_labs], t_labs, font, ytick=yticks[1], show_y=show)
        _fill_row(ax_l, [loss_color[x] for x in loss], loss, font, ytick=yticks[2], show_y=show)
        for i in marks:
            _mark_idx(ax_s, i)
        for ax in (ax_s, ax_t, ax_l):
            for s in splits:
                ax.axvline(s - 0.5, color=C_SPLIT, lw=1.2, zorder=4)
        ax_s.set_title(title, fontproperties=font, fontsize=7, pad=3)
        ax_l.set_xticks(list(range(len(seq))))
        ax_l.set_xticklabels([str(k) for k in range(len(seq))], fontsize=6)
    fig.supxlabel(xlabel, fontsize=8, y=-0.02)
    add_legend(
        fig,
        [
            Patch(facecolor=C_TOK, label="prefix / unknown token"),
            Patch(facecolor=C_KNOWN, label="known remainder, t=1"),
            Patch(facecolor=C_CE, label="CE only (read / pop)"),
            Patch(facecolor=C_MSE, label="MSE only"),
            Patch(facecolor=C_NONE, edgecolor="#b0b7bd", label="unused"),
        ],
        ncol=5,
        fontsize=7,
    )
    save(fig, stem)


def plot_belf_clean_cont(font: mpl.font_manager.FontProperties) -> None:
    # W=4, prefix L=6 so r=2. Finish mixed block, then a full unknown block.
    n = 12

    def frame(n_kv: int, n_known: int, n_unk: int, *, decode: bool) -> tuple:
        n_pad = n - n_kv - n_known - n_unk
        seq = [""] * n_pad + ["P"] * n_kv + ["K"] * n_known + ["·"] * n_unk
        t_unk = ["1-ε"] * n_unk if decode else ["t"] * n_unk
        t_labs = [""] * n_pad + [""] * n_kv + ["1"] * n_known + t_unk
        loss_unk = ["CE"] * n_unk if decode else ["MSE"] * n_unk
        loss = [""] * (n_pad + n_kv + n_known) + loss_unk
        cur = n_pad + n_kv
        marks = list(range(cur + n_known, cur + n_known + n_unk)) if decode else []
        splits = [cur]
        if n_known:
            splits.append(cur + n_known)
        return seq, t_labs, loss, marks, splits

    _plot_cont_strip(
        font,
        [
            frame(4, 2, 2, decode=False),
            frame(4, 2, 2, decode=True),
            frame(8, 0, 4, decode=False),
            frame(8, 0, 4, decode=True),
        ],
        [
            r"$r{=}2$, flow",
            r"decode: emit 2",
            r"next block, flow",
            r"decode: emit 4",
        ],
        "belf_clean_cont",
        r"pack index  (left $\to$ right = successive $G$ frames; $W{=}4$, prefix $L{=}6$)",
        figsize=(9.8, 2.55),
    )


def plot_relf_clean_cont(font: mpl.font_manager.FontProperties) -> None:
    # 左切混合窗：K 从窗右端进入，每次 hop 左移 S；其右未知沿 F 从 L0 升到 1-ε 才 CE。
    w, s = 8, 2
    f_labs = _relf_f_labels(w, s)

    def frame(k_k: int) -> tuple:
        k_new = k_k + 1
        seq = [""] * k_k + ["K"] + ["·"] * (w - 1 - k_k)
        t_labs = [""] * k_k + ["1"] + f_labs[k_new:]
        loss = []
        for i, lab in enumerate(t_labs):
            if i <= k_k or not lab:
                loss.append("")
            elif lab == "1-ε":
                loss.append("CE")
            else:
                loss.append("MSE")
        splits = [k_new] + [s * r for r in range(1, w // s)]
        return seq, t_labs, loss, [k_new], splits

    _plot_cont_strip(
        font,
        [frame(k) for k in (6, 4, 2, 0)],
        [
            r"$r{=}1$, MSE ($L_0$)",
            r"MSE ($L_1$)",
            r"MSE ($L_2$)",
            r"CE: emit 1",
        ],
        "relf_clean_cont",
        r"virtual full-window index $k$  (left $\to$ right = $K$ moves left by $S$; $W{=}8$, $S{=}2$)",
        figsize=(9.6, 2.55),
    )


def main() -> None:
    font = setup_font()
    plot_attn(font)
    plot_belf_pack(font)
    plot_belf_clean(font)
    plot_belf_clean_cont(font)
    plot_relf_masks(font)
    plot_relf_clean(font)
    plot_relf_clean_cont(font)
    plot_relf_emit(font)
    for stem in (
        "belf_attn",
        "belf_pack",
        "belf_clean",
        "belf_clean_cont",
        "relf_attn",
        "relf_masks",
        "relf_clean",
        "relf_clean_cont",
        "relf_emit",
    ):
        print(HERE / f"{stem}.png")


if __name__ == "__main__":
    main()
