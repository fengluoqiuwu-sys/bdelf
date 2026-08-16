# report（开题报告）

`idea-kickoff` 的配套流程（本文件不是独立 skill）：规格 **PASS** 之后写给人看的开题报告。  
**不是**综述（[survey.md](survey.md)）；**不是**规格（[spec.md](spec.md)）。**不**改代码、不占 GPU。

从已冻结文稿**改写**成短报告，不要空转抄一遍；也不要写 `I-{n}`。

## 硬边界

- 只写父代理给定夹内的 `proposal/`（tex + bib + pdf）或 fallback 的 `proposal.md`。
- 禁止写 `survey.md` / `scope.md` / `stage.md` / `README.md`；禁止改 `milestones.md`。
- 面向 **subagent**：父代理须按 rule「subagent 模型」选 `model`（`inherit` / `auto` / `composer-2.5`，禁 `*-fast`）。
- 进度节对齐 `milestones.md`；实验方案节对齐 `protocol.md`；相关工作节对齐 `survey.md`（短引，不要再写一篇综述）。

## 主循环

```
父代理传入：目标夹
  → 读 survey / scope / claims / protocol / milestones / SPEC / risk
  → 探测 LaTeX（见下）
  → 有工具链：写 proposal/proposal.tex + refs.bib，立刻编译 PDF
  → 无工具链：写 proposal.md；不建 proposal/；不假装有 PDF
  → 回报 pdf 或 md
```

### 探测与编译

仓库根：

```bash
command -v latexmk && command -v xelatex && kpsewhich ctexart.cls
```

有三者：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -cd \
  temp/ideas/<name>/proposal/proposal.tex
```

- 成功须有 `proposal/proposal.pdf`。tex 写错：改源再编，不要因此改 md。
- 不要用 pdflatex 凑合。
- **无工具链**（缺命令或引擎起不来）：fallback `proposal.md`，原因写入回报（父代理记 `stage.md`）。tex 内容错误不走这条。

### `proposal.md`（fallback）

```markdown
# 开题：<短标题>

## 背景与意义
…
## 相关工作
…（短引 survey.md，不是第二篇综述）
## 研究内容与贡献
…（scope.md / claims.md；主 claim 标 hypothesis）
## 技术路线
…（SPEC.md / math.md）
## 实验方案与成功判据
…（protocol.md；写明未授权执行）
## 进度
…（milestones.md；M0 在先。路线权威是该文件）
## 风险与预案
…（risk.md）
## 预期产出
潜力档 + 一篇论文量级的 hypothesis
## 非目标与 Kill
…
```

### `proposal/proposal.tex` + `refs.bib`

中文 `ctexart` + XeLaTeX；`biblatex` / `biber`。章节与上面 fallback 相同。

```latex
\documentclass[11pt,a4paper]{ctexart}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[backend=biber,style=numeric,sorting=none]{biblatex}
\addbibresource{refs.bib}

\title{开题报告：<短标题>}
\author{}
\date{\today}

\begin{document}
\maketitle

\section{背景与意义}
…

\section{相关工作}
…（\cite{…}；对应 survey.md）

\section{研究内容与贡献}
…

\section{技术路线}
…

\section{实验方案与成功判据}
…

\section{进度}
…

\section{风险与预案}
…

\section{预期产出}
…

\section{非目标与 Kill}
…

\printbibliography
\end{document}
```

`refs.bib`：最近邻与基准至少各有条目。无文献则去掉 cite / `\printbibliography`。

回报父代理（短）：`pdf` 或 `md`、相对路径、fallback 原因（若有）。勿贴全文。
