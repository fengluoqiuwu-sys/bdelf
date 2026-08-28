# BELF / RELF：数学逻辑

本文独立陈述两族共用的流几何与各自的条件分解。符号与默认值以 [`latex/belf-relf.tex`](latex/belf-relf.tex) 为准；工程接口见 [`README.md`](README.md)。全程 **\(t=1\) 干净、\(t=0\) 纯噪**（与仓库 Cola Stage-2 的时间轴相反）。

## 1. 数据点与 rectified flow

入口给出 token 对齐后验 \(q_\phi(z\mid x)=\mathcal{N}(\mu(x),\mathrm{diag}\,\sigma^2(x))\)。训练可用样本 \(z=\mu+\sigma\odot\epsilon\)，或确定性 \(\mu\)。流空间是入口输出维 \(X=\) `latent_dim`（须与 artifact 一致），与 \(G\) 隐层宽 \(D=\) `n_embd` 无关。白化留在 \(X\)；\(G\) 茎一层 \(\mathrm{Linear}(X\to D)\)，FinalLayer 回到 \(X\)。干净端 \(x_0\in\mathbb{R}^{N\times X}\)（默认后验样本；对照均值）。独立噪声 \(\varepsilon\sim\mathcal{N}(0,\sigma^2 I)\)。线性插值

\[
z_t = t\,x_0+(1-t)\,\varepsilon,\qquad t\in[0,1].
\]

沿该路径的常速度 \(v^\star=x_0-\varepsilon\)。\(G\) 输出 x-pred \(\hat x_0=G_\theta(z_t,t,\ldots)\)，再换速度

\[
\hat v=\frac{\hat x_0-z_t}{\max(1-t,\,\varepsilon_t)},
\]

\(\varepsilon_t=\) `vel_eps`，避免 \(t\to 1\) 除零。v-MSE 监督 \(\|\hat v-v_{\mathrm{tgt}}\|^2\)（无 CFG 时 \(v_{\mathrm{tgt}}=v^\star\)）。这与「带权重 \(1/(1-t)^2\) 的 \(x_0\)-MSE」在 \(\varepsilon_t=0\) 时等价。

**一致性：** 右段加噪所用的干净端、v-MSE 靶、以及推理写入 KV 的对象必须在同一空间 \(X\)。若靶是 \(\mu\)，则 `commit_x0hat` 提交的 \(\hat x_0\) 也对准 \(\mu\)。左段用采样 \(z\) 只改变条件随机性，不改路径端点。两条 commit 策略都合法，不可混评：提交 \(\hat x_0\)；或对照臂再 encode **采 \(z\)**（`sample=(x0_source=="z")`），写入空间与 `x0_source` 一致。外部前缀完整块仍 `sample=False`。

ELF 原文是整句一个标量 \(t\)、全程双向，定义的是联合流。本规格把同一插值接到**条件**分解上：已提交前缀进 KV，流只作用在当前块或窗的未知槽。

## 2. 梯子

训练连续时间服从 \(\mathrm{logit}(t)\sim\mathcal{N}(P_m,P_s^2)\)，即 \(t=\sigma(P_m+P_s Z)\)，\(Z\sim\mathcal{N}(0,1)\)。实现用该分布的分位函数 \(Q\)：端点 \(Q(0)=0\)、\(Q(1)=1-\varepsilon\)，开区间

\[
Q(p)=\sigma\bigl(P_m+P_s\,\Phi^{-1}(p)\bigr),\qquad p\in(0,1),
\]

再夹到 \([0,1-\varepsilon]\)。档位在**概率轴**上等分，不是在 \(t\) 上等间隔。`time_step` \(T\) 是每块 / 每窗的**流**步数（亦为 \(G\) 次数），须 \(T\ge 4\)。主体没有出口 CE，因而不再另留一档 decode 跳；梯子按 \(T\) 段均分（\(T+1\) 个点），不是旧的 \(T-1\) 段流 + 1 档 CE：

\[
L_i=Q\!\left(\frac{i}{T}\right),\qquad i=0,\ldots,T.
\]

因而 \(L_0=0\)，\(L_T=Q(1)=1-\varepsilon\)。第 \(i\) 跳（\(i=0,\ldots,T-1\)）在 \(t=L_i\) 上跑 \(G\)，步长 \(\Delta t=L_{i+1}-L_i\)，末次 Euler 到 \(1-\varepsilon\)。推理在末次流的 \(\hat x_0\) 上直接 VAE-dec，**不再**于 \(t=1-\varepsilon\) 多跑一次 \(G\)。训练与推理共用这把确定梯子：推理不再从 logit-normal 随机采样，也不使用 \(\mathrm{linspace}(0,1)\)。默认 \(T=16\) 不变（100m 仍 \(W=T\)、RELF 仍 \(S\cdot T=W\)）；变的是分位分母与是否多一档 CE。

## 3. 条件通道

\(G\) 一份网络。AdaLN-Zero **逐列**以 \((t,w_{\mathrm{sc}},m)\) 为条件（\(m\) 不是序列 token，不改变 2L 长度）：

| 列 | \(t\) | \(m\) | \(w_{\mathrm{sc}}\) |
|---|---|---|---|
| 左段 / 已知 / PAD | \(1\) | 不加 | 不以之为条件 |
| 未知（一律流） | 该列流时间 \(L_i\) | `denoise` | `sc_cfg` 为真时以之为条件 |

不再设 `m=decode` 档：读出用末次流的 \(\hat x_0\)，不在 \(t=1-\varepsilon\) 另开 hop。BOS 前、EOS 后丢掉的格不在窗，不加 \(m\)。不用 ELF in-context control tokens（`num_time_tokens=0`）。

2L 布局 \([\mathrm{sg}(h_{\mathrm{left}})\mid h_t]\)：已完成段写入 KV 后**组内双向、组间单向**；右段可见左段。BELF 的组是 `block_size` \(W\)。RELF 已完成段按加载入口块长（必须为 1，即逐 token 因果）；窗内组是档宽 \(S\)。

## 4. 损失（只 v-MSE）与出口

主体损失只有速度均方误差，**没有出口 CE**（亦无 `lambda_ce` / `ce_detach_g`）。逐 token \(\|\cdot\|^2\) 对 \(X\) 取平均。左段与已知槽不进入 \(\mathcal{L}\)。

\[
\mathcal{L}
=\lambda_{\mathrm{mse}}\,\mathcal{L}_{\mathrm{mse}}.
\]

- **BELF**：抽一跳 \(i\sim\mathrm{Unif}\{0,\ldots,T-1\}\)，\(t=L_i\)，右段全部未知有效位算 v-MSE，\(m=\mathrm{denoise}\)（可 CFG）。末跳 \(i=T-1\) 在 \(t=L_{T-1}\)（不是 \(1-\varepsilon\)），同样 Euler / 同样 CFG。
- **RELF**：切完 \(F\) 后对仍在窗的未知真槽一律 v-MSE（\(m=\mathrm{denoise}\)，可 CFG）。窗内最左档是 \(t=L_{T-1}\)，不是 \(1-\varepsilon\)；不再用 \(m_c\)/\(m_d\) 拆 CE。未知真槽数为 0 时该项为 0。

出口**锁死** VAE-dec：\(\hat x_0\in X\) 直接走加载的 decoder 得 logits。无 `exit` 键，不做 `linear`（ELF 隐状态 \(D\to V\)）对照。训练主体不算 CE；VAE-dec 只在 **推理读出** 与可训档的 \(\mathcal{L}_{\mathrm{s1}}\) 重建里出现。须具备 VAE-dec。

可训 latent 时另加、且不并进 \(\mathcal{L}\)：

\[
\mathcal{L}_{\mathrm{s1}}
=\lambda_{\mathrm{vae}}\,\mathrm{CE}_{\mathrm{rec}}
+\beta\,\mathrm{KL}(q_\phi\|N(0,I))
+\lambda_{\mathrm{mask}}\,\mathcal{L}_{\mathrm{mask}}
+\lambda_{\mathrm{ref}}\,\mathrm{KL}(q_\phi\|q_{\phi_{\mathrm{ref}}}).
\]

重建与 BERT-mask 都经 VAE-dec。\(q_{\phi_{\mathrm{ref}}}\) 是 s2 开始时加载器给出的 encoder 冻结副本。

## 5. CFG

两轴独立。\(w_{\mathrm{sc}}\) 只进入右段未知槽 AdaLN（未知一律 denoise）；左段 / 已知 / PAD 不以之为条件。推理单次前向；扫描 \(w_{\mathrm{sc}}\) 不重算前缀 KV。无「纯 decode 步」：每跳都可跑 teacher。

`sc_cfg` 为真时，每样本采 \(z\sim\mathcal{N}(P_m^{\mathrm{sc}},(P_s^{\mathrm{sc}})^2)\)，\(u=\sigma(z)\)，再

\[
a=1+w^{\mathrm{sc}}_{\min},\quad
b=1+w^{\mathrm{sc}}_{\max},\quad
w_{\mathrm{sc}}=a\,(b/a)^{u}-1.
\]

并独立抽 \(g\sim\mathrm{Bern}(p_g^{\mathrm{sc}})\)。每跳两个 `no_grad` teacher（uncond 通道全 0；cond 通道 \(\mathrm{sg}(\hat x_0^{u})\)）得 \(v_u,v_c\)，学生

\[
v_{\mathrm{tgt}}=\begin{cases}
v_z+\bigl(1-1/w_{\mathrm{sc}}\bigr)(v_c-v_u) & g=1,\\
v_z & g=0.
\end{cases}
\]

修正施加于右段未知列。\(g=0\) 时 sc 通道为 0，但未知槽 AdaLN 仍以已采样的 \(w_{\mathrm{sc}}\) 为条件。`sc_cfg` 为假：无通道、\(v_{\mathrm{tgt}}=v_z\)。不可单独再开 `self_cond`。

\(w_{\mathrm{ctx}}\) 不进 AdaLN。训练以 \(p_{\mathrm{drop}}^{\mathrm{ctx}}\) 丢弃 2L 左段。推理默认只跑带前缀一次前向；外推时无条件支路为空前缀、仅当前块/窗。RELF 最右 \(S\) 个新噪声槽 sc 恒为 0。

## 6. BELF：块条件流

序列切成块 \(z_0^{(b)}\in\mathbb{R}^{W\times X}\)。联合

\[
p_G(z_0)=\prod_b p_G\bigl(z_0^{(b)}\mid z_0^{(<b)}\bigr).
\]

每一因子是条件 rectified flow：**未知槽共享同一标量 \(t\)**；已知余数钉 \(t=1\)。去噪块长 \(W\) 为 BELF 的 `block_size`：100m 默认 16，主跑 \(W=T\)。加载入口块长须 \(\in\{1,W\}\)（逐 token 因果，或与本题 \(W\) 相同）；入口注意力按加载结果，不随 \(W\) 改写。

训练：序列长度须被 \(W\) 整除，否则报错；抽一跳 \(i\sim\mathrm{Unif}\{0,\ldots,T-1\}\)，把 \(t=L_i\) 广播到未知槽，对该跳未知有效位算 v-MSE。推理 `block_generate`：跳 \(i=0,\ldots,T-1\) 均 Euler \(\Delta t=L_{i+1}-L_i\)（末流从 \(L_{T-1}\) 走到 \(1-\varepsilon\)）；然后用末次流的 \(\hat x_0\) 走 VAE-dec，不再多一次 \(G\)。已知槽每跳覆写 encoder 干净码。SDE churn 关在最后一次流（跳 \(T-1\)）。推理末块可短。

`cond_mode=clean`：前缀长 \(L\) 时 \(r=L\bmod W\)。完整块进 KV；当前块槽 \([0,r)\) 为已知余数（\(t=1\)，不进损失）。一块一次读出，不会在同一块内把刚 decode 的 token 立刻改成已知余数再继续流。

训练已知条件不得含未来信息。右段 unknown Q **不可见同块 PAD K**（入口块长 1 也会发生）；PAD 仍钉 \(t=1\)、不进损失。可训档（`full` / `mid`）硬拒入口块长 \(\neq 1\)，块内未来看不见，不必二次 encode。仅冻结档且入口块长 \(=W\) 时，余数（抽到 `clean_block_prob`）再 encode 一份条件句：当前块 \([r,W)\) 写成 PAD，已知覆写与 PAD 干净码用这份；插值靶 \(x_0\) 仍来自整句 encode。`rem` 全 0 时跳过第二次 encode。左段整句教师强制不改。

链式法则写的是 \(p_G(z^{(b)}\mid z^{(<b)})\)。推理左段默认是提交的 \(\hat x_0\)。训练默认左段仍是 encoder 干净码；以概率 `self_left_prob`（默认 \(0.25\)，eval / 生成视为 0）把左段换成 **stop-grad 的末流 \(\hat x_0\)**：在 CFG 之前，用 GT 左 + 右段钉 \(t=L_{T-1}\)、\(m=\mathrm{denoise}\) 做一次 `no_grad` \(G\)（与推理末跳同条件，取 x-pred，不 Euler），按样本 Bernoulli 替换 `h_left`，再跑 teacher / 学生。已知余数与 PAD 仍钉 encoder 干净码。这不是当前训练 hop 的 \(\hat x_0\)（随机 \(t\) 离提交码太远）。

## 7. RELF：局部时间场

须 \(S\cdot T=W\)。窗内位置 \(k\) 有局部时间，插值独立：

\[
z^k=t_k z_0^k+(1-t_k)\varepsilon^k.
\]

整窗模板（freeroll）

\[
F_k=L_{T-1-\lfloor k/S\rfloor},\qquad k=0,\ldots,W-1.
\]

档 \(r\) 占槽 \([rS,(r+1)S)\)，值为 \(L_{T-1-r}\)。最左档 \(t=L_{T-1}\)（末流起点），右组纯噪 \(L_0\)。窗内不出现 \(L_T=1-\varepsilon\)：那是末档 Euler 的终点。每档 \(S\) 槽共享同一 \(L_i\)，不是逐槽独立采样 \(t\)。

**半群（不依赖 \(G\)）。** 推理稳态：每步一次 \(G\)，未知档（含最左）各 Euler 升一档（最左从 \(L_{T-1}\) 走到 \(1-\varepsilon\)）；用最左档该步的 \(\hat x_0\) 读出并 pop 这 \(S\) 槽，右端补 \(S\) 个 \(L_0\) 新噪声后，时间场回到 \(F\)。证明只依赖梯子算术与 \(S\cdot T=W\)。Euler 有局部截断误差；SDE churn 关在升档侧（含最左末流）。

**截断。** 虚拟满窗铺 \(F\)。窗起点 \(u\) 按文档 \(S\) 对齐。对格 \(k\)、文档下标 \(j=u+k\)：\(j<i_{\mathrm{bos}}\) 丢掉（句首切），\(j>i_{\mathrm{eos}}\) 丢掉（句尾切），否则 \(t_k=F_k\)。丢掉的位置不在窗：不赋 \(t=0\)、不加 \(m\)，也不把留下的梯子挪到另一头凑满 \(W\)。两切独立可叠加。PAD 不是边界。抽窗须保住 BOS / EOS。

**余数爬梯。** 前缀 \(L\) 不对齐档界时 \(r=L\bmod S>0\)。新词从 \(L_0\) 爬到 \(L_{T-1}\) 再 Euler 到 \(1-\varepsilon\) 才 **读出 / pop**，禁止钉满窗 \(F\) 左侧一次读出。hop \(h=0,\ldots,T-1\) 的虚拟余数起点 \(k_0=S(T-h-1)\)；\(h<T-1\) 时最左流档已切掉（只更噪档 v-MSE、不 pop）；\(h=T-1\) 时 \(k_0=0\)，未知最左档算 v-MSE 并 Euler，推理再 VAE-dec 后 pop。RELF 左前缀同样可按 `self_left_prob` 换成末流 \(\mathrm{sg}(\hat x_0)\)。

**对齐前缀冷启动。** \(L>0,\,r=0\) 且尚无 \(z_{\mathrm{carry}}\)（对齐前缀第一次滚动）时，不得把整窗标满 \(F\) 当 freeroll：训练从未见过「纯噪却钉左档 \(t=L_{T-1}\)」。须走与余数相同的 \(T\) 帧爬梯（known 为空，\(n_{\mathrm{write}}=S\)，末 hop 再 pop 并写出 carry）。**\(r=0\) 仅在已有 carry 时跳过爬梯**（preroll / 上次 pop 之后）。无条件 \(L=0\) preroll 不变。

训练抽余数：先抽 \(h\) 得 \(k_0=S(T-1-h)\)，再 \(\mathrm{bos\_cut}=(u+k_0)<i_{\mathrm{bos}}\)（留下的第一格是否仍在 BOS 前）。真 preroll（留下的格仍有 \(j<i_{\mathrm{bos}}\)）不抽；虚拟 \(u<0\) 但留下的格已过 BOS 的早 hop 仍可抽。未做句首切时以 `clean_block_prob` 抽一帧 \((h,r)\)。

RELF 的联合由「pop 后条件于 KV 的窗上局部时间流」迭代定义，不是 ELF 整句联合流的条件化，也不是每槽独立边缘的乘积路径（档内强制共享 \(L_i\)）。学习目标是该规定 \((z,t)\) 场上的速度回归；只要推理每一步的 \((z,t)\) 落在训练由截断给出的场上，回归是良定的。

## 8. 自检要点

1. **时间轴。** 干净端 \(t=1\)。2L 左段标 \(t=1\)，不对噪声 \(t\) 做 v-MSE。
2. **半群。** RELF 滑窗后梯子复位只依赖 \(\Delta t\) 与 \(S\cdot T=W\)；状态 Euler 的截断误差是数值问题，用末流关 churn 避免读出前把左槽拉回噪声。
3. **分母。** 训练 \(t\) 落在 \(L_0,\ldots,L_{T-1}\)，末点 \(L_T=1-\varepsilon\) 只作 Euler 终点；分母有 \(\varepsilon_t\)，v-MSE 有界。丢掉的格不进损失。
4. **左条件。** 默认训练左段是 encoder 码，推理是提交的 \(\hat x_0\)。`self_left_prob>0` 时按样本用末流 \(t=L_{T-1}\) 的 \(\mathrm{sg}(\hat x_0)\) 替换训练左段（见第 6 节）。
5. **出口。** 锁死 VAE-dec，无层数、无 `exit`/`linear`。主体 \(\mathcal{L}\) 不含 CE。推理 BELF 拼接已提交码再读当前块；RELF 读出前缀是 \(G\) 左段 \(\hat x_0\)，丢掉格不 scatter 进因果前缀。
6. **两族。** 不同 \(t\) 场是不同输入分布。共享结构合法；共享一份权重会让同一网络逼近两种条件场。规格是两个模型族、两套训练剖面。
7. **日程档。** `fast` / `mid` / `full` 是 checkpoint 变体（`cache/checkpoints/{fast|mid|full}/…`），与 `latent_tune=mid`（15B 解冻）不是同一键。`mid` 仅 Stage1：全局序列批 128、主预算 10B、`eval_step=1000`、`ema_decay=0.999`，用于中档验证。
