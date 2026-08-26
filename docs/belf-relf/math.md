# BELF / RELF：数学逻辑

本文独立陈述两族共用的流几何与各自的条件分解。符号与默认值以 [`latex/belf-relf.tex`](latex/belf-relf.tex) 为准；工程接口见 [`README.md`](README.md)。全程 **\(t=1\) 干净、\(t=0\) 纯噪**（与仓库 Cola Stage-2 的时间轴相反）。

## 1. 数据点与 rectified flow

入口给出 token 对齐后验 \(q_\phi(z\mid x)=\mathcal{N}(\mu(x),\mathrm{diag}\,\sigma^2(x))\)。训练可用样本 \(z=\mu+\sigma\odot\epsilon\)，或确定性 \(\mu\)。映射到 \(G\) 的宽度 \(D\) 后，记干净端为 \(x_0\in\mathbb{R}^{N\times D}\)（默认后验样本经 Linear；对照均值）。独立噪声 \(\varepsilon\sim\mathcal{N}(0,\sigma^2 I)\)。线性插值

\[
z_t = t\,x_0+(1-t)\,\varepsilon,\qquad t\in[0,1].
\]

沿该路径的常速度 \(v^\star=x_0-\varepsilon\)。\(G\) 输出 x-pred \(\hat x_0=G_\theta(z_t,t,\ldots)\)，再换速度

\[
\hat v=\frac{\hat x_0-z_t}{\max(1-t,\,\varepsilon_t)},
\]

\(\varepsilon_t=\) `vel_eps`，避免 \(t\to 1\) 除零。v-MSE 监督 \(\|\hat v-v_{\mathrm{tgt}}\|^2\)（无 CFG 时 \(v_{\mathrm{tgt}}=v^\star\)）。这与「带权重 \(1/(1-t)^2\) 的 \(x_0\)-MSE」在 \(\varepsilon_t=0\) 时等价。

**一致性：** 右段加噪所用的干净端、v-MSE 靶、以及推理写入 KV 的对象必须在同一空间。若靶是 \(\mu\)，则 `commit_x0hat` 提交的 \(\hat x_0\) 也对准 \(\mu\)。左段用采样 \(z\) 只改变条件随机性，不改路径端点。两条 commit 策略（提交 \(\hat x_0\) vs 再 encode 采 \(z\)）都合法，不可混评。

ELF 原文是整句一个标量 \(t\)、全程双向，定义的是联合流。本规格把同一插值接到**条件**分解上：已提交前缀进 KV，流只作用在当前块或窗的未知槽。

## 2. 梯子

训练连续时间服从 \(\mathrm{logit}(t)\sim\mathcal{N}(P_m,P_s^2)\)，即 \(t=\sigma(P_m+P_s Z)\)，\(Z\sim\mathcal{N}(0,1)\)。实现用该分布的分位函数 \(Q\)：端点 \(Q(0)=0\)、\(Q(1)=1-\varepsilon\)，开区间

\[
Q(p)=\sigma\bigl(P_m+P_s\,\Phi^{-1}(p)\bigr),\qquad p\in(0,1),
\]

再夹到 \([0,1-\varepsilon]\)。档位在**概率轴**上等分，不是在 \(t\) 上等间隔。`time_step` \(T\) 是每块 / 每窗的 \(G\) 次数（\(T-1\) 次流 + 1 次 decode），须 \(T\ge 4\)：

\[
L_i=Q\!\left(\frac{i}{T-1}\right),\qquad i=0,\ldots,T-1.
\]

因而 \(L_0=0\)，\(L_{T-1}=1-\varepsilon\)。第 \(i\) 跳（\(i=0,\ldots,T-2\)）步长 \(\Delta t=L_{i+1}-L_i\)，末次流 Euler 到 \(1-\varepsilon\)。decode 也在 \(t=1-\varepsilon\)，不再往更干净走。训练与推理共用这把确定梯子：推理不再从 logit-normal 随机采样，也不使用 \(\mathrm{linspace}(0,1)\)。\(t=1-\varepsilon\) 上没有以该 \(t\) 为起点的 v-MSE \(G\)。

## 3. 条件通道

\(G\) 一份网络。AdaLN-Zero **逐列**以 \((t,w_{\mathrm{sc}},m)\) 为条件（\(m\) 不是序列 token，不改变 2L 长度）：

| 列 | \(t\) | \(m\) | \(w_{\mathrm{sc}}\) |
|---|---|---|---|
| 左段 / 已知 / PAD | \(1\) | 不加 | 不以之为条件 |
| 未知 denoise | 该列流时间 | `denoise` | `sc_cfg` 为真时以之为条件 |
| 未知 decode | \(1-\varepsilon\) | `decode` | 不以之为条件 |

BOS 前、EOS 后丢掉的格不在窗，不加 \(m\)。不用 ELF in-context control tokens（`num_time_tokens=0`）。

2L 布局 \([\mathrm{sg}(h_{\mathrm{left}})\mid h_t]\)：已完成段写入 KV 后**组内双向、组间单向**；右段可见左段。BELF 的组是 `block_size` \(W\)。RELF 已完成段按加载入口块长（必须为 1，即逐 token 因果）；窗内组是档宽 \(S\)。

## 4. 损失拆分

流与出口按槽拆开，各自对有效位平均再 \(\lambda\) 相加：

\[
\mathcal{L}
=\lambda_{\mathrm{mse}}\,\mathcal{L}_{\mathrm{mse}}
+\lambda_{\mathrm{ce}}\,\mathcal{L}_{\mathrm{ce}}.
\]

本步无读出位置时 \(\mathcal{L}_{\mathrm{ce}}=0\)。逐 token \(\|\cdot\|^2\) 对 \(D\) 取平均。左段与已知槽不进入 \(\mathcal{L}\)。

- **BELF** 用跳号拆、不引入 \(m_d\)/\(m_c\)：\(i=0,\ldots,T-2\) 右段未知有效位只 v-MSE；\(i=T-1\) 只 CE。
- **RELF** 切完 \(F\) 后对未知真槽

\[
m_c(k)=\mathbf{1}[F_k=1-\varepsilon],\qquad
m_d(k)=\mathbf{1}[F_k<1-\varepsilon].
\]

\(m_c=1\) 则 \(m_d=0\)。\(\sum m_c=0\) 时 CE 项为 0；\(\sum m_d=0\) 时 MSE 项为 0。

同一 \(\hat x_0\) 上不同时施加「经 CFG 修正的速度」与「真词 CE」。默认 `exit=decoder`、`ce_detach_g=false`：CE 对 \(G\) 反向传播。

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

两轴独立。\(w_{\mathrm{sc}}\) 只进入右段未知 **denoise** 槽 AdaLN；decode 列与左段不以之为条件。推理单次前向；扫描 \(w_{\mathrm{sc}}\) 不重算前缀 KV。

`sc_cfg` 为真时，每样本采 \(z\sim\mathcal{N}(P_m^{\mathrm{sc}},(P_s^{\mathrm{sc}})^2)\)，\(u=\sigma(z)\)，再

\[
a=1+w^{\mathrm{sc}}_{\min},\quad
b=1+w^{\mathrm{sc}}_{\max},\quad
w_{\mathrm{sc}}=a\,(b/a)^{u}-1.
\]

并独立抽 \(g\sim\mathrm{Bern}(p_g^{\mathrm{sc}})\)。非纯 decode 步：两个 `no_grad` teacher（uncond 通道全 0；cond 通道 \(\mathrm{sg}(\hat x_0^{u})\)）得 \(v_u,v_c\)，学生

\[
v_{\mathrm{tgt}}=\begin{cases}
v_z+\bigl(1-1/w_{\mathrm{sc}}\bigr)(v_c-v_u) & g=1,\\
v_z & g=0.
\end{cases}
\]

修正仅施加于 denoise 列。\(g=0\) 时 sc 通道为 0，但 denoise 未知槽 AdaLN 仍以已采样的 \(w_{\mathrm{sc}}\) 为条件。`sc_cfg` 为假：无通道、\(v_{\mathrm{tgt}}=v_z\)。不可单独再开 `self_cond`。

\(w_{\mathrm{ctx}}\) 不进 AdaLN。训练以 \(p_{\mathrm{drop}}^{\mathrm{ctx}}\) 丢弃 2L 左段。推理默认只跑带前缀一次前向；外推时无条件支路为空前缀、仅当前块/窗。RELF 最右 \(S\) 个新噪声槽 sc 恒为 0。

## 6. BELF：块条件流

序列切成块 \(z_0^{(b)}\in\mathbb{R}^{W\times D}\)。联合

\[
p_G(z_0)=\prod_b p_G\bigl(z_0^{(b)}\mid z_0^{(<b)}\bigr).
\]

每一因子是条件 rectified flow：**未知槽共享同一标量 \(t\)**；已知余数钉 \(t=1\)。去噪块长 \(W\) 为 BELF 的 `block_size`，合法值 \(1\) 或加载入口块长。默认 \(W=\) 加载且主跑 \(W=T\)；若 \(i\sim\mathrm{Unif}\{0,\ldots,T-1\}\)，满未知块上每次前向施加 CE 的期望 token 数为 \(W/T\)。

训练：抽一跳，把 \(t=L_i\) 广播到未知槽。推理 `block_generate`：跳 \(i=0,\ldots,T-2\) Euler \(\Delta t=L_{i+1}-L_i\)（末流到 \(1-\varepsilon\)）；跳 \(T-1\) 在 \(t=1-\varepsilon\)、\(m=\mathrm{decode}\)，**不 Euler**，出口读 token。已知槽每跳覆写 encoder 干净码。SDE churn 关在最后一次流（跳 \(T-2\)）。

`cond_mode=clean`：前缀长 \(L\) 时 \(r=L\bmod W\)。完整块进 KV；当前块槽 \([0,r)\) 为已知余数（\(t=1\)，不进损失）。一块一次读出，不会在同一块内把刚 decode 的 token 立刻改成已知余数再继续流。

链式法则写的是 \(p_G(z^{(b)}\mid z^{(<b)})\)；训练左段是 encoder 干净码，推理左段默认是 \(\hat x_0\)。这是教师强制缺口，不是推导跳步。

## 7. RELF：局部时间场

须 \(S\cdot T=W\)。窗内位置 \(k\) 有局部时间，插值独立：

\[
z^k=t_k z_0^k+(1-t_k)\varepsilon^k.
\]

整窗模板（freeroll）

\[
F_k=L_{T-1-\lfloor k/S\rfloor},\qquad k=0,\ldots,W-1.
\]

档 \(r\) 占槽 \([rS,(r+1)S)\)，值为 \(L_{T-1-r}\)。最左档 \(t=1-\varepsilon\)（decode），右组纯噪 \(L_0\)。每档 \(S\) 槽共享同一 \(L_i\)，不是逐槽独立采样 \(t\)。

**半群（不依赖 \(G\)）。** 推理稳态：每步一次 \(G\)，最左档不 Euler；右侧未知档升一档（接到更高的 \(L_i\)）；pop 左端 \(S\) 槽，右端补 \(S\) 个 \(L_0\) 新噪声后，时间场回到 \(F\)。证明只依赖梯子算术与 \(S\cdot T=W\)。Euler 有局部截断误差；SDE churn 关在升档侧、decode 前向无速度。

**截断。** 虚拟满窗铺 \(F\)。窗起点 \(u\) 按文档 \(S\) 对齐。对格 \(k\)、文档下标 \(j=u+k\)：\(j<i_{\mathrm{bos}}\) 丢掉（句首切），\(j>i_{\mathrm{eos}}\) 丢掉（句尾切），否则 \(t_k=F_k\)。丢掉的位置不在窗：不赋 \(t=0\)、不加 \(m\)，也不把留下的梯子挪到另一头凑满 \(W\)。两切独立可叠加。PAD 不是边界。抽窗须保住 BOS / EOS。

**余数爬梯。** 前缀 \(L\) 不对齐档界时 \(r=L\bmod S>0\)。新词从 \(L_0\) 爬到 \(1-\varepsilon\) 才 CE/pop，禁止钉满窗 \(F\) 左侧一次 CE。hop \(h=0,\ldots,T-1\) 的虚拟余数起点 \(k_0=S(T-h-1)\)；\(h<T-1\) 时 decode 档已切掉（\(\sum m_c=0\)，只 MSE、不 pop）；\(h=T-1\) 时 \(k_0=0\)，未知 decode 格纯 CE 后 pop。训练在未做句首切时以 `clean_block_prob` 抽一帧 \((h,r)\)。

RELF 的联合由「pop 后条件于 KV 的窗上局部时间流」迭代定义，不是 ELF 整句联合流的条件化，也不是每槽独立边缘的乘积路径（档内强制共享 \(L_i\)）。学习目标是该规定 \((z,t)\) 场上的速度回归；只要推理每一步的 \((z,t)\) 落在训练由截断给出的场上，回归是良定的。

## 8. 自检要点

1. **时间轴。** 干净端 \(t=1\)。2L 左段标 \(t=1\)，不对噪声 \(t\) 做 v-MSE。
2. **半群。** RELF 滑窗后梯子复位只依赖 \(\Delta t\) 与 \(S\cdot T=W\)；状态 Euler 的截断误差是数值问题，用末流关 churn 避免读出前把左槽拉回噪声。
3. **分母。** \(t\) 截到 \(1-\varepsilon\) 且分母有 \(\varepsilon_t\)，v-MSE 有界。丢掉的格不进 \(m_d\)。
4. **教师强制。** 训练条件是 encoder 码，推理条件是 \(\hat x_0\)。要闭合需 self-forcing，本规格不强制。
5. **出口。** token 对齐 \(z\) 上，`linear` 出口可能退化成浅 unembed；这不影响流匹配合法性。默认等宽 decoder，CE 回 \(G\)。
6. **两族。** 不同 \(t\) 场是不同输入分布。共享结构合法；共享一份权重会让同一网络逼近两种条件场。规格是两个模型族、两套训练剖面。
