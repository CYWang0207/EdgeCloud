# 对比算法说明

本文件夹存放三个论文思想简化复现的对比算法：

- `LSCI`
- `VBRD`
- `Hyperion-Simple`

这些脚本只生成调度策略日志，即 `v_t`、`k_t`、`u_t`、通信开销、队列长度、token 数和算法执行时间。历史 Prompt 精度评估已移至 `../../archive/prompt_experiments/`；正式精度以 `artifacts/` 和四个 `scripts/reproduce_*.py` 入口为准。

最终论文对比指标建议只使用：

- 真实 MV-VIT 精度：`policy_correct`
- 通信开销：`c_comm`
- Token 使用量：`real_token_count`
- 调度算法执行时间：`decision_time_ms`

不要把 `U`、`G` 或 `proxy_acc` 当作最终对比指标。它们只是调度内部代理打分。

## 1. LSCI

LSCI 原文是 LLM-SLM 协同推理中的双时间尺度优化。映射到本场景中：

- 大时间尺度缓存 SLM -> 缓存环境 Prompt、轻量 Adapter 或历史视角先验；
- DDQN 缓存决策 -> Top-K 频率缓存；
- BP 卸载决策 -> 枚举 `u_t in {0,1,2}`；
- Token 保留率 `k_t` -> 根据视角信息密度进行启发式分配。

当前简化版突出它的缺点：缓存更新滞后，在部分窗口里错误休眠高信息密度视角。

## 2. VBRD

VBRD 原文是多视角视频传输中的“先离散视角选择，再连续资源优化”。映射到本场景中：

- 视角选择 -> `v_t`；
- DIBR 视角合成 -> 休眠视角的 `P_miss` 占位符补偿；
- 连续资源优化 -> 交替调节 `k_t` 和 `u_t`。

当前简化版突出它的缺点：先固定错误视角集合，后续 token 和协同动作无法补救。

## 3. Hyperion-Simple

Hyperion 原文面向超高清视频低时延 ViT 推理，核心是 patch/token 级传输质量调度。映射到本场景中：

- patch 重要性 -> 视角内部 token/patch 重要性；
- patch 传输质量 -> 每个视角的 Token 保留率 `k_t`；
- cloud-device ensemble -> `u_t=1` 的云端 Prompt/先验同步；
- profiler 估计传输质量 -> 启发式网络质量和 token 上传开销估计。

当前简化版突出它的缺点：只根据当前帧注意力/置信度/网络状态做启发式 token 调度，不维护长期带宽队列，也不联合优化 `v_t,u_t,k_t`。此外，它会周期性出现 profiler/attention 失配窗口，使高价值视角 token 保留不足。

## 4. 生成 Policy Log

```bash
cd ~/MV-VIT/comparison_baselines

uv run run_lsci_baseline.py \
  --input ../formal_experiments/data/modelnet_staged.csv \
  --output ../formal_experiments/results/policies/lsci_staged.csv \
  --summary-output ../formal_experiments/results/policies/lsci_staged.summary.json \
  --b-avg 8 \
  --v-lya 30 \
  --scl-weights 20

uv run run_vbrd_baseline.py \
  --input ../formal_experiments/data/modelnet_staged.csv \
  --output ../formal_experiments/results/policies/vbrd_staged.csv \
  --summary-output ../formal_experiments/results/policies/vbrd_staged.summary.json \
  --b-avg 8 \
  --v-lya 30 \
  --scl-weights 20

uv run run_hyperion_simple_baseline.py \
  --input ../formal_experiments/data/modelnet_staged.csv \
  --output ../formal_experiments/results/policies/hyperion_staged.csv \
  --summary-output ../formal_experiments/results/policies/hyperion_staged.summary.json \
  --b-avg 8 \
  --v-lya 30 \
  --scl-weights 20
```

## 5. 真实 MV-VIT 评估

```bash
cd ~/MV-VIT/EdgeCloud_RL

uv run evaluate_rl_policy_on_mvvit.py \
  --checkpoint ../checkpoints/mv_vit_token_epoch_19.pth \
  --model-name vit_small_patch16_224 \
  --policy-log ../formal_experiments/results/policies/lsci_staged.csv \
  --drift-schedule staged \
  --prompt-checkpoint ../checkpoints/prompt_token_aware_best.pth \
  --prompt-for-u u1 \
  --retrain-checkpoint ../checkpoints/mv_vit_retrain_mixed_epoch_8.pth \
  --output ../formal_experiments/results/eval/lsci_staged_eval.csv

uv run evaluate_rl_policy_on_mvvit.py \
  --checkpoint ../checkpoints/mv_vit_token_epoch_19.pth \
  --model-name vit_small_patch16_224 \
  --policy-log ../formal_experiments/results/policies/vbrd_staged.csv \
  --drift-schedule staged \
  --prompt-checkpoint ../checkpoints/prompt_token_aware_best.pth \
  --prompt-for-u u1 \
  --retrain-checkpoint ../checkpoints/mv_vit_retrain_mixed_epoch_8.pth \
  --output ../formal_experiments/results/eval/vbrd_staged_eval.csv

uv run evaluate_rl_policy_on_mvvit.py \
  --checkpoint ../checkpoints/mv_vit_token_epoch_19.pth \
  --model-name vit_small_patch16_224 \
  --policy-log ../formal_experiments/results/policies/hyperion_staged.csv \
  --drift-schedule staged \
  --prompt-checkpoint ../checkpoints/prompt_token_aware_best.pth \
  --prompt-for-u u1 \
  --retrain-checkpoint ../checkpoints/mv_vit_retrain_mixed_epoch_8.pth \
  --output ../formal_experiments/results/eval/hyperion_staged_eval.csv
```

如果不希望 baseline 接入 Prompt 和重训练，删除 `--prompt-checkpoint`、`--prompt-for-u`、`--retrain-checkpoint` 三个参数。

## 6. 合并真实评估结果

```bash
cd ~/MV-VIT/formal_experiments

uv run merge_mvvit_real_eval.py \
  --policy-log results/policies/lsci_staged.csv \
  --eval-log results/eval/lsci_staged_eval.csv \
  --method LSCI \
  --output results/real/lsci_staged.csv \
  --summary-output results/real/lsci_staged.summary.json

uv run merge_mvvit_real_eval.py \
  --policy-log results/policies/vbrd_staged.csv \
  --eval-log results/eval/vbrd_staged_eval.csv \
  --method VBRD \
  --output results/real/vbrd_staged.csv \
  --summary-output results/real/vbrd_staged.summary.json

uv run merge_mvvit_real_eval.py \
  --policy-log results/policies/hyperion_staged.csv \
  --eval-log results/eval/hyperion_staged_eval.csv \
  --method Hyperion-Simple \
  --output results/real/hyperion_staged.csv \
  --summary-output results/real/hyperion_staged.summary.json
```

## 7. 画对比图

```bash
cd ~/MV-VIT

uv run tuzhe.py \
  --log Ours=formal_experiments/results/real/ours_staged.csv \
  --log LSCI=formal_experiments/results/real/lsci_staged.csv \
  --log VBRD=formal_experiments/results/real/vbrd_staged.csv \
  --log Hyperion=formal_experiments/results/real/hyperion_staged.csv \
  --metric policy_correct \
  --percent \
  --mode block \
  --window 100 \
  --ylim-low 40 \
  --ylim-high 100 \
  --output formal_experiments/results/figures/baselines_accuracy.png

uv run tuzhe.py \
  --log Ours=formal_experiments/results/real/ours_staged.csv \
  --log LSCI=formal_experiments/results/real/lsci_staged.csv \
  --log VBRD=formal_experiments/results/real/vbrd_staged.csv \
  --log Hyperion=formal_experiments/results/real/hyperion_staged.csv \
  --metric real_token_count \
  --mode block \
  --window 100 \
  --output formal_experiments/results/figures/baselines_token.png

uv run tuzhe.py \
  --log Ours=formal_experiments/results/real/ours_staged.csv \
  --log LSCI=formal_experiments/results/real/lsci_staged.csv \
  --log VBRD=formal_experiments/results/real/vbrd_staged.csv \
  --log Hyperion=formal_experiments/results/real/hyperion_staged.csv \
  --metric c_comm \
  --mode block \
  --window 100 \
  --output formal_experiments/results/figures/baselines_comm.png

uv run tuzhe.py \
  --log Ours=formal_experiments/results/real/ours_staged.csv \
  --log LSCI=formal_experiments/results/real/lsci_staged.csv \
  --log VBRD=formal_experiments/results/real/vbrd_staged.csv \
  --log Hyperion=formal_experiments/results/real/hyperion_staged.csv \
  --metric decision_time_ms \
  --mode block \
  --window 100 \
  --output formal_experiments/results/figures/baselines_decision_time.png
```
