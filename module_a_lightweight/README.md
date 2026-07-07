# 模块 A · 模型轻量化 / 压缩（东方老师组）

> 技术底座：AutoTailor（SuperNet + TailorIR 图编译 + learning-free 预测器）

## 职责

- 将国产开源大模型（如 DeepSeek）/ MV-ViT 压缩为一族**权重共享、可运行时切换**的 SuperNet 变体
- 为每个 SubNet 变体打上 `(TTFT, 内存, 精度代理)` 标签，产出**预测器查找表（LUT）**
- 边侧信创部署（NCNN / FP16）

## 对标指标

| 指标 | 要求 |
|---|---|
| 边侧能力保持（数学/代码/NL 推理） | 满血的 80%–90% |
| TTFT（首 token 时延） | 减少 75% |
| 单次推理内存占用 | ≤ 1.5 GB |

## 对外接口（与模块 B 的契约，详见 `docs/接口契约_v0.md`）

- 输出：`{SubNet_id → (TTFT, mem, acc_surrogate)}` 查找表 + 可执行权重
- 调度器（模块 B）按 `{v_t, k_t}` 索引对应 SubNet 并切换

## 目录约定（待补充）

```
module_a_lightweight/
├── supernet/      # SuperNet 构建与训练
├── compiler/      # TailorIR 图编译
├── predictor/     # learning-free 预测器（时延 LUT / 精度敏感度 / 内存）
└── deploy/        # NCNN / 信创部署脚本
```
