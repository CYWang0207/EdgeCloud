# 模块一：边缘实时感知

## 职责

- MV-ViT 多视角推理（ViT-Small, 2200万参数, 4视角早期融合）
- AdaptFormer adapter：FFN 旁路 PEFT，主干冻结只训 adapter（已落地，8/3 验收通过）
- Token 剪枝：运行时动态保留率 k_t
- 漂移模拟与感知：5种环境漂移 + 香农熵 Edrift + 结构性漂移
- Prompt 注入：旧方案遗留，作"环境漂移快响应"可选辅助（存在两套实现，待清理）
- 性能测量：TTFT、推理延迟、GPU 内存

## 对标指标

| 指标 | 要求 | 方法 |
|------|------|------|
| 边侧参数量 | 千万级 | ViT-Small 2200万 |
| TTFT | 降低 >= 75% | Token 剪枝 |
| 推理内存 | <= 1.5GB | ViT-Small + 剪枝 |

## 目录约定（实际为扁平结构，非子目录）

```
module_edge_perception/
├── model.py                          # MV-ViT 模型（EarlyFusionMultiViewViT）
├── adaptformer.py                    # AdaptFormer PEFT 模块（已落地，8/3 验收通过）
├── train_adapter.py                  # 旧 VLM 类别软标签蒸馏实验（不用于漂移校正主线）
├── train_boxcars_drift_adapter.py    # 成对干净/漂移监督的 Adapter 专家训练
├── evaluate_boxcars_drift_adapters.py # baseline/统一/专家 Adapter 同条件评估
├── verify_adaptformer.py             # AdaptFormer 三点验收脚本（零初始化/参数量/三条前向）
├── boxcars_dataset.py                # 场景二 BoxCars116k 数据加载（4逻辑视图 + view_mask）
├── boxcars_drift_dataset.py          # BoxCars 漂移包装
├── train_boxcars.py                  # BoxCars baseline DDP 训练（test Top-1=88.04%）
├── evaluate_boxcars.py               # BoxCars 官方评估
├── train_boxcars_retrain_drift.py    # BoxCars 漂移全量重训
├── train_boxcars_token_prompt.py     # BoxCars Prompt 适配训练（过渡）
├── test_boxcars_inference.py         # BoxCars→MV-ViT 前向冒烟测试
├── dataset.py                        # 场景一 ModelNet40 数据加载
├── train.py / train_retrain_drift.py / train_token_prompt.py  # 场景一训练
├── test.py / evaluate_train_set.py   # 场景一评估
├── prompt_tuning/                    # PromptGenerator（可选辅助，存在两套实现待清理）
└── benchmarks/                       # TTFT / 内存 / 延迟 / 一体化测量
```

## 代码来源

核心代码由学长提供，位于本地 D:\challenge\Code\code2\，包括：
- model.py — MV-ViT 模型
- dataset.py — ModelNet40 数据加载
- boxcars_dataset.py — 第二场景 BoxCars116k 车辆轨迹多视图分类加载
- test_boxcars_inference.py — BoxCars116k 到现有 MV-ViT 的无训练前向链路验证
- train_boxcars.py — 已验证的 BoxCars116k 双卡 DDP 训练（仿照 `train.py`）
- evaluate_boxcars.py — 在 BoxCars116k 官方划分上评估 checkpoint
- drift_dataset.py — 漂移模拟器
- prompt_tuning/prompt_model.py — Prompt 生成器
- train.py / train_retrain_drift.py / train_token_prompt.py — 训练脚本
- test.py / evaluate_train_set.py — 评估脚本

## 负责人任务

- 测量 TTFT、推理延迟、GPU 内存
- Token 剪枝前后对比实验
- 换 ViT-Tiny（如需更小模型）只需改一行 model_name 参数

## BoxCars 漂移 Adapter 实验

漂移校正直接使用 BoxCars 的真实类别标签，不使用通用 VLM 的车辆品牌软标签。
训练集按样本独立采样漂移，避免旧的时间 schedule 将漂移类型与数据集顺序混淆；
损失由漂移分类、干净/漂移 CLS 特征对齐、干净基线输出一致性三部分组成。

先训练一个统一兜底 Adapter：

```bash
torchrun --standalone --nproc_per_node=2 train_boxcars_drift_adapter.py \
  --dataset-path /root/autodl-tmp/EdgeCloud/data/BoxCars116k_kaggle/BoxCars116k \
  --baseline-checkpoint checkpoints/boxcars_make_baseline/best.pth \
  --expert-name general \
  --drift-types bright,dark,blur,noise,occlusion
```

只有统一 Adapter 在某类漂移上恢复不足时，再训练该类专家，例如：

```bash
torchrun --standalone --nproc_per_node=2 train_boxcars_drift_adapter.py \
  --dataset-path /root/autodl-tmp/EdgeCloud/data/BoxCars116k_kaggle/BoxCars116k \
  --baseline-checkpoint checkpoints/boxcars_make_baseline/best.pth \
  --expert-name blur --drift-types blur
```

统一在相同样本、相同漂移和相同强度下比较，不用训练时的随机 batch 精度下结论：

```bash
python evaluate_boxcars_drift_adapters.py \
  --dataset-path /root/autodl-tmp/EdgeCloud/data/BoxCars116k_kaggle/BoxCars116k \
  --baseline-checkpoint checkpoints/boxcars_make_baseline/best.pth \
  --adapter general=checkpoints/boxcars_drift_adapters/general/best.pth \
  --adapter blur=checkpoints/boxcars_drift_adapters/blur/best.pth \
  --severity 0.8 --output-json checkpoints/drift_adapter_comparison.json
```

建议先跑 baseline 漂移矩阵，再训练 `general`；只有专家相对 `general` 有稳定收益时才保留
Adapter bank。当前人工模拟类型可直接作为路由 oracle，VLM 只作为未来真实场景中的可选漂移识别器。

### 连续 VLM 条件与视图可靠性（升级版）

新训练链路用连续环境向量对每层 AdaptFormer 做 FiLM 调制，并显式预测每个视图的可靠性。
不提供 VLM 缓存时可先训练纯边缘条件闭环：

```bash
python train_boxcars_drift_adapter.py \
  --dataset-path /path/to/BoxCars116k \
  --baseline-checkpoint checkpoints/boxcars_make_baseline/best.pth \
  --expert-name conditioned_general \
  --independent-view-drifts --condition-dim 128
```

VLM hidden states 采用离线缓存。原始文件按 split 保存 `[N, ..., D]` tensor，先统一池化、PCA
和归一化，再传给训练脚本：

```bash
python export_boxcars_vlm_hidden_states.py \
  --dataset-path /path/to/BoxCars116k \
  --model-path ../models/Qwen3-VL-8B-Instruct-4bit-group \
  --output checkpoints/qwen3vl_train_hidden_states.pt \
  --split train --independent-view-drifts

python prepare_vlm_condition_cache.py \
  --input checkpoints/qwen3vl_train_hidden_states.pt \
  --output checkpoints/vlm_conditions_128.pt \
  --condition-dim 128

python train_boxcars_drift_adapter.py \
  --dataset-path /path/to/BoxCars116k \
  --baseline-checkpoint checkpoints/boxcars_make_baseline/best.pth \
  --expert-name vlm_conditioned_general \
  --independent-view-drifts --condition-dim 128 \
  --vlm-condition-cache checkpoints/vlm_conditions_128.pt
```

默认只训练 Adapter、FiLM、边缘环境编码器和坏视图 token；`norm` 与 `head` 保持冻结，以便
隔离真正的漂移校正收益。需要做解冻消融时再加 `--train-norm` 或 `--train-head`。
