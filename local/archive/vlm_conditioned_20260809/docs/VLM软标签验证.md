# VLM 软标签导出方案与验证

## 方案定位

云端 VLM 不直接承担车辆品牌分类，而是为漂移样本生成 16 类软标签，作为
AdaptFormer 的蒸馏监督。adapter 训练完成后下发到边缘，最终分类仍由
`MV-ViT + AdaptFormer adapter` 完成。

```text
BoxCars 图像
  -> BoxCarsDriftWrapper 注入环境漂移
  -> 云端 VLM 生成 16 类 teacher logits
  -> train_adapter.py 蒸馏 adapter
  -> 边缘 MV-ViT + adapter 完成分类
```

本阶段只验证真实生成式 VLM 能否稳定生成与现有蒸馏接口兼容的 `[N, 16]`
teacher logits。最终效果应以 adapter 蒸馏后的边缘分类准确率和漂移鲁棒性为准。

## 软标签构造方法

### 为什么不能直接读取分类 logits

VLM 是生成式模型，输出层对应整个语言词表，没有天然的 BoxCars 16 类分类头，
因此不能像 ViT 一样直接取得 16 类 logits。需要把 16 个类别转换成生成式模型可
比较的候选答案，再从模型内部 logits 构造软标签。

### 可选方法一：逐品牌 forced decoding

固定提示词并显式列出 16 个品牌，然后依次把每个品牌名称作为候选答案，只计算
答案 token 的条件 log probability，最后对 16 个分数做 softmax。

该方法语义直接，但存在两个问题：

- 不同品牌在 tokenizer 中可能包含不同数量的 token，分数会受到答案长度影响；
- 每个样本需要分别计算 16 个候选序列，推理开销较大。

脚本保留该方法作为对照，并对多 token 品牌使用平均 token log probability。

### 默认方法：单 token 选项映射

将 16 个品牌在提示词中映射为 A--P：

```text
A. ford
B. skoda
...
P. kia
```

要求 VLM 只回答选项字母，然后从下一 token 的完整词表 logits 中抽取 A--P 的
16 个分数：

\[
q_c = \operatorname{softmax}(z_{A:P}/T)_c
\]

选择这种方式的依据很简单：固定候选集合通常可以按多选题的条件似然进行评分；
把所有类别映射成相同长度的单 token，可以避免品牌名称分词长度不同造成的偏差，
同时一次 forward 就能取得 16 个候选分数。映射完整写在提示词中，只改变输入格式，
不修改 VLM 参数，也不需要重新训练 VLM。

脚本启动时会检查 A--P 在当前 tokenizer 中是否均为单 token；不满足时直接报错。

## 代码与接口

导出脚本：

```text
module_edge_perception/export_boxcars_vlm_soft_labels.py
```

主要步骤：

1. 读取同一车辆的四个逻辑视图；
2. 可选通过 `BoxCarsDriftWrapper` 注入确定性环境漂移；
3. 构造包含 16 个品牌及 A--P 映射的固定提示词；
4. 提取 A--P 对应的内部 logits；
5. 保存原始 logits、概率、真实标签和数据集指纹。

输出字段：

```text
train_logits: [N_train, 16]
val_logits:   [N_val, 16]
```

该格式与 `train_adapter.py` 的输入约定一致。蒸馏训练读取原始 logits，并由训练
代码统一应用温度，不使用 VLM 自由生成的文本答案。

## 当前验证结果

使用服务器已有的 `Qwen3-VL-8B-Instruct-4bit-group`，在 BoxCars make validation
全部 649 个样本上验证默认方法：

| 诊断项 | 结果 |
|---|---:|
| 样本数 | 649 |
| teacher logits Top-1 | 62.25% |
| 平均最大概率（T=1） | 0.9288 |
| 平均熵（T=1） | 0.1883 |
| 归一化平均熵（T=1） | 0.0679 |
| 概率行和 | 1.0000 |

这里的 Top-1 只用于确认 teacher logits 与真实类别存在相关性，不是最终边缘分类器
的准确率。最终对标 baseline 88.04% 的对象是蒸馏后的 `MV-ViT + adapter`。

原始生成 logits 在 T=1 时偏尖。现有 `train_adapter.py` 默认使用 `T=4`，同一批
logits 经温度软化后的统计为：

| 温度 | 平均最大概率 | 平均熵 | 归一化平均熵 |
|---:|---:|---:|---:|
| 1 | 0.9288 | 0.1883 | 0.0679 |
| 2 | 0.8485 | 0.4445 | 0.1603 |
| 4 | 0.6486 | 1.2522 | 0.4516 |
| 8 | 0.3189 | 2.3481 | 0.8469 |

T=4 时分布既不是纯 one-hot，也不是均匀分布，可以作为现有蒸馏损失的输入。
当前结果证明四图输入、内部 logits 提取、16 类映射、温度软化和 `.npz` 接口均已
跑通，但不能单独证明 VLM 蒸馏会提升最终分类器。

## 后续评测

选定最终 VLM 后，应在相同漂移数据和训练配置下比较：

```text
MV-ViT baseline
MV-ViT + hard-label adapter
MV-ViT + VLM-KD adapter
```

以最后两项的分类准确率、漂移鲁棒性和 adapter 大小判断 VLM 蒸馏是否有效。

## 运行示例

```bash
cd ~/autodl-tmp/EdgeCloudRuntime/current/module_edge_perception

python export_boxcars_vlm_soft_labels.py \
  --dataset-path ../data/BoxCars116k_kaggle/BoxCars116k \
  --model-path ../models/Qwen3-VL-8B-Instruct-4bit-group \
  --split validation \
  --output qwen3_vlm_val_soft_labels.npz
```

对现有环境漂移数据导出时增加：

```bash
  --drift-schedule mixed
```

运行逐品牌 forced-decoding 对照时增加：

```bash
  --method brand_sequence
```
