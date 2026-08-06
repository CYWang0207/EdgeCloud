# 多视图漂移校正 Adapter：后续工作说明

## 背景

第二场景是 BoxCars116k 的多视图车辆品牌识别。边缘侧的 MV-ViT 负责真正的四视图实时分类；环境漂移会造成亮度、模糊、噪声、遮挡和视图失效等输入分布变化，使原模型性能下降。

云端生成式 VLM 在低分辨率监控车辆图上的直接品牌判断不可靠，因此不应把它输出的 16 类品牌概率作为 MV-ViT 的分类监督或软标签。漂移校正与“将全量大模型压缩为边缘模型”也是两件不同的事：本阶段的 Adapter 是小型视觉校正模块，不将其表述为 VLM 本体压缩。

## 要解决的问题

目标不是替换 MV-ViT，也不是让 VLM 负责车辆品牌分类，而是在发生已知或可识别的环境退化时，以很小的更新代价恢复边缘多视图分类器的表现。

关键映射为：

```text
漂移后的多视图特征  ->  接近干净域下的多视图特征  ->  正确品牌分类
```

因此，Adapter 的职责应是漂移校正和多视图可靠性调整，而不是泛化地学习 VLM 的类别答案。

## 升级后的系统逻辑

```text
训练阶段：
  四视图 / 时间窗口 -> 冻结开源 VLM 的多层视觉 hidden states
                    -> 连续环境教师向量 z_vlm
                                      |
                                      | 表征蒸馏
                                      v
  漂移四视图 -> MV-ViT patch 特征 -> 边缘环境编码器 -> z_edge
                                      |
                                      +-> 每视图可靠性 q_1...q_4
                                      +-> FiLM 动态调制各层 AdaptFormer
                                      v
                                16 类品牌分类

部署阶段：
  普通时刻由 z_edge 在边缘连续控制 Adapter；云端 VLM 只对关键帧或稳定时间窗口
  异步产生 z_vlm，用于周期校准。云端不可用时退回 z_edge 或无条件 general Adapter。
```

VLM 不输出品牌类别，也不以自由文本或固定漂移类型作为主要控制接口。主控制信号是从视觉
编码器中间层和最后层提取、投影到 128--256 维的连续环境向量。它可以同时表达光照、模糊、
遮挡、摄像头状态和未知复合变化，避免将真实环境硬切成互斥类别。

为可解释性、日志和答辩展示，VLM 仍可附带输出受约束的退化描述，例如：

```json
{
  "low_light": 0.8,
  "over_exposure": 0.0,
  "blur": 0.3,
  "noise": 0.1,
  "occlusion": 0.0,
  "view_failure": 0.0,
  "unknown": 0.0
}
```

该 JSON 是辅助解释，不作为 Adapter 的唯一输入。最终品牌始终由 MV-ViT 输出。

## 条件化 AdaptFormer 与视图可靠性

现有 AdaptFormer 继续保留在每个 Transformer block 的 FFN 旁路。将固定旁路

```text
y = FFN(x) + scale * Adapter(x)
```

升级为连续条件调制：

```text
h = GELU(W_down x)
(gamma_l, beta_l) = Controller_l(z)
h_cond = (1 + gamma_l) * h + beta_l
y = FFN(x) + scale_l(z) * W_up(h_cond)
```

第一版采用每层 bottleneck FiLM，控制器零初始化，因此未提供环境向量或刚挂载模块时与原
AdaptFormer 完全兼容。后续只有在消融证明有必要时，才尝试多个无人工命名的低秩 Adapter
基进行软混合，不把专家强行命名为 `blur`、`dark` 等类别。

边缘环境编码器同时预测每个视图的可靠性 `q_i in [0,1]`。低质量视图的 patch token 向
`bad_view_token` 收缩后再进入跨视图 Transformer，从机制上显式验证“降低坏视图干扰”，而
不只依赖最终 CLS 隐式学习。

## Adapter bank 的定位调整

原方案可为常见退化预先训练少量专家 Adapter：

```text
A_clean       正常环境
A_low_light   暗光校正
A_blur        模糊校正
A_noise       噪声校正
A_occlusion   遮挡鲁棒校正
A_general     未知或混合退化的保守兜底
```

升级后它不再是主线。优先训练一个由连续环境向量控制的 general Adapter；只有在多强度、
复合漂移和 test split 上证明某个专家稳定优于条件化 general Adapter 时才保留该专家，避免
维护人工漂移 taxonomy 和脆弱的硬路由。

Adapter 优先放在多视图特征融合附近，学习两类小幅修正：

1. 校正受漂移影响的视图特征；
2. 降低低质量视图在融合中的权重。

## 更新策略

已知且反复出现的退化不应每次重新训练：边缘持续使用本地条件化 general Adapter，云端只需
周期性更新连续环境向量或新的小参数包，不对每帧执行硬切换。

只有出现持续的未知/混合退化、且后续获得了可用标注数据时，才在云端训练一个新的或更新后的 Adapter，并将小参数包下发。单帧局部遮挡不适合每帧请求 VLM，应由边缘的视图质量和融合机制自行处理；VLM 面向稳定的时间窗口或摄像头级环境变化。

## 当前实验中的边界

现有漂移模拟器已经提供 bright、dark、blur、noise、occlusion 等真实退化类型。这些元数据可作为训练和验证阶段的 oracle，用于先验证不同退化 Adapter 是否有效；此阶段无需假装由 VLM 识别漂移。

在后续真实环境中，不再存在该 oracle 时，才评估云端 VLM 输出的退化描述能否替代它。若 VLM 的退化判断不可靠，则不应参与 Adapter 选择。

升级方案先用合成器提供的连续强度和每视图质量作为 oracle，验证条件化校正的收益上限；再用
缓存的 `z_vlm` 替代人工类型。不能从“能识别漂移标签”直接推导“能改善最终分类”，两者必须
分别评估。

## 升级训练目标

```text
L = L_cls(drift, label)
  + lambda_global * L_feature_global(drift CLS, clean CLS)
  + lambda_consistency * L_KL(drift logits, clean logits)
  + lambda_vlm * L_distill(z_edge, z_vlm)
  + lambda_quality * L_quality(q_view, synthetic quality)
```

后续取得真实时间窗口后再增加时间一致性损失。训练时交替使用 VLM 教师向量、边缘学生向量和
空条件，避免部署时依赖每帧云端请求。建议初始比例为 50% / 30% / 20%；在没有 VLM 缓存的
第一阶段，直接使用 `z_edge` 完成条件化闭环。

## 数据与评估要求

下一阶段不只对整条样本施加同一种漂移。训练和评估至少覆盖：

1. 多个 severity 档位和三个随机种子；
2. 每个视图独立采样退化类型和强度；
3. 两种退化叠加、局部视图失效和干净/坏视图混合；
4. validation 与 test split；
5. 无 Adapter、原 general Adapter、oracle 条件、VLM 条件和边缘学生条件的消融；
6. 只训 Adapter、Adapter + norm、Adapter + norm + head 的参数解冻消融。

当前 severity=0.8 时 baseline 下降很小，因此已有 0.31--1.08 pp 增益只能证明闭环可行，
不能证明复杂路由必要。新方案应首先扩大和真实化退化分布，再评价条件控制器。

## 3090 实施策略

冻结 2B--8B 开源 VLM，使用 4-bit 权重仅做离线前向；不在 Adapter 训练 batch 中在线运行
VLM。对 13,098 条训练样本缓存每视图或每窗口的 128--1024 维 fp16 表征，通常只占几十到
数百 MB。后续只训练环境编码器、FiLM 控制器、质量头和 AdaptFormer，可训练参数预计在数百万
以内，单张 24 GB RTX 3090 足够。只有原始 VLM 表征确实不包含所需环境信息时，才考虑对视觉
塔最后一两层或投影层做 LoRA。

## 当前实现、验收口径与结论

本阶段的最小闭环是：**在已知合成漂移下，训练一个小型 Adapter，使冻结的 BoxCars
MV-ViT 在漂移域的品牌分类结果更接近其干净域表现。** 它不是完整的“VLM 路由 + 专家
Adapter bank”系统，验收时必须将这两层区分开。

当前已实现的训练闭环如下：

```text
同一 BoxCars 样本
  -> 干净四视图：冻结 MV-ViT、关闭 Adapter，产生 CLS 特征和分类输出作为参考
  -> 漂移四视图：开启 Adapter，使用真实品牌标签训练

训练目标 = 漂移图分类损失
         + 漂移 CLS 与干净 CLS 的特征对齐
         + 漂移输出与干净输出的一致性
```

漂移类型对每个样本独立随机采样，而不是沿用按时间/数据顺序切换漂移的 schedule；这样可避免
模型把漂移种类与样本顺序错误关联。MV-ViT 主干冻结，只训练 AdaptFormer 参数，因此更新包仍约
1.2 MB，符合边缘侧小参数下发的约束。

在该已验证闭环之上，代码已加入下一阶段所需的 FiLM 条件控制器、边缘环境编码器、显式视图
可靠性门控、独立视图漂移和 VLM 条件缓存接口。新增模块已在 RTX 3090 上通过零初始化、前向、
反向和单 batch 训练保存测试；Qwen3-VL-8B 4-bit 的第 8/16/24/最终视觉层缓存也已用真实
BoxCars 样本跑通。尚未完成的是全量特征导出和正式多 epoch 对比，因此下面的准确率仍全部属于
原 `general` Adapter，不能归因于升级模块。

当前第一轮实验只训练了一个覆盖 `bright,dark,blur,noise,occlusion` 的 `general` Adapter，
并在同一批验证样本、同一漂移随机种子和同一强度下与无 Adapter baseline 比较。severity=0.8、
649 个 validation 样本上的结果为：

| 输入域 | Baseline Top-1 | General Top-1 | 变化 |
|---|---:|---:|---:|
| clean | 93.84% | 94.30% | +0.46 pp |
| bright | 91.99% | 92.76% | +0.77 pp |
| dark | 93.84% | 94.14% | +0.31 pp |
| blur | 92.30% | 93.37% | +1.08 pp |
| noise | 92.30% | 93.37% | +1.08 pp |
| occlusion | 91.83% | 92.14% | +0.31 pp |

因此可以得出的结论是：**统一 Adapter 的漂移校正主线已获得初步验证**——五个测得漂移域均有
正增益，且干净域没有回退。此处的增益为约 0.31--1.08 个百分点，说明其有效但仍属温和恢复；
不应将其表述为已经解决所有真实环境漂移。

以下内容尚未完成，不能由这轮结果外推：

1. `blur`、`noise` 等专家 Adapter 是否能稳定优于 `general`；
2. 坏视图是否确实被降低融合权重（当前只评估了总体 Top-1，未按视图质量消融）；
3. 多个漂移强度、test split 和真实摄像头数据上的泛化；
4. 云端 VLM 的受约束退化识别能否可靠替代训练/验证中的 oracle，并据此路由 Adapter。

完整命令、训练日志和结果文件位置见
[漂移校正Adapter实验记录（2026-08-05）](漂移校正Adapter实验记录_20260805.md)。

## 后续实现顺序

1. 在多个 severity、独立视图漂移和 test split 上复现 baseline 与 `general`；
2. 实现 `z_edge`、FiLM 条件化 AdaptFormer 和视图可靠性门控，先用合成质量 oracle 验证；
3. 离线缓存冻结 VLM 的中间层/最后层视觉表征，比较单层、多层和人工条件；
4. 蒸馏 `z_vlm -> z_edge`，评估 VLM 在线、边缘学生和异步校准三种部署模式；
5. 仅在条件化 general Adapter 恢复仍不足时训练专家或低秩 Adapter 基。
