# 第二场景：BoxCars116k 交通车辆识别

## 场景定义

第二场景使用 BoxCars116k 完成交通监控车辆品牌识别。一个样本由同一辆车轨迹中
时间跨度尽可能大的 4 张裁剪图组成，输入形状为 `[4, 3, 224, 224]`，默认采用官方
`make` 划分进行 16 类品牌分类。

需要注意：BoxCars116k 不提供跨摄像头车辆身份关联，因此这里的 4 张图是同一物理
摄像头下的 4 个逻辑视图，不能表述为四台摄像头同时拍摄同一车辆。少于 4 张图的
轨迹会重复末张图补齐，并通过 `view_mask` 标记无效视图。

## 与项目架构的对应关系

- MV-ViT 融合同一车辆不同时间、尺度和可见面的观测；
- Token 剪枝用于动态降低多视图推理开销；
- 漂移模块模拟明暗、模糊、噪声和遮挡等环境变化；
- Prompt 模块根据漂移类型生成条件 Prompt，完成轻量适配；
- `view_mask` 可表达轨迹补齐或视图故障。

## 当前状态

- baseline：已完成 30 轮双卡训练；
- 官方 test 集：Top-1 `88.04%`，Top-5 `97.13%`；
- 漂移全量重训练链路：已通过真实数据单批次训练验证；
- Prompt 适配链路：已通过真实数据单批次训练验证；
- 暂不完整训练后两条链路，待实验对比口径确定后统一运行。

## 主要代码

| 文件 | 用途 |
|---|---|
| `boxcars_dataset.py` | 官方划分、四逻辑视图和 `view_mask` 加载 |
| `train_boxcars.py` | BoxCars baseline 训练 |
| `evaluate_boxcars.py` | 官方 validation/test 评估 |
| `boxcars_drift_dataset.py` | BoxCars 专用漂移数据包装 |
| `train_boxcars_retrain_drift.py` | 漂移数据上的全模型重训练 |
| `train_boxcars_token_prompt.py` | 冻结主体后的 Prompt 适配训练 |

以上代码均位于 `module_edge_perception/`。数据集和权重只保存在服务器，不提交 Git。
