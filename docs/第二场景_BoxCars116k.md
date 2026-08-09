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
- 云端视觉教师仅在 `u=1` refresh 时为漂移样本产生监督；边缘运行无条件 AdaptFormer；
- `view_mask` 可表达轨迹补齐或视图故障。

## 当前状态

- baseline：已完成 30 轮双卡训练；
- 官方 test 集：Top-1 `88.04%`，Top-5 `97.13%`；
- 漂移全量重训练链路：已通过真实数据单批次训练验证；
- 旧 Prompt/VLM-conditioned 适配链路：已归档为消融，不作为正式方案；
- InternViT-6B 云端 teacher 的特征缓存与 Adapter refresh 链路：首轮代表性样本验证已完成；无上传标签 cloud Adapter 三漂移平均 85.21%，较未适配 Edge 81.97% 提升 3.24 pp，距有标签 Adapter 上界 0.87 pp。

上面的 `88.04%` 是既有 **clean 官方 test** baseline 指标；`81.97%` 是本轮固定
illumination/blur/noise validation protocol 下的 **三漂移平均**，两者不是同一指标，不能直接相减或混写。
当前 cloud-teacher 数值也属于开发集证据，因为 task head 和 Adapter 都据此选择。独立 test 的快速检查固定
checkpoint 后进行，不会据 test 重新选择 head、epoch 或 loss；其范围与限制见
[`实验状态与证据边界_20260809.md`](实验状态与证据边界_20260809.md)。

## 主要代码

| 文件 | 用途 |
|---|---|
| `boxcars_dataset.py` | 官方划分、四逻辑视图和 `view_mask` 加载 |
| `train_boxcars.py` | BoxCars baseline 训练 |
| `evaluate_boxcars.py` | 官方 validation/test 评估 |
| `boxcars_camera_drift_dataset.py` | BoxCars 专用相机漂移数据包装与固定强度协议 |
| `export_boxcars_cloud_teacher_cache.py` | 云端冻结视觉教师导出 task logits / features cache |
| `train_boxcars_cloud_teacher_adapter.py` | 使用 cloud supervision 刷新无条件 AdaptFormer |
| `evaluate_boxcars_cloud_teacher_quick_test.py` | 固定 checkpoint 的快速独立 test 检查，保存逐样本预测 |

以上代码均位于 `module_edge_perception/`。数据集和权重只保存在服务器，不提交 Git。
