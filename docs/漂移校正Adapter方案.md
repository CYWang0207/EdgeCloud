# 多场景相机退化校正 Adapter 方案

## 结论

项目采用**通用训练框架、场景专用退化配置与场景专用 Adapter 权重**的策略。不能把
ModelNet40 训练出的 adapter 直接下发给 BoxCars，反之亦然；两者可以共享 AdaptFormer
架构、成对 clean/corrupt 监督和 adapter-only 下发格式，但不共享退化参数或权重。

## 为什么不能直接共用

| 维度 | ModelNet40 | BoxCars116k |
|---|---|---|
| 图像来源 | 干净、规整的多视图物体渲染 | 同一交通摄像机中车辆轨迹的真实裁剪图 |
| 四视图关系 | 同一物体的渲染视角 | 同一车辆的不同时刻观测；短轨迹会补齐 view mask |
| 有效退化 | 渲染/相机噪声、失焦、曝光 | 低照度传感器链路、运动模糊、雾霾、监控曝光 |
| 模型敏感点 | 高频纹理与物体轮廓 | 车身品牌细粒度纹理、运动和远距离成像 |

同名的 `noise` 并不具有同一分布：ModelNet 用每视图独立的 Poisson-Gaussian
传感器噪声模拟渲染图被采集后的缺陷；BoxCars 则以同一摄像机的 ISO/gain 为轨迹级
条件、以每帧独立噪声为观测结果。BoxCars 还可能有雾霾、运动模糊和多视图冗余，
所以简单全局增亮/变暗、同一高斯白噪声或复制 ModelNet 的 severity 都不可靠。

## 通用闭环

```text
真实场景语义 -> 场景专用退化生成器 -> baseline 全集校准
    -> 选取约 8–15pp（可接受约 7pp）的有效强度
    -> clean/corrupt 成对训练冻结主干的 AdaptFormer
    -> clean 与每类固定退化的独立对比 -> adapter-only 下发
```

训练损失保持一致：

```text
L = L_cls(corrupt, label)
  + 0.25 * (1 - cosine(feature_corrupt, feature_clean))
  + 0.20 * KL(logits_corrupt || logits_clean)
```

每个训练混合保留 20% clean 样本，防止为恢复退化域而损害干净域。MV-ViT 主干冻结，
只训练 AdaptFormer（r=32，299,916 个 adapter 参数）；输出只含 adapter、norm、head，
可独立部署而不覆盖基座权重。

## 场景配置

### ModelNet40

- `illumination=1.0`：gamma/曝光、色温与局部阴影；
- `defocus=0.2`：失焦；
- `sensor_noise=0.4`：每视图独立 Poisson-Gaussian 噪声；
- 采样权重 `0.3 / 0.3 / 0.4`。

### BoxCars116k

- `illumination=1.0`：夜间或逆光的非均匀曝光、色温与阴影；
- `motion_blur=0.8`：与车辆运动/快门相关的方向性模糊；
- `sensor_noise=0.6`：低照度 gain 下的 Poisson-Gaussian 噪声，轨迹级参数一致；
- `haze=1.0`：雾霾大气散射导致的对比度和颜色衰减；
- 采样权重 `0.25 / 0.25 / 0.30 / 0.20`。

`compression`、`partial_occlusion`、`glare`、`lens_obstruction` 已在 BoxCars validation
集扫描，但未达到有效下降阈值，因此不进入正式训练，避免用无影响的干预凑类别。

## 权重隔离与路由

权重按“数据域 × baseline × 退化配置”隔离：

```text
modelnet40_drift_adapters/camera_mixture_calibrated/best.pth
boxcars_drift_adapters/general/best.pth                 # 保留的旧实验
boxcars_drift_adapters/camera_mixture_calibrated/best.pth # 新实验
```

目前每个场景先维护一个经校准的 `general`/mixture adapter。只有在同一场景中某类退化
在多个强度上显著优于 mixture adapter，才值得扩展专家 bank；跨场景权重共享须以单独
的交叉下发实验验证，不能默认成立。
