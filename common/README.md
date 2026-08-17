# common · 跨模块共用代码

存放边缘感知（模块一）与云边调度（模块二）的共享代码。

## 内容

- **`drift_dataset.py`** — 确定性环境漂移模拟器。基于 `DRIFT_SCHEDULES` 字典定义 5 档漂移模式（`none`/`light`/`mixed`/`staged`/`highfreq`），通过 `DeterministicDriftWrapper` 在 Dataset 数据读取时实时注入高亮、暗光、模糊、高斯噪声、随机遮挡等物理变化。读取时同步计算并返回样本对应的环境漂移强度（severity）与结构化漂移指标（struct_drift）。

> 注：`drift_dataset.py` 是**旧漂移框架**（5 档 schedule + 5 种退化），现仅用于 RL 轨迹生成（E_drift / struct_drift）。正式 cloud-teacher adapter 评估使用的是校准相机漂移（3 类：illumination / motion_blur / defocus / sensor_noise），见 `module_edge_perception/boxcars_camera_drift_dataset.py` / `modelnet_camera_drift_dataset.py`。

## 调用场景

| 模块 | 脚本 | 用途 |
|:---|:---|:---|
| 边缘感知 | `boxcars_camera_drift_dataset.py` / `modelnet_camera_drift_dataset.py` | 构建漂移数据集，训练 adapter |
| 云边调度 | `generate_real_trajectory.py` / `evaluate_rl_policy_on_mvvit.py` | 生成漂移轨迹特征（E_drift, struct_drift），输入给 RL 主循环 |

> ⚠️ 修改本目录下的数据结构或接口，须先在 `docs/接口契约.md` 中达成一致。
