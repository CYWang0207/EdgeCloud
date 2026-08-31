# EdgeCloud 提交材料阅读入口

本文件是赛事组委会和评审专家的首要阅读入口。建议按以下顺序查看材料。

## 1. 先了解作品

1. 阅读[作品报告](submission/作品报告.pdf)，了解应用背景、总体架构、核心方法和实验结论。
2. 观看[作品运行效果视频](submission/作品运行效果视频.mp4)，了解多视角感知、漂移发现、云端蒸馏、Adapter 下发与恢复流程。
3. 访问[在线交互演示](https://cywang0207.github.io/EdgeCloud-Demo/)，体验 BoxCars116k 与 ModelNet40 双场景回放。
4. 阅读[项目 README](README.md)，快速查看技术路线、核心指标和运行命令。

## 2. 核验实验结果

- 两场景恢复能力：[实验结果总览](docs/实验结果总览_20260809.md)
- ModelNet40：[第一场景实验报告](docs/第一场景_ModelNet40.md)
- BoxCars116k：[第二场景实验报告](docs/第二场景_BoxCars116k.md)
- 弱网业务保持：[网络波动模拟器设计](docs/网络波动模拟器设计.md)
- 多节点一致性：[冲突仲裁测试结果](docs/多节点冲突仲裁测试结果_20260811.md)
- 原始指标与精简证据：`artifacts/`

所有指标均应结合其口径阅读。准确率来自公开数据集完整测试集；相机漂移为合成且经过校准；网络时延和多节点部分包含仿真结果。网络模拟中的 `T_edge=80 ms` 为固定参数。

## 3. 运行作品

- 无数据快速验证：按照 README 的“5 分钟快速体验”执行。
- 完整实验复现：按照 [REPRODUCE.md](REPRODUCE.md) 准备数据、权重并运行对应命令。
- 网页演示：直接打开 `demo-web/index.html`，或启动本地静态服务器。

## 4. 提交包校验

- 文件清单：[SUBMISSION_MANIFEST.md](SUBMISSION_MANIFEST.md)
- 文件哈希：[MANIFEST.sha256](MANIFEST.sha256)
- 代码许可：[LICENSE](LICENSE)
- 第三方资源：[THIRD_PARTY.md](THIRD_PARTY.md)

正式提交版本以 Git 标签 `challengecup-2026-final` 和对应 GitHub Release 为准。
