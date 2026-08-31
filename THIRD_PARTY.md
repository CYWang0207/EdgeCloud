# 第三方材料与许可证清单

本清单记录截至 2026-08-31 已核验的官方来源和使用边界。项目许可证不覆盖第三方模型、数据、Logo、音频或商标；完整数据集和 InternViT 权重均不在本仓库或提交包中重复分发。

| 名称 | 项目用途 | 来源 | 许可证/使用边界 | 提交方式 |
|---|---|---|---|---|
| InternViT-6B-224px | 云端冻结视觉教师 | [OpenGVLab 模型卡](https://huggingface.co/OpenGVLab/InternViT-6B-224px)，核验 revision `084bfb607eb04f373bf0349c49eb6bc3ba094919` | 模型卡标注 MIT；OpenGVLab/InternVL 代码仓同为 MIT。训练数据仍遵循各自条款 | 不重复分发 6B 权重，提供官方链接、revision 和本项目所用文件校验值 |
| PyTorch / torchvision | 训练与推理框架 | [PyTorch 官方仓库](https://github.com/pytorch/pytorch) | 官方 BSD-style 许可及仓库内第三方声明 | 通过依赖文件安装，不打包框架二进制 |
| timm | ViT-Small 模型实现与预训练权重接口 | [pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | Apache-2.0；预训练权重还须遵循对应模型卡 | 通过依赖文件安装 |
| ModelNet40 | 四视图三维物体识别数据 | [Princeton ModelNet 官方下载页](https://modelnet.cs.princeton.edu/download.html) | 官方页面面向研究者提供下载并要求引用，但未展示标准 SPDX 许可证或明确再分发授权。按研究评测用途保守使用 | 不重复分发完整数据集；提供官方获取和准备脚本。仓库仅保留少量派生渲染样例，若赛事要求授权凭证则应向数据集作者确认或从提交包移除 |
| BoxCars116k | 交通监控车辆多视图识别数据 | [作者官方仓库](https://github.com/JakubSochor/BoxCars) | 官方 README 明确代码为 `research only`，仓库未提供标准 LICENSE 文件；数据集按作者研究用途与引用要求使用 | 不重复分发完整数据集；提供官方获取和准备脚本。仓库中的少量样例仅用于赛事复现，禁止据此主张商业或再分发权 |
| AdaptFormer | Adapter 结构参考 | [ShoufaChen/AdaptFormer](https://github.com/ShoufaChen/AdaptFormer) | MIT | 在作品报告中引用论文，并保留来源与许可证说明 |
| 演示网页图片、Logo 与音频 | 项目展示 | 团队制作、学校/合作单位官方素材及经授权音频 | 提交前逐项核对商标与音频授权 | 只用于赛事展示或按授权范围发布 |

## 仍需由团队留存的材料

1. 演示视频 BGM、音效和图片素材的来源、生成方式及授权凭证。
2. 东南大学与浪潮 Logo 用于赛事展示的授权或官方素材来源记录。
3. 若最终压缩包保留 ModelNet40/BoxCars116k 派生样例，应留存赛事研究展示用途说明；若无法确认再分发边界，则从压缩包移除样例并保留下载脚本。
4. 复制或修改的第三方代码文件、版权头和对应许可证全文。

项目自有代码许可证不能覆盖第三方模型、数据、Logo、音频或商标。
