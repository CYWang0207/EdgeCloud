# EdgeCloud 赛事提交材料清单

本清单描述最终交付结构。正式提交时，GitHub 标签、Release 附件与压缩包内文件应保持一致。

| 材料 | 路径或位置 | 用途 |
|---|---|---|
| 首要阅读入口 | `README_FIRST.md` | 评委阅读与运行导航 |
| 项目说明 | `README.md` | 技术路线、指标与快速体验 |
| 作品报告 | `submission/作品报告.pdf` | 正式作品报告 |
| 运行效果视频 | `submission/作品运行效果视频.mp4` | 带 BGM 和分镜音效的演示视频 |
| 在线演示 | <https://cywang0207.github.io/EdgeCloud-Demo/> | 交互式端边云回放 |
| 源代码 | 仓库根目录 | 感知、调度、网络和仲裁实现 |
| 复现导航 | `REPRODUCE.md` | 输入、命令、输出和口径说明 |
| 实验报告 | `docs/` | 两场景、弱网和多节点实验说明 |
| 精简证据 | `artifacts/` | CSV、JSON、日志、绘图数据和配置 |
| 数据说明 | `data/README.md` | 数据来源、许可证和准备方法 |
| 模型说明 | `models/README.md` | 权重版本、下载位置和 SHA-256 |
| 大文件附件 | GitHub Release | 正式 Adapter、依赖清单和必要证据包 |
| 文件校验 | `MANIFEST.sha256` | 提交材料完整性校验 |
| 代码许可 | `LICENSE` | 项目代码授权边界 |
| 第三方清单 | `THIRD_PARTY.md` | 模型、数据集和依赖来源 |

## 正式指标证据

| 指标组 | 文档 | 证据目录 |
|---|---|---|
| 两场景漂移恢复 | `docs/实验结果总览_20260809.md` | `artifacts/modelnet40/`、`artifacts/boxcars/` |
| TTFT 与内存 | `module_edge_perception/benchmarks/` | `artifacts/performance/` |
| 弱网业务保持 | `docs/网络波动模拟器设计.md` | `artifacts/network/` |
| 冲突检测与仲裁 | `docs/多节点冲突仲裁测试结果_20260811.md` | `artifacts/multi_node/` |

## 提交前确认

- Git 标签：`challengecup-2026-final`
- Release 名称：`EdgeCloud Challenge Cup 2026 Final`
- 压缩包命名：`东南大学-王成洋-EdgeCloud-联系电话.zip`
- 报告、视频、代码和数据说明均能离线打开
- Release 下载链接无需登录
- `MANIFEST.sha256` 已在最终压缩包生成后重新计算
- 新增权重和证据包后，使用 `find submission docs/assets artifacts -type f -print0 | sort -z | xargs -0 shasum -a 256` 重新生成哈希清单
- README、作品报告、网页和视频中的指标口径一致
