# models · 模型权重清单

模型权重不直接进入 Git 历史。正式文件通过 [GitHub Release](https://github.com/CYWang0207/EdgeCloud/releases/latest) 提供；文件名、校验值和发布步骤见 [`RELEASE_ASSETS.md`](../RELEASE_ASSETS.md)。下载后按以下结构放置：

```text
models/
├── InternViT-6B-224px/                                # 云端冻结视觉教师
├── mv_vit_token_epoch_30.pth                          # ModelNet40 Edge baseline
├── boxcars_make_baseline/
│   └── best.pth                                       # BoxCars Edge baseline
├── modelnet40_cloud_teacher_adapter/
│   └── cloud_unlabeled/best.pth                       # ModelNet40 正式 Adapter
└── boxcars_cloud_teacher_adapter/
    └── cloud_unlabeled/best.pth                       # BoxCars 正式 Adapter
```

## 正式 Adapter

| 场景 | 参数量 | 大小 | SHA-256 |
|---|---:|---:|---|
| ModelNet40 | 299,916 | 1,219,859 bytes | `1e24728b3ffa1f44f0dfd1db64c7b0f2e195566df8cb2b38e2dbebb037f4d82a` |
| BoxCars116k | 299,916 | 1,216,745 bytes | `78aa499188e4fd5f7d966fbad5ca01f3fcaa550e5c6524b1ea19fe091593410a` |

Adapter checkpoint 只包含 AdaptFormer 参数，不包含 MV-ViT backbone、分类头、归一化层或云端 projector。两场景权重相互隔离，不跨数据集复用。

## 校验

```bash
shasum -a 256 models/modelnet40_cloud_teacher_adapter/cloud_unlabeled/best.pth
shasum -a 256 models/boxcars_cloud_teacher_adapter/cloud_unlabeled/best.pth
```

## Edge baseline

| 场景 | Release 文件 | 大小 | SHA-256 | 版本/适用范围 |
|---|---|---:|---|---|
| ModelNet40 | `mv_vit_token_epoch_30.pth` | 86,795,875 bytes | `5dab25c437bfc0692c39ead16f778298830b788726c284e95642ffe670d85645` | MV-ViT-S，40 类四视图 Edge baseline |
| BoxCars116k | `boxcars_make_baseline_best.pth` | 260,246,732 bytes | `2c925452089814c7cd9267da3b56188a8eba103e39a40fa61a87879ad3b18f0d` | MV-ViT-S，官方 `make` 16 类 Edge baseline |

## 云端教师

- 模型：[`OpenGVLab/InternViT-6B-224px`](https://huggingface.co/OpenGVLab/InternViT-6B-224px)，224px 视觉教师版本。
- 用途：云端离线生成 task logits 与 3,200D 视觉特征；不参与边缘在线前向。
- 下载：`huggingface-cli download OpenGVLab/InternViT-6B-224px --local-dir models/InternViT-6B-224px`
- 校验：Release 附件 `edgecloud-weights.sha256` 给出本次复现使用的完整文件 SHA-256；下载后执行
  `shasum -a 256 -c edgecloud-weights.sha256`。其 `config.json` 校验值为
  `7c267cf46453bd2e6d07d3136c7bb4f842664be5ac9f7cb1b196cefd4cdb1026`；本项目复现快照的
  `pytorch_model-00001-of-00002.bin` 为
  `c49bb4c1a68fa2a45999324f70cbae8a64f6c9be78dbcd754c27079a983a9298`，
  `pytorch_model-00002-of-00002.bin` 为
  `6a381569a3ace3cce7e74dad9682614942d2d057c3db0ef7d6339ce7951a51bb`。

Release 说明必须包含每个文件的版本、大小、SHA-256、训练 commit、适用场景和许可证边界；不要将
模型权重提交到 Git 历史。
