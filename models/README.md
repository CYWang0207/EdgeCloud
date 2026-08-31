# models · 模型权重清单

模型权重不直接进入 Git 历史，正式文件通过 [GitHub Release](https://github.com/CYWang0207/EdgeCloud/releases/latest) 提供。下载后按以下结构放置：

```text
models/
├── InternViT-6B/                                      # 云端冻结视觉教师
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
| BoxCars116k | 299,916 | 1,216,745 bytes | 提交前按 Release 文件补充 |

Adapter checkpoint 只包含 AdaptFormer 参数，不包含 MV-ViT backbone、分类头、归一化层或云端 projector。两场景权重相互隔离，不跨数据集复用。

## 校验

```bash
shasum -a 256 models/modelnet40_cloud_teacher_adapter/cloud_unlabeled/best.pth
shasum -a 256 models/boxcars_cloud_teacher_adapter/cloud_unlabeled/best.pth
```

正式提交前必须在 Release 说明中补齐每个文件的版本、大小、SHA-256、训练 commit、适用场景和许可证边界。
