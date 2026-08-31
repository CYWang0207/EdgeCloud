# GitHub Release 附件清单

本文件准备最终 Release，不执行上传。发布者在 GitHub 建立
`EdgeCloud Challenge Cup 2026 Final` 后，上传下列本机已有文件，并将 Release 标记为公开。

| Release 文件名 | 本机来源 | 大小 | SHA-256 |
|---|---|---:|---|
| `modelnet40_cloud_teacher_adapter_best.pth` | `shared/models/modelnet40_cloud_teacher_adapter_20260812/cloud_unlabeled/best.pth` | 1,219,859 | `1e24728b3ffa1f44f0dfd1db64c7b0f2e195566df8cb2b38e2dbebb037f4d82a` |
| `boxcars_cloud_teacher_adapter_best.pth` | `shared/models/boxcars_cloud_teacher_adapter_20260809/cloud_unlabeled/best.pth` | 1,216,745 | `78aa499188e4fd5f7d966fbad5ca01f3fcaa550e5c6524b1ea19fe091593410a` |
| `mv_vit_token_epoch_30.pth` | `shared/models/modelnet40-baseline.pth` | 86,795,875 | `5dab25c437bfc0692c39ead16f778298830b788726c284e95642ffe670d85645` |
| `boxcars_make_baseline_best.pth` | `shared/models/boxcars-baseline.pth` | 260,246,732 | `2c925452089814c7cd9267da3b56188a8eba103e39a40fa61a87879ad3b18f0d` |
| `edgecloud-weights.sha256` | 由发布者根据最终上传文件生成 | - | - |

InternViT-6B-224px 请从其官方 Hugging Face 页面下载，不在本 Release 重新分发。复现快照两份权重
分片的 SHA-256 已记录在 `models/README.md`；仍应将最终下载快照中所有文件的 SHA-256 写入
`edgecloud-weights.sha256`，并注明所用 revision。完整数据集同样不上传至 Release。

发布前执行：

```bash
shasum -a 256 \
  shared/models/modelnet40_cloud_teacher_adapter_20260812/cloud_unlabeled/best.pth \
  shared/models/boxcars_cloud_teacher_adapter_20260809/cloud_unlabeled/best.pth \
  shared/models/modelnet40-baseline.pth \
  shared/models/boxcars-baseline.pth
```

对外发布、创建 `challengecup-2026-final` Git tag 和上传附件需由仓库负责人确认后完成。
