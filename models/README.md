# models · 模型权重目录（内容不进 Git）

**模型权重（*.pth / *.onnx / *.bin / *.safetensors 等）不进 Git 仓库**，本目录在仓库中只保留本说明文件。

## 使用方式

1. 从团队网盘或实验服务器下载所需权重。
2. 按数据集放入独立目录。本目录只放最终向 Edge 下发的 Adapter：

```
models/
├── README.md                                              # 本文件（唯一进 Git 的文件）
├── mvvit_base/                                            # MV-ViT 基座权重（网盘下载）
├── boxcars_cloud_teacher_adapter_20260809/
│   └── cloud_unlabeled/best.pth                           # BoxCars 正式下发 adapter（约 1.2MB）
└── modelnet40_cloud_teacher_adapter_20260809/
    └── cloud_unlabeled/best.pth                           # ModelNet40 正式下发 adapter（约 1.2MB）
```

因此当前恰好只有两个正式模型文件，每个数据集一个。云端 task head 属于实验 artifact，
放在对应 `local/results/<dataset>_.../artifacts/`；Edge baseline 和 InternViT backbone
是输入依赖，也不在这里重复存放。

`.gitignore` 已配置忽略本目录下除 README 外的所有内容，正常 `git add .` 不会误传权重。
