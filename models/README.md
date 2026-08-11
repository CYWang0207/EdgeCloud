# models · 模型权重目录（内容不进 Git）

**模型权重（*.pth / *.onnx / *.bin / *.safetensors 等）不进 Git 仓库**，本目录在仓库中只保留本说明文件。

## 使用方式

1. 从团队网盘下载所需权重（链接待补充：___________）
2. 放到本目录下，例如：

```
models/
├── README.md                                              # 本文件（唯一进 Git 的文件）
├── mvvit_base/                                            # MV-ViT 基座权重（网盘下载）
├── boxcars_cloud_teacher_adapter_20260809/
│   └── cloud_unlabeled/best.pth                           # BoxCars 正式下发 adapter（约 1.2MB）
└── modelnet40_cloud_teacher_adapter_20260809/
    └── cloud_unlabeled/best.pth                           # ModelNet40 正式下发 adapter（约 1.2MB）
```

`.gitignore` 已配置忽略本目录下除 README 外的所有内容，正常 `git add .` 不会误传权重。
