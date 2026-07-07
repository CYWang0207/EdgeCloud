# models · 模型权重目录（内容不进 Git）

**模型权重（*.pth / *.onnx / *.bin / *.safetensors 等）不进 Git 仓库**，本目录在仓库中只保留本说明文件。

## 使用方式

1. 从团队网盘下载所需权重（链接待补充：___________）
2. 放到本目录下，例如：

```
models/
├── README.md            # 本文件（唯一进 Git 的文件）
├── mvvit_base/          # MV-ViT 基座权重（网盘下载）
├── supernet_variants/   # 模块 A 产出的 SubNet 变体族（网盘下载）
└── lut/                 # 预测器查找表（若体积小可讨论后入库）
```

`.gitignore` 已配置忽略本目录下除 README 外的所有内容，正常 `git add .` 不会误传权重。
