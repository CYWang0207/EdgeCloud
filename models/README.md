# models · 模型权重目录（内容不进 Git）

**模型权重（*.pth / *.onnx / *.bin / *.safetensors 等）不进 Git 仓库**，本目录在仓库中只保留本说明文件。

## 使用方式

1. 从团队网盘或实验服务器下载所需权重。
2. 按数据集和方法放入独立目录。例如当前本机 ModelNet40 正式新权重：

```
models/
├── README.md
└── internvit6b_modelnet40_cloud_teacher_20260809/
    ├── task_head/selected_head.pth
    └── cloud_unlabeled_illumination_tuned/best.pth
```

这两个文件分别是冻结 InternViT-6B 使用的 40 类 task head，以及最终下发 Edge 的
Adapter。Edge baseline 和 InternViT backbone 是输入依赖，不作为本次新权重重复存放。

`.gitignore` 已配置忽略本目录下除 README 外的所有内容，正常 `git add .` 不会误传权重。
