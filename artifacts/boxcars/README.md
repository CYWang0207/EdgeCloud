# BoxCars116k 正式证据

实验：官方 `make/test` 全量 12,322 车辆轨迹，16 类品牌识别；结果标签为**实机测量
（GPU 推理）**，退化为**合成相机退化**。随机种子 42，正式 Adapter 在测试前固定；复现提交为
`49c3f7a`，命令见 `scripts/reproduce_boxcars.py`。

`boxcars_recovery_metrics.json` 保留成对 bootstrap 95% CI 与完整定义，`metrics.csv`
可直接用于正式恢复精度/业务保持率图表。
