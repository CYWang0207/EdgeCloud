# 历史代码归档

此目录保存不属于正式 cloud-teacher Adapter 交付链路的实现，以保留可追溯性而不让
评审误将其当作当前入口。

- `scheduling/main_edge_cloud.py`：已失效的早期调度接口；当前入口是
  `module_scheduling/EdgeCloud_RL/main_edge_cloud_new.py` 或 `main_edge_cloud_real_model.py`。
- `utilities/tu.py`、`utilities/tuzhe.py`：无正式复现职责的历史作图工具。
- `prompt_experiments/`：两套历史 Prompt 路线及依赖脚本；正式链路统一为冻结
  InternViT-6B → cloud-teacher supervision → AdaptFormer adapter-only 下发。

归档代码不纳入 CI、正式指标和 README 复现命令。
