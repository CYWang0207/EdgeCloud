"""main_edge_cloud_real_model.py — 主循环 + 真模型 + adapter + network_sim 全链路闭环（王成洋 8/11）。

在 main_edge_cloud_new.py 的调度主循环基础上，把"轨迹 CSV 里的 pred/confidence 当感知结果"
替换为"真模型按调度决策 (v*, u*, k*) 推理"：

  每时隙 t:
    1. net.step()              采样 R_t / B_t / net_state / is_disconnected
    2. Actor 生成 (v, u) 候选（状态 = [E_drift, struct_drift, Y_bw/100]）
    3. net.filter_candidates   断联→强制 u=0；--strict-bandwidth→硬过滤超带宽候选
    4. Critic(注水) 评估候选 + net.apply_network_penalty 超带宽软罚 → 选 (v*, u*, k*)
    5. ★ 真模型推理（感知替换核心）：按 v* 选视角（view_mask）、按 k* 保留 token，
       u* 决定用哪个模型（u=0 本地旧 adapter / u=1 云端新 adapter / u=2 重训权重）
       → 得到真实 pred / confidence
    6. net.compute_e2e         前台 T_comm（u=0 特判 0）→ T_e2e
    7. net.is_business_available 业务可用五条件（proxy_acc = 真模型 confidence）
    8. net.update_queues        更新 Y_bw + Q_net（后台大包跨槽消化）

对照基线：轨迹 CSV 的 pred/confidence 本身就是 generate_*_trajectory.py 用全视角模型
跑出来的，直接用作 baseline 对照，不再额外全视角推理。

u 语义（cloud-teacher 口径）：
  u=0 本地自治：--adapter-checkpoint-before（本地旧 adapter，缺省则无 adapter 的 base）
  u=1 adapter 同步：--adapter-checkpoint（云端新下发 adapter）
  u=2 重训权重：--retrain-checkpoint（整模型；缺省则退化为 u=1 的 adapter 模型）

理论故事与 main_edge_cloud_new 完全一致：Q_net 是物理积压队列不进 G 罚项，
长期平均带宽约束仍由 Y_bw 单一虚拟队列保证，Lyapunov 证明不变。

在 AutoDL 上跑（需要 CUDA + 数据集 + baseline.pth + cloud-teacher adapter_best.pth）。
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

from network_sim import DEFAULT_ACC_FLOOR, NetworkSimulator
from actor_memory import CollaborativeMemoryDNN
from critic_water_filling import WaterFilling_Critic

SCRIPT_DIR = Path(__file__).resolve().parent

# 感知模块（model.py / adaptformer.py / dataset）与调度模块同级
_PERCEPTION_DIR = SCRIPT_DIR.parent.parent / "module_edge_perception"
for _p in (str(SCRIPT_DIR), str(_PERCEPTION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import EarlyFusionMultiViewViT  # noqa: E402
from adaptformer import (  # noqa: E402
    attach_adaptformer,
    load_adapter_checkpoint,
    set_adapter_enabled,
)
from dataset import ModelNet40MultiView  # noqa: E402
from boxcars_dataset import BoxCarsMultiView, VALID_TASKS  # noqa: E402

DEFAULT_VIEW_KEEP_RATIO = 0.1  # v=0（未激活视角）的 token 保留率，与 evaluate_rl_policy 的 inactive-view-keep 一致


# ---------- 数据集 ----------
def build_dataset(args, transform):
    """按 --scene 建数据集，返回 (dataset, num_classes, num_views)。"""
    if args.scene == "modelnet":
        dataset = ModelNet40MultiView(
            root_dir=str(args.dataset_path),
            split=args.split,
            transform=transform,
            num_views=args.num_views,
        )
        num_classes = len(dataset.classes)
        return dataset, num_classes, args.num_views
    if args.scene == "boxcars":
        dataset = BoxCarsMultiView(
            args.dataset_path, args.split, args.task, args.num_views, transform
        )
        num_classes = len(dataset.classes)
        return dataset, num_classes, args.num_views
    raise ValueError(f"未知场景: {args.scene}（支持 modelnet | boxcars）")


def build_model(args, checkpoint_path, num_classes, device):
    """baseline 主干 + 挂 adapter（未加载权重）。返回模型，供 u=0/u=1 加载不同 adapter。"""
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"baseline 主干加载: missing={len(missing)}, unexpected={len(unexpected)}")
    attach_adaptformer(model, r=args.adapter_r)
    model.to(device)  # attach 后新模块在 CPU，再迁移一次
    model.eval()
    return model


def load_models(args, num_classes, device):
    """加载三套边侧模型变体：

    - base_model：baseline 主干 + 零初始化 adapter（u=0 且无旧 adapter 时用）
    - before_model：baseline + 本地旧 adapter（u=0 本地自治；--adapter-checkpoint-before）
    - current_model：baseline + 云端新 adapter（u=1 adapter 同步；--adapter-checkpoint）
    - retrain_model：重训整模型（u=2；--retrain-checkpoint，可选）
    """
    base_model = build_model(args, args.checkpoint, num_classes, device)

    before_model = None
    if args.adapter_checkpoint_before is not None:
        before_model = build_model(args, args.checkpoint, num_classes, device)
        miss, unexp = load_adapter_checkpoint(before_model, args.adapter_checkpoint_before, device)
        print(f"本地旧 adapter（u=0）加载: missing={len(miss)}, unexpected={len(unexp)}")

    current_model = None
    if args.adapter_checkpoint is not None:
        current_model = build_model(args, args.checkpoint, num_classes, device)
        miss, unexp = load_adapter_checkpoint(current_model, args.adapter_checkpoint, device)
        print(f"云端新 adapter（u=1）加载: missing={len(miss)}, unexpected={len(unexp)}")

    retrain_model = None
    if args.retrain_checkpoint is not None:
        retrain_model = build_model(args, args.retrain_checkpoint, num_classes, device)
        print(f"重训权重（u=2）加载: {args.retrain_checkpoint}")

    return base_model, before_model, current_model, retrain_model


def effective_keep_ratios(view_mask, keep_ratios, inactive_keep=DEFAULT_VIEW_KEEP_RATIO):
    """v=0 的视角 token 保留率压到 inactive_keep，v=1 用调度 k_t。与 evaluate_rl_policy 一致。"""
    keep = keep_ratios.astype(np.float32).copy()
    keep = np.where(view_mask == 1, keep, inactive_keep)
    return np.clip(keep, 0.0, 1.0)


def select_model_for_u(u, base_model, before_model, current_model, retrain_model):
    """按 u 选推理模型。u=0→旧 adapter（无则 base）；u=1→新 adapter；u=2→重训（无则新 adapter 兜底）。"""
    if u == 0:
        return before_model if before_model is not None else base_model
    if u == 1:
        if current_model is None:
            raise ValueError("u=1 需要 --adapter-checkpoint（云端新 adapter）")
        return current_model
    if u == 2:
        if retrain_model is not None:
            return retrain_model
        if current_model is not None:
            return current_model
        return base_model  # 无重训权重也无新 adapter 时退化为 base
    raise ValueError(f"非法 u={u}")


# ---------- 轨迹加载（与 main_edge_cloud_new 一致）----------
def load_trajectory(path, num_views):
    if not path.exists():
        raise FileNotFoundError(f"找不到轨迹文件: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"轨迹文件为空: {path}")
    required = ["E_drift", *[f"w_{i + 1}" for i in range(num_views)]]
    missing = [col for col in required if col not in rows[0]]
    if missing:
        raise ValueError(f"轨迹文件缺少必要列: {missing}")
    return rows


def float_or_empty(row, key):
    value = row.get(key, "")
    return "" if value == "" else float(value)


def int_or_empty(row, key):
    value = row.get(key, "")
    return "" if value == "" else int(float(value))


# ---------- 参数 ----------
def parse_args():
    default_dataset = _PERCEPTION_DIR / "data" / "modelnet40v2png_ori4"
    default_checkpoint = _PERCEPTION_DIR / "checkpoints" / "mv_vit_base_epoch_30.pth"

    parser = argparse.ArgumentParser(
        description="主循环 + 真模型 + adapter + network_sim 全链路闭环评估（在 AutoDL 上跑）"
    )
    # --- 场景与模型 ---
    parser.add_argument("--scene", default="modelnet", choices=("modelnet", "boxcars"),
                        help="modelnet=ModelNet40（默认）；boxcars=BoxCars116k")
    parser.add_argument("--dataset-path", type=Path, default=default_dataset)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint,
                        help="baseline 主干权重（轨迹 CSV 的 pred 就是它全视角跑的）")
    parser.add_argument("--adapter-checkpoint", type=Path, default=None,
                        help="云端新下发 adapter（u=1 用，cloud-teacher 产物）")
    parser.add_argument("--adapter-checkpoint-before", type=Path, default=None,
                        help="本地旧 adapter（u=0 本地自治用，可选；缺省则 u=0 走无 adapter 的 base）")
    parser.add_argument("--retrain-checkpoint", type=Path, default=None,
                        help="重训整模型权重（u=2 用，可选；缺省则 u=2 退化为 u=1 的 adapter 模型）")
    parser.add_argument("--task", default="make", choices=VALID_TASKS, help="boxcars 场景的分类任务")
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument("--num-classes", type=int, default=40,
                        help="modelnet 场景类数（boxcars 场景自动取数据集 class 数）")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--adapter-r", type=int, default=32)
    parser.add_argument("--inactive-view-keep", type=float, default=DEFAULT_VIEW_KEEP_RATIO,
                        help="v=0 视角的 token 保留率（与 evaluate_rl_policy 默认一致）")
    parser.add_argument("--max-samples", type=int, default=0, help="0=轨迹全量；>0 子集验证")
    parser.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # --- 调度与轨迹（与 main_edge_cloud_new 一致）---
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "real_trajectory_data.csv",
                        help="轨迹 CSV（modelnet 用 modelnet40_trajectory.csv，boxcars 用 real_trajectory_data.csv）")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "rl_decision_real_model.csv")
    parser.add_argument("--b-avg", type=float, default=3.0)
    parser.add_argument("--v-lya", type=float, default=20.0)
    parser.add_argument("--eps-mask", type=float, default=0.05)
    parser.add_argument("--min-active-views", type=float, default=3.0)
    parser.add_argument("--m-candidates", type=int, default=10)
    parser.add_argument("--scl-weights", type=float, default=50.0)
    parser.add_argument("--alpha-env", type=float, default=0.4)
    parser.add_argument("--alpha-struct", type=float, default=0.3)
    parser.add_argument("--retrain-bonus", type=float, default=0.2)
    parser.add_argument("--tau-retrain", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    # --- 网络韧性（与 main_edge_cloud_new 一致）---
    parser.add_argument("--network-mode", default="static",
                        choices=("static", "jitter", "jitter_outage", "markov"))
    parser.add_argument("--slot-duration", type=float, default=0.2)
    parser.add_argument("--edge-delay-ms", type=float, default=80.0,
                        help="T_edge（本脚本默认沿用写死 80ms 以便与模拟版口径可比；真机测 T_edge 后传入实测值）")
    parser.add_argument("--rtt-ms", type=float, default=10.0)
    parser.add_argument("--deadline-ms", type=float, default=200.0)
    parser.add_argument("--acc-floor", type=float, default=DEFAULT_ACC_FLOOR)
    parser.add_argument("--business-min-active-views", type=int, default=3)
    parser.add_argument("--bw-min-mbps", type=float, default=20.0)
    parser.add_argument("--bw-max-mbps", type=float, default=120.0)
    parser.add_argument("--disconnect-prob", type=float, default=0.0)
    parser.add_argument("--outage-period", type=int, default=0)
    parser.add_argument("--outage-duration", type=int, default=0)
    parser.add_argument("--strict-bandwidth", action="store_true")
    parser.add_argument("--sync-u2", action="store_true")
    parser.add_argument("--adapter-size-mb", type=float, default=1.2)
    parser.add_argument("--u2-update-size-mb", type=float, default=50.0)
    return parser.parse_args()


# ---------- 日志 ----------
def build_log_row(row, t, y_bw, w_t, struct_drift, v_opt, u_opt, k_opt,
                  c_comm, acc, cost, best_g, net_state, e2e_info,
                  business_available, q_net, real_pred, real_conf, real_correct,
                  traj_pred, traj_correct):
    """main_edge_cloud_new 的决策日志字段 + 真模型推理列（real_*）+ 轨迹基线对照列。"""
    log_row = {
        "t": t,
        "sample_id": int_or_empty(row, "sample_id"),
        "label": int_or_empty(row, "label"),
        "u": int(u_opt),
        "c_comm": float(c_comm),
        "proxy_acc": float(acc),          # = 真模型 confidence（感知替换后）
        "cost": float(cost),
        "G": float(best_g),
        "Y_bw": float(y_bw),
        "Q_net": float(q_net),
        "active_views": int(np.sum(v_opt)),
        "active_token_ratio_sum": float(np.sum(k_opt)),
        "network_state": net_state.get("network_state", ""),
        "bandwidth_mbps": float(net_state.get("bandwidth_mbps", 0.0)),
        "effective_bandwidth_mbps": float(net_state.get("effective_bandwidth_mbps", 0.0)),
        "B_t": float(net_state.get("B_t", 0.0)),
        "is_disconnected": int(bool(net_state.get("is_disconnected", False))),
        "comm_delay_ms": float(e2e_info.get("comm_delay_ms", 0.0)),
        "e2e_delay_ms": float(e2e_info.get("e2e_delay_ms", 0.0)),
        "deadline_met": int(bool(e2e_info.get("deadline_met", False))),
        "transmission_success": int(bool(e2e_info.get("transmission_success", True))),
        "business_available": int(bool(business_available)),
        # --- 真模型感知结果（本脚本核心新增）---
        "real_pred": int(real_pred),
        "real_conf": float(real_conf),
        "real_correct": int(real_correct),
        "traj_pred": int_or_empty(row, "pred"),
        "traj_correct": int(traj_correct),
    }
    for i, value in enumerate(w_t, start=1):
        log_row[f"w_{i}"] = float(value)
    for i, value in enumerate(v_opt, start=1):
        log_row[f"v_{i}"] = int(value)
    for i, value in enumerate(k_opt, start=1):
        log_row[f"k_{i}"] = float(value)
    return log_row


def save_decision_log(path, rows, num_views):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t", "sample_id", "label", "u", "c_comm", "proxy_acc", "cost", "G",
        "Y_bw", "Q_net", "active_views", "active_token_ratio_sum",
        "network_state", "bandwidth_mbps", "effective_bandwidth_mbps", "B_t",
        "is_disconnected", "comm_delay_ms", "e2e_delay_ms", "deadline_met",
        "transmission_success", "business_available",
        "real_pred", "real_conf", "real_correct", "traj_pred", "traj_correct",
        *[f"w_{i + 1}" for i in range(num_views)],
        *[f"v_{i + 1}" for i in range(num_views)],
        *[f"k_{i + 1}" for i in range(num_views)],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(log_rows, y_bw):
    total = len(log_rows)
    if total == 0:
        print("无时隙数据")
        return

    u_counter = Counter(row["u"] for row in log_rows)
    business_rate = np.mean([row["business_available"] for row in log_rows])
    deadline_rate = np.mean([row["deadline_met"] for row in log_rows])
    disconnect_rate = np.mean([row["is_disconnected"] for row in log_rows])
    e2e_values = [row["e2e_delay_ms"] for row in log_rows
                  if math.isfinite(row["e2e_delay_ms"])]
    avg_e2e = float(np.mean(e2e_values)) if e2e_values else float("nan")
    p95_e2e = float(np.percentile(e2e_values, 95)) if e2e_values else float("nan")

    real_correct = [row["real_correct"] for row in log_rows]
    traj_correct = [row["traj_correct"] for row in log_rows]
    real_acc = np.mean(real_correct)
    traj_acc = np.mean(traj_correct)
    avg_conf = np.mean([row["real_conf"] for row in log_rows])
    avg_active_views = np.mean([row["active_views"] for row in log_rows])
    avg_token_sum = np.mean([row["active_token_ratio_sum"] for row in log_rows])

    # 按漂移类型分组的策略准确率（轨迹 drift_type 列）
    drift_groups = {}
    for row in log_rows:
        key = str(row.get("drift_type", "normal") or "normal")
        drift_groups.setdefault(key, {"correct": 0, "n": 0})
        drift_groups[key]["correct"] += row["real_correct"]
        drift_groups[key]["n"] += 1

    print("-" * 72)
    print("全链路闭环评估完成（主循环 + 真模型 + adapter + network_sim）")
    print(f"总时隙数: {total}")
    print(f"最终带宽队列 Y_bw: {y_bw[-1]:.4f}")
    print(f"--- 硬指标 ---")
    print(f"业务保持率: {business_rate:.2%}  (硬指标 ≥90%)")
    print(f"端到端时延达标率: {deadline_rate:.2%}  平均={avg_e2e:.1f}ms  P95={p95_e2e:.1f}ms  (≤200ms)")
    print(f"断联率: {disconnect_rate:.2%}")
    print(f"--- 真模型感知 ---")
    print(f"真模型策略准确率（按调度 v/k/u 推理）: {real_acc:.4%}")
    print(f"轨迹基线准确率（全视角 baseline 推理）: {traj_acc:.4%}  (差距 {real_acc - traj_acc:+.4%})")
    print(f"真模型平均置信度: {avg_conf:.4f}")
    print(f"平均激活视角数: {avg_active_views:.4f}/{len([k for k in log_rows[0] if k.startswith('v_')])}")
    print(f"平均 Token 保留率总和: {avg_token_sum:.4f}")
    print("--- u 分布与按漂移分组策略准确率 ---")
    for u in [0, 1, 2]:
        count = u_counter.get(u, 0)
        print(f"u={u} 次数: {count} ({count / total:.2%})")
    for key, val in sorted(drift_groups.items()):
        if val["n"] > 0:
            print(f"  {key}: n={val['n']}  real_acc={val['correct'] / val['n']:.4%}")


# ---------- 主循环 ----------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("警告: CUDA 不可用，回退 CPU（全链路评估在 AutoDL 上跑）")
        device = torch.device("cpu")

    # --- 1. 系统参数（与 main_edge_cloud_new 一致）---
    num_views = args.num_views
    sys_params = {
        "N": 196,
        "eta": 5e-5,
        "gamma": 10.0,
        "k_min": 0.1,
        "beta_0": 0.2,
        "S_adapter": args.adapter_size_mb,
        "SCL_weights": args.u2_update_size_mb,
        "S_query": 5.0,
        "alpha_env": args.alpha_env,
        "alpha_struct": args.alpha_struct,
        "retrain_bonus": args.retrain_bonus,
        "tau_retrain": args.tau_retrain,
    }
    c_comm_map = {
        0: 0.0,
        1: sys_params["S_adapter"] + sys_params["S_query"],
        2: sys_params["SCL_weights"] + sys_params["S_query"],
    }

    # --- 2. 数据与模型 ---
    trajectory_rows = load_trajectory(args.input, num_views)
    total_slots = len(trajectory_rows)
    if args.max_samples and args.max_samples > 0:
        total_slots = min(total_slots, args.max_samples)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset, num_classes, _ = build_dataset(args, transform)
    if num_classes != args.num_classes and args.scene == "modelnet":
        # modelnet 场景类数从数据集取，避免 40 写错
        args.num_classes = num_classes
    print(f"{args.scene} {args.split}: {len(dataset)} 样本, {num_classes} 类; 轨迹 {total_slots} 时隙")

    base_model, before_model, current_model, retrain_model = load_models(
        args, num_classes, device
    )
    if current_model is None and args.adapter_checkpoint is None:
        print("提示: 未提供 --adapter-checkpoint，u=1 时隙将回退 base 推理")

    # --- 3. 调度器 + 网络韧性 ---
    mem = CollaborativeMemoryDNN(
        V=num_views,
        state_dim=3,
        eps_mask=args.eps_mask,
        min_active_views=args.min_active_views,
    )
    net = NetworkSimulator(
        mode=args.network_mode,
        slot_duration=args.slot_duration,
        b_avg=args.b_avg,
        bandwidth_min_mbps=args.bw_min_mbps,
        bandwidth_max_mbps=args.bw_max_mbps,
        disconnect_prob=args.disconnect_prob,
        outage_period=args.outage_period,
        outage_duration=args.outage_duration,
        rtt_ms=args.rtt_ms,
        edge_delay_ms=args.edge_delay_ms,
        deadline_ms=args.deadline_ms,
        acc_floor=args.acc_floor,
        business_min_active_views=args.business_min_active_views,
        adapter_size_mb=args.adapter_size_mb,
        u2_update_size_mb=args.u2_update_size_mb,
        strict_bandwidth=args.strict_bandwidth,
        sync_u2=args.sync_u2,
        seed=args.seed,
    )
    y_bw = np.zeros(total_slots)
    decision_log = []

    print(f"网络模式: {args.network_mode}, deadline={args.deadline_ms}ms, "
          f"strict_bw={args.strict_bandwidth}, sync_u2={args.sync_u2}")
    print(f"u=0→{('本地旧 adapter' if before_model is not None else 'base(无 adapter)')}  "
          f"u=1→{'云端新 adapter' if current_model is not None else 'base'}  "
          f"u=2→{'重训权重' if retrain_model is not None else '新 adapter 兜底'}")
    print("-" * 72)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp_dtype]
    use_amp = args.amp_dtype != "fp32"

    # --- 4. 主循环：调度决策 → 真模型推理 → 网络韧性 ---
    for t, row in enumerate(trajectory_rows):
        if t >= total_slots:
            break

        net_state = net.step()
        is_disc = bool(net_state["is_disconnected"])

        e_drift = float(row["E_drift"])
        struct_drift = float(row.get("struct_drift", 0.0) or 0.0)
        w_t = np.array([float(row[f"w_{i + 1}"]) for i in range(num_views)], dtype=float)

        y_bw[t] = net.y_bw
        state_t = np.array([e_drift, struct_drift, y_bw[t] / 100.0], dtype=float)

        # 4.1 Actor 候选 + 网络过滤
        candidates = mem.decode_and_quantize(state_t, w_t, M_t=args.m_candidates)
        filtered_result = net.filter_candidates(candidates, c_comm_map)
        feasible = filtered_result["candidates"]

        # 4.2 Critic 评估 + 超带宽软罚
        best_g = -np.inf
        best_action = None
        best_details = None
        if not feasible:
            v_fallback = np.zeros(num_views, dtype=int)
            v_fallback[0] = 1
            feasible = [(v_fallback, 0)]
        for v_cand, u_cand in feasible:
            g_raw, k_t, c_comm, acc, cost = WaterFilling_Critic(
                v_cand, u_cand, w_t, e_drift, struct_drift, y_bw[t], args.v_lya, sys_params
            )
            realtime_comm = net.realtime_comm_mb(u_cand, c_comm)
            g_effective = net.apply_network_penalty(g_raw, realtime_comm)["G_effective"]
            if g_effective > best_g:
                best_g = g_effective
                best_action = (v_cand, u_cand, k_t)
                best_details = (c_comm, acc, cost)

        v_opt, u_opt, k_opt = best_action
        c_comm_opt, acc_opt, cost_opt = best_details

        # 4.3 ★ 真模型推理（感知替换核心）：取轨迹样本 → 按 v*/k*/u* 推理
        sample_id = int(float(row["sample_id"])) if row.get("sample_id", "") != "" else t
        item = dataset[sample_id]
        if args.scene == "boxcars":
            views, _full_mask, label, _meta = item  # BoxCars 返回 (views, view_mask, label, meta)
        else:
            views, label = item                     # ModelNet 返回 (views, label)
        images = views.unsqueeze(0).to(device)  # [1, V, C, H, W]
        label = int(label)

        vm_tensor = torch.tensor([v_opt], dtype=torch.float32, device=device)
        keep_np = effective_keep_ratios(v_opt, k_opt, args.inactive_view_keep)
        keep_tensor = torch.tensor([keep_np], dtype=torch.float32, device=device)

        active_model = select_model_for_u(
            u_opt, base_model, before_model, current_model, retrain_model
        )
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = active_model(
                        images, view_mask=vm_tensor, keep_ratios=keep_tensor,
                        token_score_mode="importance",
                    )
            else:
                logits = active_model(
                    images, view_mask=vm_tensor, keep_ratios=keep_tensor,
                    token_score_mode="importance",
                )
        probs = torch.softmax(logits.float(), dim=1)[0]
        real_conf, real_pred = float(probs.max().item()), int(probs.argmax().item())
        real_correct = int(real_pred == label)

        # 轨迹 pred 是 generate 脚本用全视角 baseline 跑的 → 直接作 baseline 对照
        traj_pred = int_or_empty(row, "pred")
        traj_correct = int(traj_pred == label) if row.get("pred", "") != "" else -1

        # 4.4 端到端时延 + 业务可用（proxy_acc = 真模型 confidence）
        e2e_info = net.compute_e2e(
            u_opt, net.realtime_comm_mb(u_opt, c_comm_opt), t_edge=args.edge_delay_ms
        )
        decision_success = (u_opt == 0) or (not is_disc)
        business_available = net.is_business_available(
            decision_success=decision_success,
            e2e_ms=e2e_info["e2e_delay_ms"],
            active_views=int(np.sum(v_opt)),
            proxy_acc=real_conf,  # 感知替换：用真模型置信度当质量代理
            transmission_success=e2e_info["transmission_success"],
        )

        # 4.5 队列更新
        queue_result = net.update_queues(c_comm=c_comm_opt, b_avg=args.b_avg, u=u_opt)
        q_net = queue_result["Q_net"]

        mem.encode(state_t, v_opt, u_opt)

        decision_log.append(
            build_log_row(
                row, t, y_bw[t], w_t, struct_drift, v_opt, u_opt, k_opt,
                c_comm_opt, real_conf, cost_opt, best_g, net_state, e2e_info,
                business_available, q_net, real_pred, real_conf, real_correct,
                traj_pred, traj_correct,
            )
        )

        if t % 100 == 0 or t == total_slots - 1:
            print(
                f"时隙 {t:>4}: Y={y_bw[t]:>7.2f} Q={q_net:>6.2f} | v={v_opt} | u={u_opt} | "
                f"net={net_state['network_state']:>7} | real_pred={real_pred} "
                f"conf={real_conf:.3f} {'✓' if real_correct else '✗'} | "
                f"e2e={e2e_info['e2e_delay_ms']:>6.1f}ms | biz={int(business_available)}"
            )

    save_decision_log(args.output, decision_log, num_views)
    print_summary(decision_log, y_bw)
    print(f"决策日志已保存: {args.output}")


if __name__ == "__main__":
    main()
