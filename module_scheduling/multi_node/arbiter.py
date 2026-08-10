"""multi_node/arbiter.py — 多节点冲突仲裁器（思路A：纯逻辑 + 模拟自测）。

职责：收多节点上报 → 检测冲突（同 target + 预测不一 + 双方 conf>0.7）
     → 仲裁（置信度加权投票 / 贝叶斯产品融合，可插拔）→ 下发 → 回滚 → 统计。

纯 Python（random + math），不依赖 torch/numpy/真模型，本地 `python arbiter.py` 可跑。
对应硬指标：冲突率 ≤5% / 解决率 ≥90%。

思路B（可选，时间紧可跳过）：接 main_edge_cloud_new 主循环多节点仿真，
用真实轨迹数据（real_trajectory_data.csv 的 pred/confidence）替换模拟数据。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional


CONFLICT_CONF_THRESHOLD = 0.7   # 双方置信度阈值（CLAUDE.md 已定）


@dataclass
class Report:
    """一个节点对一个目标的上报。"""
    node_id: int
    target_id: int
    pred: int                              # 预测类别 index
    conf: float                            # Top-1 置信度
    softmax: Optional[List[float]] = None  # 完整 softmax（贝叶斯融合用），可选


@dataclass
class Conflict:
    """一个冲突：同 target 多节点 + 预测不一 + 双方 conf>0.7。"""
    target_id: int
    reports: List[Report]


@dataclass
class RollbackEvent:
    """回滚事件：报错节点把本轮预测改到仲裁结果（或平局未解决）。"""
    node_id: int
    target_id: int
    original_pred: int
    final_pred: int        # -1 表示平局未仲裁
    resolved: bool


# ---------- 融合策略（可插拔）----------
class Fusion:
    """融合策略接口：给定同 target 的多节点上报，返回最终预测（平局返回 None）。"""

    def combine(self, reports: List[Report]) -> Optional[int]:
        raise NotImplementedError


class WeightedVoteFusion(Fusion):
    """置信度加权投票：score(c) = Σ conf_i（预测 c 的节点），argmax；平局返回 None。"""

    def combine(self, reports):
        scores = {}
        for r in reports:
            scores[r.pred] = scores.get(r.pred, 0.0) + r.conf
        max_score = max(scores.values())
        winners = [p for p, s in scores.items() if abs(s - max_score) < 1e-9]
        return winners[0] if len(winners) == 1 else None


class BayesianFusion(Fusion):
    """贝叶斯产品融合：P(c) ∝ Π softmax_i(c)，用满分布；需各 report 带 softmax，平局返回 None。"""

    def combine(self, reports):
        sm = [r.softmax for r in reports if r.softmax]
        if not sm:
            return WeightedVoteFusion().combine(reports)
        num_classes = len(sm[0])
        log_prob = [0.0] * num_classes
        for s in sm:
            for k in range(num_classes):
                log_prob[k] += math.log(max(s[k], 1e-12))
        max_lp = max(log_prob)
        winners = [k for k in range(num_classes) if abs(log_prob[k] - max_lp) < 1e-9]
        return winners[0] if len(winners) == 1 else None


# ---------- 仲裁器 ----------
class Arbiter:
    """多节点冲突仲裁器。

    流程：receive → detect_conflicts → arbitrate → dispatch_and_rollback → stats
    """

    def __init__(self, fusion: Fusion = None,
                 conf_threshold: float = CONFLICT_CONF_THRESHOLD):
        self.fusion = fusion or WeightedVoteFusion()
        self.conf_threshold = conf_threshold
        self.reports: List[Report] = []
        self.conflicts: List[Conflict] = []
        self.decisions: dict = {}          # target_id -> final_pred 或 None
        self.rollbacks: List[RollbackEvent] = []
        self.total_overlaps = 0            # 重叠观测组数（同 target 多节点）

    def receive(self, reports: List[Report]):
        """收多节点上报。"""
        self.reports.extend(reports)

    def _group_by_target(self):
        groups = {}
        for r in self.reports:
            groups.setdefault(r.target_id, []).append(r)
        return groups

    def detect_conflicts(self):
        """检测冲突：同 target 多节点 + 预测不一 + 双方 conf>0.7。"""
        groups = self._group_by_target()
        self.conflicts = []
        for target_id, reps in groups.items():
            if len(reps) < 2:
                continue
            self.total_overlaps += 1
            # 双方 conf>0.7 且预测不一才算冲突
            high_conf = [r for r in reps if r.conf > self.conf_threshold]
            high_preds = {r.pred for r in high_conf}
            if len(high_preds) >= 2:
                self.conflicts.append(Conflict(target_id=target_id, reports=reps))
        return self.conflicts

    def arbitrate(self):
        """仲裁：对每个冲突调融合策略出 final_pred（平局返回 None）。"""
        self.decisions = {}
        for c in self.conflicts:
            self.decisions[c.target_id] = self.fusion.combine(c.reports)
        return self.decisions

    def dispatch_and_rollback(self):
        """下发 + 回滚：pred≠final 的节点改到 final，记回滚事件。平局冲突记未解决。"""
        self.rollbacks = []
        for c in self.conflicts:
            final = self.decisions.get(c.target_id)
            for r in c.reports:
                if final is None:
                    self.rollbacks.append(RollbackEvent(
                        r.node_id, c.target_id, r.pred, -1, resolved=False))
                elif r.pred != final:
                    self.rollbacks.append(RollbackEvent(
                        r.node_id, c.target_id, r.pred, final, resolved=True))
        return self.rollbacks

    def stats(self):
        """统计冲突率/解决率。"""
        num_conflicts = len(self.conflicts)
        num_overlaps = max(self.total_overlaps, 1)
        resolved = sum(1 for c in self.conflicts
                       if self.decisions.get(c.target_id) is not None)
        return {
            "total_overlaps": self.total_overlaps,
            "num_conflicts": num_conflicts,
            "num_rollbacks": len(self.rollbacks),
            "conflict_rate": num_conflicts / num_overlaps,
            "resolve_rate": resolved / num_conflicts if num_conflicts else 1.0,
        }


# ---------- 模拟自测 ----------
def _make_synthetic_reports(num_nodes=5, num_targets=80, num_classes=16,
                            error_rate=0.25, seed=42):
    """造多节点上报：每目标被 1-3 节点观测；error_rate 比例的节点预测错误。

    error_rate 控制冲突率（错误越多冲突越多）。模拟用，真实冲突率要等思路B。
    """
    rnd = random.Random(seed)
    reports = []
    for tid in range(num_targets):
        n_obs = rnd.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        true_pred = rnd.randint(0, num_classes - 1)
        for _ in range(n_obs):
            node_id = rnd.randint(0, num_nodes - 1)
            if rnd.random() < (1 - error_rate):
                pred = true_pred
                conf = rnd.uniform(0.78, 0.95)
            else:
                pred = (true_pred + rnd.randint(1, num_classes - 1)) % num_classes
                conf = rnd.uniform(0.55, 0.85)
            # 完整 softmax（贝叶斯用）：Top-1=conf，其余均分
            sm = [0.0] * num_classes
            sm[pred] = conf
            rest = (1 - conf) / (num_classes - 1)
            for k in range(num_classes):
                if k != pred:
                    sm[k] = rest
            reports.append(Report(node_id, tid, pred, conf, sm))
    return reports


def _run(fusion_name, fusion, reports):
    arb = Arbiter(fusion=fusion)
    arb.receive(reports)
    arb.detect_conflicts()
    arb.arbitrate()
    arb.dispatch_and_rollback()
    s = arb.stats()
    print(f"=== {fusion_name} ===")
    print(f"  重叠观测组数: {s['total_overlaps']}")
    print(f"  冲突数: {s['num_conflicts']}  冲突率: {s['conflict_rate']:.2%}  (硬指标≤5%)")
    print(f"  回滚数: {s['num_rollbacks']}")
    print(f"  解决率: {s['resolve_rate']:.2%}  (硬指标≥90%)")
    print()
    return s


def main():
    import argparse
    p = argparse.ArgumentParser(description="arbiter 模拟自测")
    p.add_argument("--error-rate", type=float, default=0.25, help="节点预测错误率，控制冲突率")
    p.add_argument("--num-targets", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(f"arbiter 模拟自测（error_rate={args.error_rate}, 纯逻辑，不依赖 torch/真模型）\n")
    reports = _make_synthetic_reports(num_targets=args.num_targets, error_rate=args.error_rate, seed=args.seed)
    s_a = _run("WeightedVoteFusion (加权投票)", WeightedVoteFusion(), reports)
    s_b = _run("BayesianFusion (贝叶斯产品融合)", BayesianFusion(), reports)
    print("--- 对比 ---")
    print(f"加权投票: 冲突率 {s_a['conflict_rate']:.2%}  解决率 {s_a['resolve_rate']:.2%}")
    print(f"贝叶斯  : 冲突率 {s_b['conflict_rate']:.2%}  解决率 {s_b['resolve_rate']:.2%}")
    print("\n注：冲突率由造数据 error_rate 控制（模拟）；真实冲突率要等思路B接主循环。")
    print("    解决率反映 arbiter 能力（平局未解决会拉低）。")


if __name__ == "__main__":
    main()
