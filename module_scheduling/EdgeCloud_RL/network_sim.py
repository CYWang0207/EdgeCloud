from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


__all__ = [
    "AsyncPacket",
    "DEFAULT_ACC_FLOOR",
    "NetworkSimulator",
    "mb_per_slot_to_mbps",
    "mbps_to_mb_per_slot",
]


DEFAULT_ACC_FLOOR = 0.8


MARKOV_TRANSITIONS = {
    "GOOD": (("GOOD", 0.92), ("WEAK", 0.07), ("DOWN", 0.01)),
    "WEAK": (("GOOD", 0.15), ("WEAK", 0.75), ("DOWN", 0.10)),
    "DOWN": (("GOOD", 0.30), ("WEAK", 0.20), ("DOWN", 0.50)),
}


@dataclass
class AsyncPacket:
    remaining_mb: float
    created_slot: int
    ready_slot: int
    enqueue_slot: int
    task_type: str = "scl_weights"
    payload_mb: float = 0.0
    accounting_comm_mb: float = 0.0

    @property
    def size_mb(self) -> float:
        return self.remaining_mb

    @size_mb.setter
    def size_mb(self, value: float) -> None:
        self.remaining_mb = value


def mbps_to_mb_per_slot(mbps: float, slot_duration: float) -> float:
    """Convert Mbps to MB per scheduling slot."""
    return float(mbps) * float(slot_duration) / 8.0


def mb_per_slot_to_mbps(mb_per_slot: float, slot_duration: float) -> float:
    """Convert MB per scheduling slot to Mbps."""
    if slot_duration <= 0:
        raise ValueError("slot_duration must be positive.")
    return float(mb_per_slot) * 8.0 / float(slot_duration)


class NetworkSimulator:
    """Standalone network fluctuation simulator for edge-cloud scheduling.

    The simulator follows the design document:
    - sample bandwidth jitter, packet loss, and outage states;
    - filter infeasible candidates under outage or strict bandwidth limits;
    - compute communication delay and end-to-end delay by action type;
    - maintain both Y_bw and ready-only Q_net, with Q_net_max and TTL protection;
    - evaluate business availability under the continuity target.
    """

    def __init__(
        self,
        mode: str = "static",
        slot_duration: float = 0.2,
        b_avg: float = 3.0,
        bandwidth_min_mbps: float = 20.0,
        bandwidth_max_mbps: float = 120.0,
        good_bandwidth_min_mbps: float = 40.0,
        good_bandwidth_max_mbps: float = 120.0,
        weak_bandwidth_min_mbps: float = 2.0,
        weak_bandwidth_max_mbps: float = 20.0,
        disconnect_prob: float = 0.0,
        outage_period: int = 0,
        outage_duration: int = 0,
        loss_rate_min: float = 0.0,
        loss_rate_max: float = 0.02,
        good_loss_rate_min: float = 0.0,
        good_loss_rate_max: float = 0.02,
        weak_loss_rate_min: float = 0.05,
        weak_loss_rate_max: float = 0.20,
        down_loss_rate: float = 1.0,
        rtt_ms: float = 10.0,
        edge_delay_ms: float = 80.0,
        adapter_cloud_delay_ms: float = 30.0,
        retrain_cloud_delay_ms: float = 0.0,
        adapter_size_mb: float = 1.2,
        query_size_mb: float = 5.0,
        foreground_query_size_mb: float = 0.0,
        include_adapter_in_foreground: bool = False,
        u2_update_size_mb: float = 50.0,
        scl_weights_mb: Optional[float] = None,
        scl_weights: Optional[float] = None,
        deadline_ms: float = 200.0,
        acc_floor: float = DEFAULT_ACC_FLOOR,
        business_min_active_views: int = 3,
        overflow_penalty: float = 5.0,
        strict_bandwidth: bool = False,
        sync_u2: bool = False,
        q_net_max: Optional[float] = None,
        q_net_max_mb: Optional[float] = None,
        queue_ttl_slots: int = 50,
        queue_policy: str = "drop_oldest",
        queue_overflow_policy: Optional[str] = None,
        enable_queue_force_local: bool = False,
        q_net_threshold_ratio: float = 0.8,
        force_local_slots: int = 1,
        loss_failure_threshold: float = 0.5,
        seed: Optional[int] = None,
        markov_initial_state: str = "GOOD",
        markov_transitions: Optional[Mapping[str, Sequence[Tuple[str, float]]]] = None,
    ) -> None:
        self.mode = mode
        self.slot_duration = float(slot_duration)
        self.b_avg = float(b_avg)
        self.bandwidth_min_mbps = float(bandwidth_min_mbps)
        self.bandwidth_max_mbps = float(bandwidth_max_mbps)
        self.good_bandwidth_min_mbps = float(good_bandwidth_min_mbps)
        self.good_bandwidth_max_mbps = float(good_bandwidth_max_mbps)
        self.weak_bandwidth_min_mbps = float(weak_bandwidth_min_mbps)
        self.weak_bandwidth_max_mbps = float(weak_bandwidth_max_mbps)
        self.disconnect_prob = float(disconnect_prob)
        self.outage_period = int(outage_period)
        self.outage_duration = int(outage_duration)
        self.loss_rate_min = float(loss_rate_min)
        self.loss_rate_max = float(loss_rate_max)
        self.good_loss_rate_min = float(good_loss_rate_min)
        self.good_loss_rate_max = float(good_loss_rate_max)
        self.weak_loss_rate_min = float(weak_loss_rate_min)
        self.weak_loss_rate_max = float(weak_loss_rate_max)
        self.down_loss_rate = float(down_loss_rate)
        self.rtt_ms = float(rtt_ms)
        self.edge_delay_ms = float(edge_delay_ms)
        self.adapter_cloud_delay_ms = float(adapter_cloud_delay_ms)
        self.retrain_cloud_delay_ms = float(retrain_cloud_delay_ms)
        self.adapter_size_mb = float(adapter_size_mb)
        self.query_size_mb = float(query_size_mb)
        self.foreground_query_size_mb = float(foreground_query_size_mb)
        self.include_adapter_in_foreground = bool(include_adapter_in_foreground)
        if scl_weights_mb is not None:
            u2_update_size_mb = scl_weights_mb
        if scl_weights is not None:
            u2_update_size_mb = scl_weights
        self.u2_update_size_mb = float(u2_update_size_mb)
        self.deadline_ms = float(deadline_ms)
        self.acc_floor = float(acc_floor)
        self.business_min_active_views = int(business_min_active_views)
        self.overflow_penalty = float(overflow_penalty)
        self.strict_bandwidth = bool(strict_bandwidth)
        self.sync_u2 = bool(sync_u2)
        q_net_limit = q_net_max_mb if q_net_max_mb is not None else q_net_max
        if q_net_limit is None:
            largest_background_packet = max(self.adapter_size_mb, self.u2_update_size_mb)
            q_net_limit = max(4.0 * self.b_avg, largest_background_packet)
        self.q_net_max = float(q_net_limit)
        self.queue_ttl_slots = int(queue_ttl_slots)
        self.queue_policy = queue_overflow_policy or queue_policy
        self.enable_queue_force_local = bool(enable_queue_force_local)
        self.q_net_threshold_ratio = float(q_net_threshold_ratio)
        self.q_net_threshold = self.q_net_threshold_ratio * self.q_net_max
        self.force_local_slots = int(force_local_slots)
        self.loss_failure_threshold = float(loss_failure_threshold)
        self.rng = np.random.default_rng(seed)
        self.markov_state = markov_initial_state
        self.markov_transitions = markov_transitions or MARKOV_TRANSITIONS

        self.t = 0
        self.y_bw = 0.0
        self.q_net = 0.0
        self.async_queue: List[AsyncPacket] = []
        self.total_enqueued_comm = 0.0
        self.dropped_comm = 0.0
        self.enqueued_by_type: Dict[str, float] = {}
        self.dropped_by_type: Dict[str, float] = {}
        self.completed_by_type: Dict[str, float] = {}
        self.force_local_mode = False
        self.force_local_remaining = 0
        self.current_state: Optional[Dict[str, Any]] = None

        self._validate()

    def step(self) -> Dict[str, Any]:
        """Sample one network state and return R_t, B_t, state, and outage flags."""
        network_state, bandwidth_mbps, loss_rate, is_disconnected = self._sample_state()

        if is_disconnected:
            bandwidth_mbps = 0.0
            loss_rate = self.down_loss_rate

        effective_bandwidth_mbps = max(bandwidth_mbps * (1.0 - loss_rate), 0.0)
        b_t = mbps_to_mb_per_slot(effective_bandwidth_mbps, self.slot_duration)

        state = {
            "t": self.t,
            "mode": self.mode,
            "network_state": network_state,
            "bandwidth_mbps": float(bandwidth_mbps),
            "loss_rate": float(loss_rate),
            "effective_bandwidth_mbps": float(effective_bandwidth_mbps),
            "slot_duration": self.slot_duration,
            "B_t": float(b_t),
            "raw_B_t": mbps_to_mb_per_slot(bandwidth_mbps, self.slot_duration),
            "disconnect_flag": bool(is_disconnected),
            "is_disconnected": bool(is_disconnected),
            "rtt_ms": self.rtt_ms,
        }

        self.current_state = state
        self.t += 1

        if self.mode == "markov":
            self.markov_state = self._next_markov_state(network_state)

        return dict(state)

    def filter_candidates(
        self,
        candidates: Iterable[Any],
        c_comm_map: Union[Mapping[Any, float], Callable[[Any], float]],
    ) -> Dict[str, Any]:
        """Filter candidates under outage, force-local, and strict bandwidth rules."""
        state = self._require_state()
        filtered = []
        logs = []
        forced_local = (
            state["disconnect_flag"]
            or self.force_local_mode
            or self.force_local_remaining > 0
        )

        for candidate in candidates:
            u = self.get_candidate_u(candidate)
            c_comm = self.get_candidate_comm(candidate, c_comm_map)
            realtime_comm = self.realtime_comm_mb(u, c_comm)
            penalty = self.compute_network_penalty(realtime_comm)
            comm_overflow = penalty["comm_overflow"]
            overflow_ratio = penalty["overflow_ratio"]
            network_penalty = penalty["network_penalty"]

            filtered_out = False
            reason = ""
            if forced_local and u != 0:
                filtered_out = True
                reason = "force_local"
            elif self.strict_bandwidth and u != 0 and comm_overflow > 0:
                filtered_out = True
                reason = "strict_bandwidth"

            if not filtered_out:
                filtered.append(candidate)

            logs.append(
                {
                    "candidate": candidate,
                    "u": int(u),
                    "c_comm": float(c_comm),
                    "realtime_comm": float(realtime_comm),
                    "comm_overflow": float(comm_overflow),
                    "overflow_ratio": float(overflow_ratio),
                    "network_penalty": float(network_penalty),
                    "filtered": bool(filtered_out),
                    "filter_reason": reason,
                }
            )

        if self.force_local_remaining > 0:
            self.force_local_remaining -= 1

        return {
            "candidates": filtered,
            "candidate_logs": logs,
            "feasible_candidate_count": len(filtered),
            "forced_local": bool(forced_local),
            "B_t": state["B_t"],
            "network_state": state["network_state"],
            "disconnect_flag": state["disconnect_flag"],
        }

    def compute_e2e(
        self,
        u: int,
        realtime_comm: float,
        t_edge: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return communication delay, cloud delay, e2e delay, and success flags."""
        state = self._require_state()
        edge_delay_ms = self.edge_delay_ms if t_edge is None else float(t_edge)

        if realtime_comm <= 0:
            comm_delay = 0.0
            transmission_success = True
        elif state["disconnect_flag"] or state["effective_bandwidth_mbps"] <= 0:
            comm_delay = math.inf
            transmission_success = False
        else:
            comm_delay = (
                8.0 * float(realtime_comm) / state["effective_bandwidth_mbps"] * 1000.0
                + self.rtt_ms
            )
            transmission_success = state["loss_rate"] < self.loss_failure_threshold

        cloud_delay = self.cloud_delay_ms(u)
        e2e_delay = edge_delay_ms + comm_delay + cloud_delay
        deadline_met = bool(e2e_delay <= self.deadline_ms)

        return {
            "u": int(u),
            "realtime_comm": float(realtime_comm),
            "comm_delay_ms": float(comm_delay),
            "cloud_delay_ms": float(cloud_delay),
            "edge_delay_ms": float(edge_delay_ms),
            "e2e_delay_ms": float(e2e_delay),
            "deadline_ms": self.deadline_ms,
            "deadline_met": deadline_met,
            "transmission_success": bool(transmission_success),
            "loss_rate": state["loss_rate"],
            "effective_bandwidth_mbps": state["effective_bandwidth_mbps"],
        }

    def compute_network_penalty(
        self,
        realtime_comm: float,
        B_t: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute overflow penalty after Critic returns G_raw."""
        state = self._require_state()
        capacity = state["B_t"] if B_t is None else float(B_t)
        comm_overflow = max(float(realtime_comm) - capacity, 0.0)
        overflow_ratio = comm_overflow / max(capacity, 1e-6)
        network_penalty = self.overflow_penalty * overflow_ratio
        return {
            "comm_overflow": float(comm_overflow),
            "overflow_ratio": float(overflow_ratio),
            "network_penalty": float(network_penalty),
        }

    def apply_network_penalty(
        self,
        G_raw: float,
        realtime_comm: float,
        B_t: Optional[float] = None,
    ) -> Dict[str, float]:
        """Return G_effective = G_raw - network_penalty."""
        penalty = self.compute_network_penalty(realtime_comm, B_t)
        g_effective = float(G_raw) - penalty["network_penalty"]
        return {
            "G_raw": float(G_raw),
            "G_effective": float(g_effective),
            **penalty,
        }

    def comm_cost_mb(self, u: Optional[int]) -> float:
        """Return c_comm from the scheme's u-dependent communication accounting."""
        if u is None or u == 0:
            return 0.0
        if u == 1:
            return self.query_size_mb + self.adapter_size_mb
        if u == 2:
            return self.query_size_mb + self.u2_update_size_mb
        raise ValueError(f"Unsupported u: {u}")

    def enqueue_job(
        self,
        task_type: str,
        payload_mb: float,
        enqueue_slot: Optional[int] = None,
        ready_slot: Optional[int] = None,
        created_slot: Optional[int] = None,
        accounting_comm_mb: Optional[float] = None,
    ) -> None:
        """Add an async adapter or SCL weights packet to the pending/ready queue."""
        if payload_mb <= 0:
            return
        state = self._require_state()
        ready = state["t"] if ready_slot is None else int(ready_slot)
        slot = ready if enqueue_slot is None else int(enqueue_slot)
        created = state["t"] if created_slot is None else int(created_slot)
        payload = float(payload_mb)
        accounting_comm = payload if accounting_comm_mb is None else float(accounting_comm_mb)
        packet = AsyncPacket(
            remaining_mb=payload,
            created_slot=created,
            ready_slot=ready,
            enqueue_slot=slot,
            task_type=task_type,
            payload_mb=payload,
            accounting_comm_mb=accounting_comm,
        )
        self.async_queue.append(packet)
        self._record_enqueued(task_type, packet.remaining_mb)
        self._refresh_q_net(state["t"])

    def schedule_ready_downloads(
        self,
        B_t: Optional[float] = None,
        prefer: str = "adapter",
    ) -> Dict[str, Any]:
        """Schedule ready async packets with adapter packets preferred."""
        state = self._require_state()
        capacity = state["B_t"] if B_t is None else float(B_t)
        scheduled_comm, served_by_type, completed_events = self._drain_async_queue(
            capacity,
            prefer=prefer,
            current_slot=state["t"],
        )

        return {
            "scheduled_payload_comm": float(scheduled_comm),
            "scheduled_download_comm": float(scheduled_comm),
            "scheduled_by_type": dict(served_by_type),
            "completed_events": completed_events,
            "completed_event_count": len(completed_events),
            "completed_accounting_comm_mb": float(
                sum(event["accounting_comm_mb"] for event in completed_events)
            ),
        }

    def update_queues(
        self,
        c_comm: Optional[float] = None,
        b_avg: Optional[float] = None,
        u: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update Y_bw and Q_net with Q_net_max and TTL protection."""
        state = self._require_state()
        if c_comm is None:
            c_comm = self.comm_cost_mb(u)
        c_comm = float(c_comm)
        b_avg = self.b_avg if b_avg is None else float(b_avg)

        self._refresh_q_net(state["t"])
        queue_before = self.q_net
        y_before = self.y_bw
        self.y_bw = max(self.y_bw + c_comm - b_avg, 0.0)

        realtime_comm = self.realtime_comm_mb(u, c_comm) if u is not None else 0.0
        background_comm = self.background_comm_mb(u, c_comm) if u is not None else c_comm
        background_task_type = self.background_task_type(u)
        remaining_capacity = max(state["B_t"] - realtime_comm, 0.0)

        if background_comm > 0:
            ready_slot = state["t"] + self.background_ready_delay_slots(u)
            self.enqueue_job(
                background_task_type,
                background_comm,
                enqueue_slot=ready_slot,
                ready_slot=ready_slot,
                created_slot=state["t"],
                accounting_comm_mb=c_comm,
            )

        self._refresh_q_net(state["t"])
        q_raw = self.q_net
        ttl_drop_count, ttl_drop_mb = self._drop_expired_packets(state["t"])
        cap_drop_count, cap_drop_mb = self._enforce_queue_limit(state["t"])
        q_after_protection = self._refresh_q_net(state["t"])

        schedule = self.schedule_ready_downloads(remaining_capacity, prefer="adapter")
        served_comm = schedule["scheduled_payload_comm"]
        scheduled_download_comm = schedule["scheduled_download_comm"]
        scheduled_by_type = schedule["scheduled_by_type"]
        completed_events = schedule["completed_events"]

        if self.enable_queue_force_local:
            if q_raw >= self.q_net_max:
                self.force_local_mode = True
            elif self.q_net <= self.q_net_threshold:
                self.force_local_mode = False

        queue_after = self.q_net
        queue_drop_count = ttl_drop_count + cap_drop_count
        queue_drop_mb = ttl_drop_mb + cap_drop_mb
        force_local_due_to_queue = self.force_local_mode or self.force_local_remaining > 0

        return {
            "Y_bw_before": float(y_before),
            "Y_bw": float(self.y_bw),
            "Y_bw_after": float(self.y_bw),
            "Q_net_before": float(queue_before),
            "Q_net": float(queue_after),
            "Q_net_after": float(queue_after),
            "Q_net_max": float(self.q_net_max),
            "Q_net_threshold": float(self.q_net_threshold),
            "Q_raw": float(q_raw),
            "Q_net_after_protection": float(q_after_protection),
            "queue_before": float(queue_before),
            "queue_after": float(queue_after),
            "ready_comm": float(queue_after),
            "pending_comm": float(self._pending_comm(state["t"])),
            "background_comm": float(background_comm),
            "background_task_type": background_task_type,
            "realtime_comm": float(realtime_comm),
            "remaining_capacity": float(remaining_capacity),
            "served_comm": float(served_comm),
            "scheduled_payload_comm": float(served_comm),
            "scheduled_download_comm": float(scheduled_download_comm),
            "scheduled_by_type": dict(scheduled_by_type),
            "adapter_scheduled_mb": float(scheduled_by_type.get("adapter", 0.0)),
            "scl_weights_scheduled_mb": float(scheduled_by_type.get("scl_weights", 0.0)),
            "completed_events": completed_events,
            "completed_event_count": len(completed_events),
            "completed_accounting_comm_mb": float(
                sum(event["accounting_comm_mb"] for event in completed_events)
            ),
            "queue_drop_count": int(queue_drop_count),
            "queue_drop_mb": float(queue_drop_mb),
            "dropped_comm": float(self.dropped_comm),
            "drop_ratio": self.drop_ratio(),
            "adapter_drop_ratio": self.drop_ratio("adapter"),
            "adapter_completion_rate": self.completion_rate("adapter"),
            "u2_update_completion_rate": self.completion_rate("scl_weights"),
            "scl_weights_completion_rate": self.completion_rate("scl_weights"),
            "ttl_expired_count": int(ttl_drop_count),
            "expired_update_count": int(ttl_drop_count),
            "ttl_expired_mb": float(ttl_drop_mb),
            "cap_drop_count": int(cap_drop_count),
            "cap_drop_mb": float(cap_drop_mb),
            "force_local_due_to_queue": bool(force_local_due_to_queue),
            "force_local_mode": bool(self.force_local_mode),
        }

    def is_business_available(
        self,
        decision_success: bool,
        e2e_ms: float,
        active_views: int,
        proxy_acc: float,
        transmission_success: bool = True,
    ) -> bool:
        """Evaluate one slot using the Critic quality proxy_acc floor."""
        return bool(
            decision_success
            and transmission_success
            and float(e2e_ms) <= self.deadline_ms
            and int(active_views) >= self.business_min_active_views
            and float(proxy_acc) >= self.acc_floor
        )

    def realtime_comm_mb(self, u: Optional[int], c_comm: float) -> float:
        realtime_comm = 0.0
        if u == 1:
            realtime_comm += self.foreground_query_size_mb
            if self.include_adapter_in_foreground:
                realtime_comm += self.adapter_size_mb
        if u == 2 and self.sync_u2:
            realtime_comm += float(c_comm)
        return float(realtime_comm)

    def background_comm_mb(self, u: Optional[int], c_comm: float) -> float:
        if u == 1:
            return 0.0 if self.include_adapter_in_foreground else self.adapter_size_mb
        if u == 2 and not self.sync_u2:
            return self.u2_update_size_mb
        return 0.0

    def background_ready_delay_slots(self, u: Optional[int]) -> int:
        if u == 1:
            delay_ms = self.adapter_cloud_delay_ms
        elif u == 2 and not self.sync_u2:
            delay_ms = self.retrain_cloud_delay_ms
        else:
            delay_ms = 0.0
        slot_ms = self.slot_duration * 1000.0
        return int(math.ceil(max(delay_ms, 0.0) / slot_ms))

    def background_task_type(self, u: Optional[int]) -> str:
        if u == 1 and not self.include_adapter_in_foreground:
            return "adapter"
        if u == 2 and not self.sync_u2:
            return "scl_weights"
        return "none"

    def cloud_delay_ms(self, u: int) -> float:
        if u == 2 and self.sync_u2:
            return self.retrain_cloud_delay_ms
        return 0.0

    @staticmethod
    def business_continuity_rate(values: Sequence[bool]) -> float:
        if not values:
            return 0.0
        return float(np.mean(np.asarray(values, dtype=float)))

    @staticmethod
    def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        """Summarize slot records into business and delay metrics."""
        total_slots = len(records)
        if total_slots == 0:
            return {
                "total_slots": 0.0,
                "available_slots": 0.0,
                "business_success_rate": 0.0,
                "business_continuity_rate": 0.0,
                "avg_e2e_delay_ms": math.nan,
                "p95_e2e_delay_ms": math.nan,
                "avg_available_e2e_delay_ms": math.nan,
                "deadline_met_rate": 0.0,
                "transmission_success_rate": 0.0,
                "decision_success_rate": 0.0,
                "disconnect_rate": 0.0,
                "finite_e2e_delay_rate": 0.0,
            }

        business_flags = [bool(record.get("business_available", False)) for record in records]
        e2e_values = []
        available_e2e_values = []
        for record, available in zip(records, business_flags):
            try:
                e2e_ms = float(record.get("e2e_delay_ms", math.nan))
            except (TypeError, ValueError):
                e2e_ms = math.nan
            if math.isfinite(e2e_ms):
                e2e_values.append(e2e_ms)
                if available:
                    available_e2e_values.append(e2e_ms)

        def rate(key: str) -> float:
            return float(
                np.mean([bool(record.get(key, False)) for record in records])
            )

        def avg(values: Sequence[float]) -> float:
            return float(np.mean(values)) if values else math.nan

        def p95(values: Sequence[float]) -> float:
            return float(np.percentile(values, 95)) if values else math.nan

        available_slots = float(sum(business_flags))
        business_rate = available_slots / float(total_slots)
        return {
            "total_slots": float(total_slots),
            "available_slots": available_slots,
            "business_success_rate": float(business_rate),
            "business_continuity_rate": float(business_rate),
            "avg_e2e_delay_ms": avg(e2e_values),
            "p95_e2e_delay_ms": p95(e2e_values),
            "avg_available_e2e_delay_ms": avg(available_e2e_values),
            "deadline_met_rate": rate("deadline_met"),
            "transmission_success_rate": rate("transmission_success"),
            "decision_success_rate": rate("decision_success"),
            "disconnect_rate": rate("disconnect_flag"),
            "finite_e2e_delay_rate": len(e2e_values) / float(total_slots),
        }

    def drop_ratio(self, task_type: Optional[str] = None) -> float:
        if task_type is None:
            return self.dropped_comm / max(self.total_enqueued_comm, 1e-6)
        return self.dropped_by_type.get(task_type, 0.0) / max(
            self.enqueued_by_type.get(task_type, 0.0), 1e-6
        )

    def completion_rate(self, task_type: str) -> float:
        return self.completed_by_type.get(task_type, 0.0) / max(
            self.enqueued_by_type.get(task_type, 0.0), 1e-6
        )

    @staticmethod
    def get_candidate_u(candidate: Any) -> int:
        if isinstance(candidate, Mapping):
            return int(candidate["u"])
        if hasattr(candidate, "u"):
            return int(getattr(candidate, "u"))
        if isinstance(candidate, tuple) and len(candidate) >= 2:
            return int(candidate[1])
        raise ValueError(f"Cannot infer u from candidate: {candidate!r}")

    @staticmethod
    def get_candidate_comm(
        candidate: Any,
        c_comm_map: Union[Mapping[Any, float], Callable[[Any], float]],
    ) -> float:
        if callable(c_comm_map):
            return float(c_comm_map(candidate))

        if isinstance(candidate, Mapping) and "c_comm" in candidate:
            return float(candidate["c_comm"])

        try:
            return float(c_comm_map[candidate])
        except (KeyError, TypeError):
            pass

        u = NetworkSimulator.get_candidate_u(candidate)
        if u in c_comm_map:
            return float(c_comm_map[u])
        if str(u) in c_comm_map:
            return float(c_comm_map[str(u)])

        raise KeyError(f"Cannot find c_comm for candidate={candidate!r}, u={u}.")

    def _sample_state(self) -> Tuple[str, float, float, bool]:
        if self.mode == "static":
            bandwidth = mb_per_slot_to_mbps(self.b_avg, self.slot_duration)
            loss = self._uniform(self.loss_rate_min, self.loss_rate_min)
            return "STATIC", bandwidth, loss, False

        if self.mode == "jitter":
            bandwidth = self._uniform(self.bandwidth_min_mbps, self.bandwidth_max_mbps)
            loss = self._uniform(self.loss_rate_min, self.loss_rate_max)
            return "JITTER", bandwidth, loss, False

        if self.mode == "jitter_outage":
            if self._is_outage_slot():
                return "DOWN", 0.0, self.down_loss_rate, True
            bandwidth = self._uniform(self.bandwidth_min_mbps, self.bandwidth_max_mbps)
            loss = self._uniform(self.loss_rate_min, self.loss_rate_max)
            return "JITTER", bandwidth, loss, False

        if self.mode == "markov":
            state = self.markov_state
            if state == "DOWN":
                return "DOWN", 0.0, self.down_loss_rate, True
            if state == "WEAK":
                bandwidth = self._uniform(
                    self.weak_bandwidth_min_mbps, self.weak_bandwidth_max_mbps
                )
                loss = self._uniform(self.weak_loss_rate_min, self.weak_loss_rate_max)
                return "WEAK", bandwidth, loss, False
            bandwidth = self._uniform(
                self.good_bandwidth_min_mbps, self.good_bandwidth_max_mbps
            )
            loss = self._uniform(self.good_loss_rate_min, self.good_loss_rate_max)
            return "GOOD", bandwidth, loss, False

        raise ValueError(f"Unsupported network mode: {self.mode}")

    def _is_outage_slot(self) -> bool:
        random_outage = self.disconnect_prob > 0 and self.rng.random() < self.disconnect_prob
        periodic_outage = (
            self.outage_period > 0
            and self.outage_duration > 0
            and (self.t % self.outage_period) < self.outage_duration
        )
        return bool(random_outage or periodic_outage)

    def _next_markov_state(self, state: str) -> str:
        transitions = self.markov_transitions[state]
        names = [item[0] for item in transitions]
        probs = np.asarray([item[1] for item in transitions], dtype=float)
        probs = probs / probs.sum()
        return str(self.rng.choice(names, p=probs))

    def _drain_async_queue(
        self,
        capacity_mb: float,
        prefer: Optional[str] = None,
        current_slot: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
        state = self._require_state()
        slot = state["t"] if current_slot is None else int(current_slot)
        self._refresh_q_net(slot)
        remaining = max(float(capacity_mb), 0.0)
        served = 0.0
        served_by_type: Dict[str, float] = {}
        completed_events: List[Dict[str, Any]] = []

        while remaining > 0 and self.async_queue:
            packet_index = self._next_packet_index(prefer, slot)
            if packet_index is None:
                break
            packet = self.async_queue[packet_index]
            delta = min(packet.size_mb, remaining)
            packet.size_mb -= delta
            self.q_net -= delta
            served += delta
            served_by_type[packet.task_type] = served_by_type.get(packet.task_type, 0.0) + delta
            remaining -= delta
            if packet.size_mb <= 1e-9:
                finished = self.async_queue.pop(packet_index)
                self._record_completed(finished.task_type, delta)
                completed_events.append(self._build_completion_event(finished, slot))
            else:
                self._record_completed(packet.task_type, delta)

        self._refresh_q_net(slot)
        return served, served_by_type, completed_events

    def _next_packet_index(self, prefer: Optional[str], current_slot: int) -> Optional[int]:
        if prefer is None:
            for index, packet in enumerate(self.async_queue):
                if packet.ready_slot <= current_slot:
                    return index
            return None
        for index, packet in enumerate(self.async_queue):
            if packet.task_type == prefer and packet.ready_slot <= current_slot:
                return index
        for index, packet in enumerate(self.async_queue):
            if packet.ready_slot <= current_slot:
                return index
        return None

    def _drop_expired_packets(self, current_slot: int) -> Tuple[int, float]:
        if self.queue_ttl_slots <= 0:
            self._refresh_q_net(current_slot)
            return 0, 0.0

        drop_count = 0
        drop_mb = 0.0
        kept = []
        for packet in self.async_queue:
            is_ready = packet.ready_slot <= current_slot
            is_expired = current_slot - packet.enqueue_slot >= self.queue_ttl_slots
            if is_ready and is_expired:
                drop_count += 1
                drop_mb += packet.size_mb
                self._record_dropped(packet.task_type, packet.size_mb)
            else:
                kept.append(packet)

        if drop_count:
            self.async_queue = kept

        self._refresh_q_net(current_slot)
        return drop_count, drop_mb

    def _enforce_queue_limit(self, current_slot: Optional[int] = None) -> Tuple[int, float]:
        state = self._require_state()
        slot = state["t"] if current_slot is None else int(current_slot)
        self._refresh_q_net(slot)
        if self.q_net <= self.q_net_max:
            return 0, 0.0

        if self.enable_queue_force_local or self.queue_policy in {"force_local", "hybrid"}:
            self.force_local_mode = True
            self.force_local_remaining = max(self.force_local_remaining, self.force_local_slots)

        drop_count = 0
        drop_mb = 0.0
        while self.q_net > self.q_net_max:
            packet_index = self._next_packet_index(None, slot)
            if packet_index is None:
                break
            packet = self.async_queue.pop(packet_index)
            drop_mb += packet.size_mb
            drop_count += 1
            self._record_dropped(packet.task_type, packet.size_mb)
            self._refresh_q_net(slot)

        self._refresh_q_net(slot)
        return drop_count, drop_mb

    def _refresh_q_net(self, current_slot: Optional[int] = None) -> float:
        state = self._require_state()
        slot = state["t"] if current_slot is None else int(current_slot)
        self.q_net = sum(
            packet.remaining_mb
            for packet in self.async_queue
            if packet.ready_slot <= slot
        )
        self.q_net = max(float(self.q_net), 0.0)
        return self.q_net

    def _pending_comm(self, current_slot: Optional[int] = None) -> float:
        state = self._require_state()
        slot = state["t"] if current_slot is None else int(current_slot)
        return float(
            sum(
                packet.remaining_mb
                for packet in self.async_queue
                if packet.ready_slot > slot
            )
        )

    def _record_enqueued(self, task_type: str, size_mb: float) -> None:
        self.total_enqueued_comm += float(size_mb)
        self.enqueued_by_type[task_type] = (
            self.enqueued_by_type.get(task_type, 0.0) + float(size_mb)
        )

    def _record_dropped(self, task_type: str, size_mb: float) -> None:
        self.dropped_comm += float(size_mb)
        self.dropped_by_type[task_type] = (
            self.dropped_by_type.get(task_type, 0.0) + float(size_mb)
        )

    def _record_completed(self, task_type: str, size_mb: float) -> None:
        self.completed_by_type[task_type] = (
            self.completed_by_type.get(task_type, 0.0) + float(size_mb)
        )

    def _build_completion_event(self, packet: AsyncPacket, current_slot: int) -> Dict[str, Any]:
        task_latency_slots = max(current_slot - packet.created_slot + 1, 0)
        queue_latency_slots = max(current_slot - packet.enqueue_slot + 1, 0)
        ready_to_complete_slots = max(current_slot - packet.ready_slot + 1, 0)
        return {
            "task_type": packet.task_type,
            "created_slot": int(packet.created_slot),
            "ready_slot": int(packet.ready_slot),
            "enqueue_slot": int(packet.enqueue_slot),
            "completion_slot": int(current_slot),
            "payload_comm_mb": float(packet.payload_mb),
            "accounting_comm_mb": float(packet.accounting_comm_mb),
            "task_latency_slots": int(task_latency_slots),
            "task_latency_ms": float(task_latency_slots * self.slot_duration * 1000.0),
            "queue_latency_slots": int(queue_latency_slots),
            "queue_latency_ms": float(queue_latency_slots * self.slot_duration * 1000.0),
            "ready_to_complete_slots": int(ready_to_complete_slots),
            "ready_to_complete_ms": float(
                ready_to_complete_slots * self.slot_duration * 1000.0
            ),
        }

    def _require_state(self) -> Dict[str, Any]:
        if self.current_state is None:
            raise RuntimeError("Call step() before using this API.")
        return self.current_state

    def _uniform(self, low: float, high: float) -> float:
        low, high = sorted((float(low), float(high)))
        if math.isclose(low, high):
            return low
        return float(self.rng.uniform(low, high))

    def _validate(self) -> None:
        if self.slot_duration <= 0:
            raise ValueError("slot_duration must be positive.")
        if self.mode not in {"static", "jitter", "jitter_outage", "markov"}:
            raise ValueError(f"Unsupported network mode: {self.mode}")
        if self.markov_state not in self.markov_transitions:
            raise ValueError(f"Unknown markov_initial_state: {self.markov_state}")
        if self.q_net_max < 0:
            raise ValueError("q_net_max must be non-negative.")
        if not 0 <= self.q_net_threshold_ratio <= 1:
            raise ValueError("q_net_threshold_ratio must be in [0, 1].")
        if self.queue_policy not in {"drop_oldest", "force_local", "hybrid"}:
            raise ValueError("queue_policy must be drop_oldest, force_local, or hybrid.")
