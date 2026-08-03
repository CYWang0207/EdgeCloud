from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


MARKOV_TRANSITIONS = {
    "GOOD": (("GOOD", 0.92), ("WEAK", 0.07), ("DOWN", 0.01)),
    "WEAK": (("GOOD", 0.15), ("WEAK", 0.75), ("DOWN", 0.10)),
    "DOWN": (("GOOD", 0.30), ("WEAK", 0.20), ("DOWN", 0.50)),
}


@dataclass
class AsyncPacket:
    size_mb: float
    enqueue_slot: int
    kind: str = "u2_update"


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
    - maintain both Y_bw and Q_net, with Q_net_max and TTL protection;
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
        prompt_cloud_delay_ms: float = 30.0,
        retrain_cloud_delay_ms: float = 0.0,
        deadline_ms: float = 200.0,
        acc_floor: float = 0.0,
        business_min_active_views: int = 1,
        overflow_penalty: float = 5.0,
        strict_bandwidth: bool = False,
        sync_u2: bool = False,
        q_net_max: float = 100.0,
        queue_ttl_slots: int = 50,
        queue_policy: str = "drop_oldest",
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
        self.prompt_cloud_delay_ms = float(prompt_cloud_delay_ms)
        self.retrain_cloud_delay_ms = float(retrain_cloud_delay_ms)
        self.deadline_ms = float(deadline_ms)
        self.acc_floor = float(acc_floor)
        self.business_min_active_views = int(business_min_active_views)
        self.overflow_penalty = float(overflow_penalty)
        self.strict_bandwidth = bool(strict_bandwidth)
        self.sync_u2 = bool(sync_u2)
        self.q_net_max = float(q_net_max)
        self.queue_ttl_slots = int(queue_ttl_slots)
        self.queue_policy = queue_policy
        self.force_local_slots = int(force_local_slots)
        self.loss_failure_threshold = float(loss_failure_threshold)
        self.rng = np.random.default_rng(seed)
        self.markov_state = markov_initial_state
        self.markov_transitions = markov_transitions or MARKOV_TRANSITIONS

        self.t = 0
        self.y_bw = 0.0
        self.q_net = 0.0
        self.async_queue: List[AsyncPacket] = []
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
        forced_local = state["disconnect_flag"] or self.force_local_remaining > 0

        for candidate in candidates:
            u = self.get_candidate_u(candidate)
            c_comm = self.get_candidate_comm(candidate, c_comm_map)
            realtime_comm = self.realtime_comm_mb(u, c_comm)
            comm_overflow = max(realtime_comm - state["B_t"], 0.0)
            overflow_ratio = comm_overflow / max(state["B_t"], 1e-6)
            network_penalty = self.overflow_penalty * overflow_ratio

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

        if u == 2 and not self.sync_u2:
            realtime_comm = 0.0

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

    def update_queues(
        self,
        c_comm: float,
        b_avg: Optional[float] = None,
        u: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update Y_bw and Q_net with Q_net_max and TTL protection."""
        state = self._require_state()
        c_comm = float(c_comm)
        b_avg = self.b_avg if b_avg is None else float(b_avg)

        queue_before = self.q_net
        y_before = self.y_bw
        self.y_bw = max(self.y_bw + c_comm - b_avg, 0.0)

        realtime_comm = self.realtime_comm_mb(u, c_comm) if u is not None else 0.0
        background_comm = self.background_comm_mb(u, c_comm) if u is not None else c_comm
        remaining_capacity = max(state["B_t"] - realtime_comm, 0.0)

        if background_comm > 0:
            self.async_queue.append(AsyncPacket(background_comm, state["t"]))
            self.q_net += background_comm

        served_comm = self._drain_async_queue(remaining_capacity)
        ttl_drop_count, ttl_drop_mb = self._drop_expired_packets(state["t"])
        cap_drop_count, cap_drop_mb = self._enforce_queue_limit()

        queue_after = self.q_net
        queue_drop_count = ttl_drop_count + cap_drop_count
        queue_drop_mb = ttl_drop_mb + cap_drop_mb
        force_local_due_to_queue = self.force_local_remaining > 0

        return {
            "Y_bw_before": float(y_before),
            "Y_bw": float(self.y_bw),
            "Y_bw_after": float(self.y_bw),
            "Q_net_before": float(queue_before),
            "Q_net": float(queue_after),
            "Q_net_after": float(queue_after),
            "Q_net_max": float(self.q_net_max),
            "queue_before": float(queue_before),
            "queue_after": float(queue_after),
            "background_comm": float(background_comm),
            "realtime_comm": float(realtime_comm),
            "remaining_capacity": float(remaining_capacity),
            "served_comm": float(served_comm),
            "queue_drop_count": int(queue_drop_count),
            "queue_drop_mb": float(queue_drop_mb),
            "ttl_expired_count": int(ttl_drop_count),
            "ttl_expired_mb": float(ttl_drop_mb),
            "cap_drop_count": int(cap_drop_count),
            "cap_drop_mb": float(cap_drop_mb),
            "force_local_due_to_queue": bool(force_local_due_to_queue),
        }

    def is_business_available(
        self,
        decision_success: bool,
        e2e_ms: float,
        active_views: int,
        proxy_acc: float,
        transmission_success: bool = True,
    ) -> bool:
        """Evaluate the business availability conditions for one slot."""
        return bool(
            decision_success
            and transmission_success
            and float(e2e_ms) <= self.deadline_ms
            and int(active_views) >= self.business_min_active_views
            and float(proxy_acc) >= self.acc_floor
        )

    def realtime_comm_mb(self, u: Optional[int], c_comm: float) -> float:
        if u == 1:
            return float(c_comm)
        if u == 2 and self.sync_u2:
            return float(c_comm)
        return 0.0

    def background_comm_mb(self, u: Optional[int], c_comm: float) -> float:
        if u == 2 and not self.sync_u2:
            return float(c_comm)
        return 0.0

    def cloud_delay_ms(self, u: int) -> float:
        if u == 1:
            return self.prompt_cloud_delay_ms
        if u == 2 and self.sync_u2:
            return self.retrain_cloud_delay_ms
        return 0.0

    @staticmethod
    def business_continuity_rate(values: Sequence[bool]) -> float:
        if not values:
            return 0.0
        return float(np.mean(np.asarray(values, dtype=float)))

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

    def _drain_async_queue(self, capacity_mb: float) -> float:
        remaining = max(float(capacity_mb), 0.0)
        served = 0.0

        while remaining > 0 and self.async_queue:
            packet = self.async_queue[0]
            delta = min(packet.size_mb, remaining)
            packet.size_mb -= delta
            self.q_net -= delta
            served += delta
            remaining -= delta
            if packet.size_mb <= 1e-9:
                self.async_queue.pop(0)

        self.q_net = max(self.q_net, 0.0)
        return served

    def _drop_expired_packets(self, current_slot: int) -> Tuple[int, float]:
        if self.queue_ttl_slots <= 0:
            return 0, 0.0

        drop_count = 0
        drop_mb = 0.0
        kept = []
        for packet in self.async_queue:
            if current_slot - packet.enqueue_slot >= self.queue_ttl_slots:
                drop_count += 1
                drop_mb += packet.size_mb
            else:
                kept.append(packet)

        if drop_count:
            self.async_queue = kept
            self.q_net = max(self.q_net - drop_mb, 0.0)

        return drop_count, drop_mb

    def _enforce_queue_limit(self) -> Tuple[int, float]:
        if self.q_net <= self.q_net_max:
            return 0, 0.0

        if self.queue_policy in {"force_local", "hybrid"}:
            self.force_local_remaining = max(self.force_local_remaining, self.force_local_slots)

        drop_count = 0
        drop_mb = 0.0
        while self.q_net > self.q_net_max and self.async_queue:
            packet = self.async_queue.pop(0)
            self.q_net -= packet.size_mb
            drop_mb += packet.size_mb
            drop_count += 1

        self.q_net = min(max(self.q_net, 0.0), self.q_net_max)
        return drop_count, drop_mb

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
        if self.queue_policy not in {"drop_oldest", "force_local", "hybrid"}:
            raise ValueError("queue_policy must be drop_oldest, force_local, or hybrid.")


if __name__ == "__main__":
    sim = NetworkSimulator(mode="markov", seed=7)
    for _ in range(5):
        net = sim.step()
        delay = sim.compute_e2e(u=1, realtime_comm=7.0)
        queues = sim.update_queues(c_comm=7.0, u=1)
        print({**net, **delay, **queues})
